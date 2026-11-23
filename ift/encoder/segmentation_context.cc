#include "ift/encoder/segmentation_context.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "ift/common/font_helper.h"
#include "ift/common/int_set.h"
#include "ift/common/try.h"
#include "ift/encoder/activation_condition.h"
#include "ift/encoder/dependency_closure.h"
#include "ift/encoder/glyph_condition_set.h"
#include "ift/encoder/glyph_segmentation.h"
#include "ift/encoder/init_subset_defaults.h"
#include "ift/encoder/subset_definition.h"
#include "ift/encoder/types.h"

using ift::config::ConditionAnalysisMode;
using ift::config::DEP_GRAPH_ONLY_WITH_SIMPLIFICATION;

using absl::btree_map;
using absl::Status;
using absl::StatusOr;
using ift::common::DataFileResolver;
using ift::common::GlyphSet;
using ift::common::IntSet;
using ift::common::SegmentSet;

namespace ift::encoder {

Status SegmentationContext::CacheInitialGlyphProbabilities(
    absl::Span<const freq::ProbabilityCalculator* const>
        probability_calculators) {
  initial_glyph_probabilities_.clear();
  for (const freq::ProbabilityCalculator* calculator :
       probability_calculators) {
    calculator->ResetSegmentProbabilities(
        segmentation_info_->Segments().size());
  }

  for (const auto& [cond, gids] : glyph_groupings.ConditionsAndGlyphs()) {
    double max_prob = 0.0;
    for (const freq::ProbabilityCalculator* calculator :
         probability_calculators) {
      double prob =
          TRY(cond.Probability(segmentation_info_->Segments(), *calculator));
      if (prob > max_prob) {
        max_prob = prob;
      }
    }
    for (glyph_id_t gid : gids) {
      initial_glyph_probabilities_[gid] = max_prob;
    }
  }
  return absl::OkStatus();
}

btree_map<std::pair<ActivationCondition, SegmentSet>, GlyphSet>
SegmentationContext::InteractingSegmentsAndGlyphs() const {
  btree_map<std::pair<ActivationCondition, SegmentSet>, GlyphSet> result;

  const auto& interacting_map =
      dependency_closure_->AllGlyphInteractingSegments();
  for (const auto& [condition, glyphs] :
       glyph_groupings.ConditionsAndGlyphs()) {
    for (glyph_id_t gid : glyphs) {
      auto it = interacting_map.find(gid);
      if (it != interacting_map.end() && it->second.size() > 1) {
        result[{condition, it->second}].insert(gid);
      }
    }
  }
  return result;
}

Status SegmentationContext::ValidateSegmentation(
    const GlyphSegmentation& segmentation) const {
  GlyphSet visited;
  const auto& initial_closure = segmentation.InitialFontGlyphClosure();
  auto check_disjoint = [&](const GlyphSet& gids) -> Status {
    for (glyph_id_t gid : gids) {
      if (initial_closure.contains(gid)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Initial font glyph g", gid, " is present in a patch."));
      }
      if (visited.contains(gid)) {
        return absl::FailedPreconditionError(
            "Glyph segments are not disjoint.");
      }
      visited.insert(gid);
    }
    return absl::OkStatus();
  };

  for (const auto& [id, gids] : segmentation.GidSegments()) {
    TRYV(check_disjoint(gids));
  }

  if (target_mode_ == config::UNICODE_RANGE) {
    for (const auto& condition : segmentation.Conditions()) {
      if (!condition.IsExclusive()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Non-exclusive condition found in UNICODE_RANGE mode: ",
            condition.ToString()));
      }
    }

    // In unicode range mode not all conditions get record into segmentation, so
    // check disjointness across the non-exclusive patches which were skipped.
    GlyphSet unreachable_glyphs;
    for (const auto& cond : glyph_groupings.OrderedConditions()) {
      if (cond.IsExclusive()) {
        continue;
      }
      const GlyphSet& gids = glyph_groupings.ConditionsAndGlyphs().at(cond);
      TRYV(check_disjoint(gids));
      unreachable_glyphs.union_set(gids);
    }

    LogUnreachableGlyphs(unreachable_glyphs);
    LogSegmentCodepointCountHistogram(segmentation);
  }

  GlyphSet full_minus_initial = segmentation_info_->FullClosure();
  full_minus_initial.subtract(initial_closure);

  const GlyphSet& required_glyphs = TargetGlyphs();

  if (!required_glyphs.is_subset_of(visited)) {
    GlyphSet missing = required_glyphs;
    missing.subtract(visited);
    return absl::FailedPreconditionError(
        "Not all glyphs in the full closure have been placed. Missing: " +
        missing.ToString());
  }
  if (!visited.is_subset_of(full_minus_initial)) {
    GlyphSet extra = visited;
    extra.subtract(full_minus_initial);
    return absl::FailedPreconditionError(
        "Unexpected glyphs outside the full closure have been placed: " +
        extra.ToString());
  }

  for (const auto& condition : segmentation.Conditions()) {
    auto it = segmentation.GidSegments().find(condition.activated());
    if (it == segmentation.GidSegments().end()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Patch ", condition.activated(), " does not have glyphs specified"));
    }

    const auto& glyphs = it->second;
    for (const auto& sub_group : condition.conditions()) {
      if (TRY(glyph_closure_cache->HasAdditionalConditions(
              &SegmentationInfo(), sub_group, glyphs))) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Found a condition which does not satisfy the glyph closure "
            "requirement: ",
            condition.ToString(), " ", glyphs.ToString(), ", sub group ",
            sub_group.ToString(), " failed the check."));
      }
    }
  }

  return absl::OkStatus();
}

void SegmentationContext::LogUnreachableGlyphs(
    const GlyphSet& unreachable_glyphs) const {
  std::vector<std::pair<glyph_id_t, double>> ordered_glyphs;
  ordered_glyphs.reserve(unreachable_glyphs.size());
  for (glyph_id_t gid : unreachable_glyphs) {
    double initial_prob = 0.0;
    if (auto it = initial_glyph_probabilities_.find(gid);
        it != initial_glyph_probabilities_.end()) {
      initial_prob = it->second;
    }
    ordered_glyphs.emplace_back(gid, initial_prob);
  }

  std::sort(ordered_glyphs.begin(), ordered_glyphs.end(),
            [](const auto& a, const auto& b) {
              if (a.second != b.second) {
                return a.second > b.second;
              }
              return a.first < b.first;
            });

  auto gid_to_unicode = common::FontHelper::GidToUnicodeMap(original_face.get());
  for (const auto& [gid, initial_prob] : ordered_glyphs) {
    std::string cp_str;
    auto cp_it = gid_to_unicode.find(gid);
    if (cp_it != gid_to_unicode.end() && !cp_it->second.empty()) {
      if (cp_it->second.size() == 1) {
        cp_str = absl::StrFormat("U+%X, ", *cp_it->second.min());
      } else {
        std::vector<std::string> hex_cps;
        for (uint32_t cp : cp_it->second) {
          hex_cps.push_back(absl::StrFormat("U+%X", cp));
        }
        cp_str =
            absl::StrCat("{", absl::StrJoin(hex_cps, ", "), "}, ");
      }
    }

    std::string gid_str = absl::StrFormat("g%u, ", gid);
    std::string prob_str = absl::StrFormat("%.3g%%", initial_prob * 100.0);
    VLOG(0) << absl::StrFormat("Unreachable glyph %-8s%-10sP = %-9s",
                               gid_str, cp_str, prob_str);
  }
  VLOG(0) << "Total unreachable glyphs: " << unreachable_glyphs.size();
}

void SegmentationContext::LogSegmentCodepointCountHistogram(
    const GlyphSegmentation& segmentation) const {
  absl::btree_map<size_t, uint32_t> codepoint_count_histogram;
  for (const auto& segment : segmentation.Segments()) {
    if (segment.codepoints.empty()) {
      continue;
    }
    codepoint_count_histogram[segment.codepoints.size()]++;
  }

  std::stringstream histogram_string;
  histogram_string << "codepoint_count, segment_count" << std::endl;
  for (const auto& [codepoint_count, segment_count] :
       codepoint_count_histogram) {
    histogram_string << codepoint_count << ", " << segment_count << std::endl;
  }
  VLOG(0) << "Segment Codepoint Count Histogram:" << std::endl
          << histogram_string.str();
}

Status SegmentationContext::ReprocessChanged(InvalidationSet modified) {
  GlyphSet affected =
      dependency_closure_->SegmentsToAffectedGlyphs({modified.base_segment});
  affected.intersect(TargetGlyphs());
  modified.glyphs.union_set(affected);
  TransferDependencyGraphGlyphConditions(modified.glyphs);

  TRYV(GroupGlyphs(modified.glyphs, modified.segments));
  inert_segments_.subtract(modified.segments);
  if (IsSegmentInert(modified.base_segment)) {
    inert_segments_.insert(modified.base_segment);
  } else {
    inert_segments_.erase(modified.base_segment);
  }
  return absl::OkStatus();
}

Status SegmentationContext::ReprocessAll() {
  // Pull conditions directly out of the dep graph instead of running closure
  // processing.
  TransferDependencyGraphGlyphConditions(InitFontMoveTargetGlyphs());

  TRYV(GroupGlyphs(InitFontMoveTargetGlyphs(), {}));
  RefreshInertSegments();
  return absl::OkStatus();
}

Status SegmentationContext::ExcludeEmptyGlyphs() {
  if (target_mode_ == config::UNICODE_RANGE || exclude_empty_glyphs_) {
    return absl::OkStatus();
  }

  GlyphSet empty_gids = InitFontMoveTargetGlyphs();
  empty_gids.subtract(TargetGlyphs());
  exclude_empty_glyphs_ = true;
  if (empty_gids.empty()) {
    return absl::OkStatus();
  }

  glyph_condition_set.InvalidateGlyphInformation(empty_gids);
  TRYV(GroupGlyphs(empty_gids, {}));
  RefreshInertSegments();
  return absl::OkStatus();
}

Status SegmentationContext::MoveGlyphsToInitFont(const GlyphSet& glyphs) {
  if (glyphs.empty()) {
    return absl::OkStatus();
  }

  SegmentSet affected_segments;
  for (glyph_id_t gid : glyphs) {
    auto condition = glyph_groupings.GlyphToCondition(gid);
    if (condition.has_value()) {
      affected_segments.union_set(condition->TriggeringSegments());
    }
  }

  segmentation_info_->MoveGlyphsToInitFont(glyphs);
  RefreshUnicodeRangePatchSizeCaches();

  glyph_condition_set.InvalidateGlyphInformation(glyphs);
  TRYV(GroupGlyphs(glyphs, {}));

  for (segment_index_t s : affected_segments) {
    if (IsSegmentInert(s)) {
      inert_segments_.insert(s);
    } else {
      inert_segments_.erase(s);
    }
  }
  return absl::OkStatus();
}

namespace {

bool IsSatisfiedBySingleSegment(const ActivationCondition& condition,
                                segment_index_t segment_index) {
  for (const SegmentSet& clause : condition.conditions()) {
    if (!clause.contains(segment_index)) {
      return false;
    }
  }
  return !condition.conditions().empty();
}

}  // namespace

Status SegmentationContext::PromoteInitFontSegments(
    const SegmentSet& eligible_segments) {
  SegmentSet segments_in_moved_glyphs;
  const auto& all_glyph_conditions = dependency_closure_->AllGlyphConditions();
  for (glyph_id_t gid : segmentation_info_->InitFont().GlyphKeyedGlyphs()) {
    auto it = all_glyph_conditions.find(gid);
    if (it != all_glyph_conditions.end()) {
      segments_in_moved_glyphs.union_set(it->second.TriggeringSegments());
    }
  }

  SegmentSet candidates = eligible_segments;
  candidates.intersect(segments_in_moved_glyphs);

  const auto& segments = segmentation_info_->Segments();
  for (segment_index_t s : candidates) {
    const SubsetDefinition& seg_def = segments.at(s).Definition();
    if (seg_def.Empty()) {
      continue;
    }
    bool has_single_segment_patch = false;
    for (const ActivationCondition& c :
         glyph_groupings.TriggeringSegmentToConditions(s)) {
      if (IsSatisfiedBySingleSegment(c, s)) {
        has_single_segment_patch = true;
        break;
      }
    }
    if (!has_single_segment_patch) {
      AddToInitFontSegment(seg_def);
    }
  }

  SegmentSet unused_segments;
  for (segment_index_t s = 0; s < segments.size(); ++s) {
    const SubsetDefinition& seg_def = segments.at(s).Definition();
    if (!seg_def.Empty() &&
        segmentation_info_->InitFont().ContainsSegment(seg_def) &&
        glyph_groupings.TriggeringSegmentToConditions(s).empty()) {
      unused_segments.insert(s);
    }
  }
  return ClearUnusedInitSegments(unused_segments);
}

Status SegmentationContext::ResetSegments(std::vector<Segment> segments) {
  segmentation_info_->ResetSegments(std::move(segments));
  TRYV(dependency_closure_->SegmentsReset());
  uint32_t glyph_count = hb_face_get_glyph_count(original_face.get());
  glyph_condition_set = GlyphConditionSet(glyph_count);
  glyph_groupings = GlyphGroupings(
      glyph_count,
      condition_analysis_mode_ == config::DEP_GRAPH_ONLY_WITH_SIMPLIFICATION);
  inert_segments_.clear();
  return ReprocessAll();
}

StatusOr<SegmentationContext>
SegmentationContext::InitializeSegmentationContext(
    hb_face_t* face, SubsetDefinition initial_segment,
    std::vector<Segment> segments,
    ConditionAnalysisMode condition_analysis_mode, uint32_t brotli_quality,
    uint32_t init_font_brotli_quality,
    std::shared_ptr<DataFileResolver> resolver,
    ift::config::TargetMode target_mode, bool include_empty_glyphs) {
  if (!hb_face_get_glyph_count(face)) {
    return absl::InvalidArgumentError("Provided font has no glyphs.");
  }

  // The IFT compiler has a set of defaults always included in the initial font
  // add them here so we correctly factor them into the generated segmentation.
  AddInitSubsetDefaults(initial_segment);

  // No merging is done during init.
  SegmentationContext context = TRY(SegmentationContext::Create(
      face, initial_segment, segments, condition_analysis_mode, brotli_quality,
      init_font_brotli_quality, std::move(resolver), target_mode,
      include_empty_glyphs));

  // ### Generate the initial conditions and groupings by processing all
  // segments and glyphs. ###
  VLOG(0) << "Forming initial segmentation plan.";
  TRYV(context.ReprocessAll());
  GlyphSet always_true_glyphs = context.dependency_closure_->AlwaysTrueGlyphs();
  always_true_glyphs.intersect(context.TargetGlyphs());
  if (!always_true_glyphs.empty()) {
    VLOG(0) << "Moving " << always_true_glyphs.size()
            << " always-true glyphs into the initial font.";
    TRYV(context.MoveGlyphsToInitFont(always_true_glyphs));
  }

  return context;
}

void SegmentationContext::TransferDependencyGraphGlyphConditions(
    const GlyphSet& gids) {
  const auto& conditions = dependency_closure_->AllGlyphConditions();
  for (glyph_id_t g : gids) {
    ActivationCondition condition = conditions.at(g);
    if (condition.IsUnitary()) {
      condition = ActivationCondition::exclusive_segment(
          *condition.TriggeringSegments().begin(), 0);
    } else if (condition_analysis_mode_ ==
                   config::DEP_GRAPH_ONLY_WITH_SIMPLIFICATION &&
               !condition.IsPurelyConjunctive() &&
               !condition.IsPurelyDisjunctive()) {
      // In DEP_GRAPH_ONLY_WITH_SIMPLIFICATION mode mixed conditions are
      // simplified into a disjunctive superset.
      // TODO(garretrieger): it may be better to do this conversion during dep
      // graph construction, since it could it some cases reduce the cost of dep
      // graph construction signficantly.
      condition = condition.NonCompositeSuperset();
    }
    glyph_condition_set.SetCondition(g, condition);
  }
}

bool SegmentationContext::IsSegmentInert(segment_index_t s) const {
  if (segmentation_info_->Segments().at(s).Definition().Empty()) {
    return false;
  }
  for (const auto& c : glyph_groupings.TriggeringSegmentToConditions(s)) {
    if (!c.IsUnitary()) {
      return false;
    }
  }
  return true;
}

void SegmentationContext::RefreshInertSegments() {
  SegmentSet candidates = segmentation_info_->NonEmptySegments();
  for (const auto& [condition, _] : glyph_groupings.ConditionsAndGlyphs()) {
    if (!condition.IsUnitary()) {
      candidates.subtract(condition.TriggeringSegments());
    }
  }
  inert_segments_ = std::move(candidates);
}

}  // namespace ift::encoder