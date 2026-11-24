#include "ift/encoder/candidate_merge.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ift/common/compat_id.h"
#include "ift/common/font_data.h"
#include "ift/common/font_helper.h"
#include "ift/common/int_set.h"
#include "ift/common/try.h"
#include "ift/common/woff2.h"
#include "ift/encoder/activation_condition.h"
#include "ift/encoder/glyph_condition_set.h"
#include "ift/encoder/invalidation_set.h"
#include "ift/encoder/merger.h"
#include "ift/encoder/requested_segmentation_information.h"
#include "ift/encoder/segment.h"
#include "ift/encoder/segmentation_context.h"
#include "ift/encoder/subset_definition.h"
#include "ift/encoder/types.h"
#include "ift/freq/probability_calculator.h"
#include "ift/glyph_keyed_diff.h"

namespace ift::encoder {

using absl::flat_hash_map;
using absl::flat_hash_set;
using absl::Status;
using absl::StatusOr;
using ift::common::CodepointSet;
using ift::common::CompatId;
using ift::common::FontData;
using ift::common::FontHelper;
using ift::common::GlyphSet;
using ift::common::SegmentSet;
using ift::common::Woff2;

static double SumDeltas(absl::Span<double> deltas) {
  // Delta lists often come from hash containers which do not have stable
  // ordering. Since floating point addition is not associative we need
  // to first sort into a determnistic order or risk the sum result being
  // non-deterministic.
  std::sort(deltas.begin(), deltas.end(), [](double a, double b) {
    // Sort by magnitude (increasing) then sign to maximize accuracy of the sum.
    // It minimizes how many low order bits get rounded away when
    // each new term is added.
    double magnitude_a = std::abs(a);
    double magnitude_b = std::abs(b);
    if (magnitude_a != magnitude_b) {
      return magnitude_a < magnitude_b;
    }
    return a < b;
  });

  double total = 0.0;
  for (double delta : deltas) {
    total += delta;
  }
  return total;
}

static double SumDeltas(std::vector<double>& deltas) {
  return SumDeltas(absl::MakeSpan(deltas));
}

// Bounds the maximum possible floating-point rounding discrepancy between an
// unsorted running sum (current_delta) of up to max_terms elements and the
// final sorted sum produced by SumDeltas().
static double MaxSumError(size_t max_terms, double initial_magnitude_sum,
                          double initial_delta, double current_delta) {
  double magnitude_sum =
      initial_magnitude_sum + (current_delta - initial_delta);
  return 2.0 * max_terms * std::numeric_limits<double>::epsilon() *
         magnitude_sum;
}

StatusOr<bool> CandidateMerge::IsPatchTooSmall(
    Merger& merger, segment_index_t base_segment_index,
    const GlyphSet& glyphs) {
  uint32_t patch_size_bytes =
      TRY(merger.Context().patch_size_cache->GetPatchSize(glyphs));
  if (patch_size_bytes >= merger.Strategy().PatchSizeMinBytes()) {
    return false;
  }

  VLOG(0) << "Patch for segment " << base_segment_index << " is too small "
          << "(" << patch_size_bytes << " < "
          << merger.Strategy().PatchSizeMinBytes() << "). Merging...";

  return true;
}

StatusOr<InvalidationSet> CandidateMerge::Apply(Merger& merger) {
  if (!merged_segment_.has_value()) {
    TRYV(ApplyPatchMerge(merger));
    return InvalidationSet(base_segment_index_);
  }

  SegmentSet segments_to_merge_with_base = segments_to_merge_;
  segments_to_merge_with_base.insert(base_segment_index_);

  const auto& segments = merger.Context().SegmentationInfo().Segments();
  uint32_t size_before =
      segments[base_segment_index_].Definition().codepoints.size();
  uint32_t size_after =
      merger.AssignMergedSegment(base_segment_index_, segments_to_merge_,
                                 *merged_segment_, input_segments_are_inert_);

  VLOG(0) << "  Merged " << size_before << " codepoints up to " << size_after
          << " codepoints for segment " << base_segment_index_ << "."
          << std::endl
          << "  New patch size " << new_patch_size_ << " bytes. " << std::endl
          << "  Cost delta is " << cost_delta_ << "." << std::endl
          << "  New average probability is "
          << (TRY(merger.AggregateProbability(base_segment_index_)) /
              merger.MaxAggregateProbability());

  // Regardless of wether the new segment is inert all of the information
  // associated with the segments removed by the merge should be removed.
  TRYV(merger.Context().InvalidateGlyphInformationForMerge(
      invalidated_glyphs_, segments_to_merge_with_base, base_segment_index_));

  return InvalidationSet(invalidated_glyphs_, segments_to_merge_,
                         base_segment_index_);
}

Status CandidateMerge::ApplyPatchMerge(Merger& merger) {
  if (!patch_merge_target_conditions_.has_value() ||
      !patch_merge_glyphs_.has_value()) {
    return absl::InternalError(
        "patch_merge_target_conditions_/patch_merge_glyphs_ is not set for "
        "patch merge");
  }

  VLOG(0) << "  Merged patches from "
          << patch_merge_target_conditions_->first.ToString() << " with "
          << patch_merge_target_conditions_->second.ToString() << " up to "
          << patch_merge_glyphs_->size() << " glyphs." << std::endl
          << "  New patch size " << new_patch_size_ << " bytes. " << std::endl
          << "  Cost delta is " << cost_delta_ << ".";

  // CombinePatches() will do invalidation as needed, so nothing else needs to
  // be done to apply this merge.
  merger.InvalidateExclusivePatchSizes();
  return merger.Context().glyph_groupings.CombinePatches(*patch_merge_glyphs_,
                                                         {});
}

bool CandidateMerge::WouldMixFeaturesAndCodepoints(
    const RequestedSegmentationInformation& segment_info,
    segment_index_t base_segment_index, const SegmentSet& segments) {
  const auto& base =
      segment_info.Segments().at(base_segment_index).Definition();
  bool base_codepoints_only =
      !base.codepoints.empty() && base.feature_tags.empty();
  bool base_features_only =
      base.codepoints.empty() && !base.feature_tags.empty();

  if (!base_codepoints_only && !base_features_only) {
    return false;
  }

  for (segment_index_t id : segments) {
    const auto& s = segment_info.Segments().at(id).Definition();

    if (base_codepoints_only && !s.feature_tags.empty()) {
      return true;
    } else if (base_features_only && !s.codepoints.empty()) {
      return true;
    }
  }

  return false;
}

static void MergeSegments(const Merger& merger, const SegmentSet& segments,
                          Segment& base) {
  std::vector<const Segment*> merged_segments{&base};
  const auto& segmentation_info = merger.Context().SegmentationInfo();

  SubsetDefinition union_def = base.Definition();
  for (segment_index_t next : segments) {
    const auto& s = segmentation_info.Segments()[next];
    union_def.Union(s.Definition());
    merged_segments.push_back(&s);
  }

  // Compute probability before modifying base since it's in the merged segments
  // array.
  base.Definition() = std::move(union_def);
}

static Status FindModifiedConditions(
    const Merger& merger, const SegmentSet& merged_segments,
    flat_hash_map<const ActivationCondition*, const GlyphSet*>&
        modified_conditions) {
  const auto& groupings = merger.Context().glyph_groupings;
  const auto& conditions_and_glyphs = groupings.ConditionsAndGlyphs();

  for (auto s : merged_segments) {
    for (const auto& c : groupings.TriggeringSegmentToConditions(s)) {
      auto glyph_it = conditions_and_glyphs.find(c);
      if (glyph_it == conditions_and_glyphs.end()) {
        return absl::InternalError(
            "Condition which should be present wasn't found.");
      }
      modified_conditions.try_emplace(&glyph_it->first, &glyph_it->second);
    }
  }

  return absl::OkStatus();
}

StatusOr<uint32_t> CandidateMerge::Woff2SizeOf(hb_face_t* original_face,
                                               const SubsetDefinition& def,
                                               int quality) {
  hb_subset_input_t* input = hb_subset_input_create_or_fail();
  if (!input) {
    return absl::InternalError("Failed to create subset input.");
  }
  def.ConfigureInput(input, original_face);

  hb_face_t* init_face = hb_subset_or_fail(original_face, input);
  hb_subset_input_destroy(input);
  if (!init_face) {
    return absl::InternalError("Failed to create initial face subset.");
  }

  FontData init_data(init_face);
  hb_face_destroy(init_face);

  FontData woff2 = TRY(Woff2::EncodeWoff2(init_data.str(), false, quality));
  return (double)woff2.size();
}

StatusOr<double> CandidateMerge::ComputeInitFontCostDelta(
    Merger& merger, uint32_t existing_init_font_size,
    const ActivationCondition& condition, const GlyphSet& glyphs,
    flat_hash_map<ift::common::GlyphSet, uint32_t>& smallest_size_increases,
    size_t profile_index) {
  const auto* calculator =
      TRY(merger.Strategy().ProbabilityCalculator(profile_index));

  // Brotli compression results can be a bit noisy and it's common to see very
  // different size increase (in both directions) for adding the same glyphs as
  // the base changes. So to help control some of this noise we use
  // smallest_size_increases to track the smallest size increase we've seen for
  // moving a specific set of glyphs.
  VLOG(1) << "cost_delta for move of glyphs " << glyphs.ToString()
          << " (condition " << condition.ToString()
          << ") to the initial font =";

  GlyphSet new_init_glyphs =
      merger.Context().SegmentationInfo().InitFontGlyphs();
  new_init_glyphs.union_set(glyphs);

  uint32_t after =
      TRY(merger.Context().patch_size_cache_for_init_font->GetPatchSize(
          new_init_glyphs));
  uint32_t init_increase = 0;

  if (after >= existing_init_font_size) {
    init_increase = after - existing_init_font_size;
    auto [it, _] = smallest_size_increases.emplace(glyphs, UINT32_MAX);
    if (init_increase < it->second) {
      it->second = init_increase;
    } else {
      init_increase = it->second;
    }
  }

  double patch_probability = TRY(condition.Probability(
      merger.Context().SegmentationInfo().Segments(), *calculator));
  uint32_t patch_size = TRY(
      merger.Context().patch_size_cache_for_init_font->GetPatchSize(glyphs));
  double per_request_overhead = merger.Strategy().NetworkOverheadCost();
  double total_delta = static_cast<double>(init_increase) -
                       patch_probability * (static_cast<double>(patch_size) +
                                            per_request_overhead);

  VLOG(1) << "    + " << init_increase << " [init font increase]";
  VLOG(1) << "    - (" << patch_probability << " * "
          << (static_cast<double>(patch_size) + per_request_overhead) << ") -> "
          << total_delta;
  return total_delta;
}

StatusOr<double> CandidateMerge::ComputeBestCaseInitFontCostDelta(
    Merger& merger, const ActivationCondition& condition,
    const GlyphSet& glyphs, size_t profile_index) {
  const auto* calculator =
      TRY(merger.Strategy().ProbabilityCalculator(profile_index));

  double best_case_reduction =
      merger.Strategy().BestCaseSizeReductionFraction();
  double per_request_overhead = merger.Strategy().NetworkOverheadCost();
  double patch_probability = TRY(condition.Probability(
      merger.Context().SegmentationInfo().Segments(), *calculator));
  double patch_size = TRY(
      merger.Context().patch_size_cache_for_init_font->GetPatchSize(glyphs));

  double cost_delta = best_case_reduction * patch_size -
                      patch_probability * (patch_size + per_request_overhead);

  VLOG(1) << "best case cost_delta for move of glyphs " << glyphs.ToString()
          << " to the initial font = " << cost_delta;

  return cost_delta;
}

// Estimates a merged patch size from the largest input patch and the remaining
// incremental bytes scaled by the best-case reduction fraction.
static uint32_t EstimateBestCaseMergedSize(const Merger& merger,
                                           uint32_t largest_patch_size,
                                           uint32_t extra_incremental_size) {
  double reduction_fraction = merger.Strategy().BestCaseSizeReductionFraction();
  uint32_t extra = std::max(
      static_cast<uint32_t>(extra_incremental_size * reduction_fraction),
      Merger::BEST_CASE_MERGE_SIZE_DELTA);
  return largest_patch_size + extra;
}

static void MaybeRecordMergedSizeReduction(Merger& merger,
                                           uint32_t largest_patch_size,
                                           uint32_t extra_incremental_size,
                                           uint32_t actual_size) {
  if (!merger.ShouldRecordMergedSizeReductions() ||
      extra_incremental_size == 0) {
    return;
  }
  int32_t extra_actual = static_cast<int32_t>(actual_size) -
                         static_cast<int32_t>(largest_patch_size);
  merger.RecordMergedSizeReduction(static_cast<double>(extra_actual) /
                                   static_cast<double>(extra_incremental_size));
}

template <bool best_case>
static StatusOr<double> ComputeInertPairCostDelta(
    Merger& merger, const SegmentSet& merged_segments, segment_index_t u,
    segment_index_t v, uint32_t size_u, uint32_t size_v,
    std::optional<double> delta_to_beat) {
  const auto& context = merger.Context();
  const auto& patch_size_cache = context.patch_size_cache;
  const uint32_t per_request_overhead = merger.Strategy().NetworkOverheadCost();
  const uint32_t base_overhead = TRY(merger.BasePatchOverhead());

  double prob_u = TRY(merger.AggregateProbability(u));
  double prob_v = TRY(merger.AggregateProbability(v));
  double d_u = prob_u * (size_u + per_request_overhead);
  double d_v = prob_v * (size_v + per_request_overhead);

  uint32_t inc_u = (size_u > base_overhead) ? (size_u - base_overhead) : 0;
  uint32_t inc_v = (size_v > base_overhead) ? (size_v - base_overhead) : 0;
  uint32_t largest_patch_size = std::max(size_u, size_v);
  uint32_t extra_raw = std::min(inc_u, inc_v);

  uint32_t size = 0;
  if (best_case) {
    size = EstimateBestCaseMergedSize(merger, largest_patch_size, extra_raw);
  } else {
    GlyphSet combined_glyphs = context.glyph_groupings.ExclusiveGlyphs(u);
    combined_glyphs.union_set(context.glyph_groupings.ExclusiveGlyphs(v));
    if (combined_glyphs.empty()) {
      std::array<double, 2> deltas = {-d_u, -d_v};
      return SumDeltas(absl::MakeSpan(deltas));
    }
    size = TRY(patch_size_cache->GetPatchSize(combined_glyphs));
    MaybeRecordMergedSizeReduction(merger, largest_patch_size, extra_raw, size);
  }

  double s = (size + per_request_overhead);
  double initial_delta = -d_u - d_v;
  double initial_magnitude_sum = -initial_delta;

  if (best_case && delta_to_beat.has_value()) {
    // We can cheaply compute a lower bound to the post merge probability.
    // Since delta = added - removed, delta will be most negative
    // with a lower probability for the merged patch. Thus if
    // if the lower bound cannot beat the threshold then this case
    // can be rejected.
    double min_prob_u = TRY(merger.MinAggregateProbability(u));
    double min_prob_v = TRY(merger.MinAggregateProbability(v));
    double min_merged_prob = std::max(min_prob_u, min_prob_v);
    double min_d_merged = min_merged_prob * s;
    double min_current_delta = initial_delta + min_d_merged;
    if (min_current_delta > *delta_to_beat) {
      double max_error = MaxSumError(3, initial_magnitude_sum, initial_delta,
                                     min_current_delta);
      if (min_current_delta - max_error > *delta_to_beat) {
        return absl::CancelledError();
      }
    }
  }

  double p = TRY(merger.AggregateMergedProbability(merged_segments));
  double d_merged = p * s;
  double current_delta = initial_delta + d_merged;
  if (delta_to_beat.has_value() && current_delta > *delta_to_beat) {
    double max_error =
        MaxSumError(3, initial_magnitude_sum, initial_delta, current_delta);
    if (current_delta - max_error > *delta_to_beat) {
      return absl::CancelledError();
    }
  }

  std::array<double, 3> deltas = {-d_u, -d_v, d_merged};
  double cost_delta = SumDeltas(absl::MakeSpan(deltas));
  VLOG(1) << "    = " << cost_delta;
  return cost_delta;
}

namespace {

struct MergedConditionInfo {
  const ActivationCondition* first_condition = nullptr;
  const GlyphSet* first_glyphs = nullptr;
  uint32_t source_count = 0;
  bool skip_new_condition = false;
  GlyphSet glyphs;
  uint32_t largest_patch_size = 0;
  uint32_t largest_incremental_size = 0;
  uint32_t combined_incremental_size = 0;
};

// Picks the representative segment `rep` that merged segments map onto.
// In best-case mode, conditions touching only `rep` map to themselves and
// cannot collide with one another, so choosing the segment with the most
// conditions minimizes the number of remaining conditions to transform.
template <bool best_case>
std::optional<segment_index_t> SelectRepresentativeSegment(
    const GlyphGroupings& groupings, const SegmentSet& merged_segments) {
  if constexpr (!best_case) {
    return *merged_segments.min();
  }

  std::optional<segment_index_t> rep;
  size_t max_cond_count = 0;
  for (segment_index_t s : merged_segments) {
    size_t count = groupings.TriggeringSegmentToConditions(s).size();
    if (count > max_cond_count) {
      max_cond_count = count;
      rep = s;
    }
  }
  return rep;
}

// Evaluates the exact or best-case cost delta of merging `merged_segments`,
// mapping affected activation conditions onto representative segment `rep`.
template <bool best_case>
class SegmentMergeCostEvaluator {
 public:
  SegmentMergeCostEvaluator(Merger& merger, const SegmentSet& merged_segments,
                            segment_index_t rep, uint32_t base_overhead)
      : merger_(merger),
        groupings_(merger.Context().glyph_groupings),
        merged_segments_(merged_segments),
        rep_(rep),
        unicode_range_mode_(merger.Context().GetTargetMode() ==
                            config::UNICODE_RANGE),
        base_overhead_(base_overhead),
        per_request_overhead_(merger.Strategy().NetworkOverheadCost()) {}

  StatusOr<double> Evaluate(std::optional<double> delta_to_beat) {
    TRYV(CollectModifiedConditions());
    if (new_conditions_.empty()) {
      return 0.0;
    }

    size_t active_new_count = TRY(FinalizeNewConditions());
    if (active_new_count == 0) {
      return 0.0;
    }

    return ComputeAddedCostsAndSum(active_new_count, delta_to_beat);
  }

 private:
  Status CollectModifiedConditions() {
    if constexpr (best_case) {
      if (merged_segments_.size() == 2) {
        segment_index_t u = *merged_segments_.min();
        segment_index_t v = *merged_segments_.max();
        segment_index_t other = (rep_ == u) ? v : u;
        const auto& other_conditions =
            groupings_.TriggeringSegmentToConditions(other);
        if (other_conditions.empty()) {
          return absl::OkStatus();
        }
        new_conditions_.reserve(other_conditions.size());
        for (const ActivationCondition& condition : other_conditions) {
          TRYV(AddModifiedCondition(&condition, nullptr));
        }
        return absl::OkStatus();
      }
    }

    std::optional<SegmentSet> other_segments;
    const SegmentSet* segments_to_scan = &merged_segments_;
    if constexpr (best_case) {
      // Conditions touching only `rep_` cannot collide with one another and
      // are probed directly in FinalizeNewConditions().
      other_segments = merged_segments_;
      other_segments->erase(rep_);
      segments_to_scan = &*other_segments;
    }

    flat_hash_map<const ActivationCondition*, const GlyphSet*>
        modified_conditions;
    TRYV(FindModifiedConditions(merger_, *segments_to_scan,
                                modified_conditions));
    if (modified_conditions.empty()) {
      return absl::OkStatus();
    }

    new_conditions_.reserve(modified_conditions.size());
    if constexpr (!best_case) {
      deltas_.reserve(2 * modified_conditions.size() + 1);
    }
    for (const auto& [condition, glyphs] : modified_conditions) {
      TRYV(AddModifiedCondition(condition, glyphs));
    }
    return absl::OkStatus();
  }

  StatusOr<size_t> FinalizeNewConditions() {
    size_t active_new_count = 0;
    for (auto& [c, info] : new_conditions_) {
      if constexpr (best_case) {
        // Check whether a condition touching only `rep_` (or exclusive(rep_))
        // collides with this post-merge condition.
        TRYV(AddExistingConditionIfPresent(c, info));
        if (c.IsUnitary()) {
          TRYV(AddExistingConditionIfPresent(
              ActivationCondition::exclusive_segment(rep_, 0), info));
        }
        if (info.source_count == 1) {
          // Non-colliding conditions keep their glyph set and cannot shrink, so
          // their best-case delta contribution is 0.
          info.skip_new_condition = true;
          continue;
        }
      } else if (info.source_count == 1) {
        bool was_active_before_merge =
            (!unicode_range_mode_ || info.first_condition->IsExclusive());
        if (was_active_before_merge &&
            info.first_condition->IsUnchangedByMerge(merged_segments_)) {
          info.skip_new_condition = true;
          continue;
        }
        TRYV(
            AddRemovedCondition(info.first_condition, info.first_glyphs, info));
      }
      active_new_count++;
    }
    return active_new_count;
  }

  StatusOr<double> ComputeAddedCostsAndSum(
      size_t active_new_count, std::optional<double> delta_to_beat) {
    size_t max_terms = deltas_.size() + active_new_count;
    deltas_.reserve(max_terms);

    double initial_magnitude_sum = -current_delta_;
    double initial_delta = current_delta_;

    for (auto& [c, info] : new_conditions_) {
      if (info.skip_new_condition) {
        continue;
      }

      uint32_t size = TRY(ComputeMergedPatchSize(info));
      // Because we haven't actuated the merge yet (segments does not reflect
      // it), AggregateMergedProbability() is needed to correctly compute the
      // new probability.
      double p =
          TRY(merger_.AggregateMergedProbability(c, rep_, merged_segments_));
      double s = size + per_request_overhead_;
      double d = p * s;
      VLOG(1) << "    + (" << p << " * " << s << ") -> " << d << " [new patch "
              << c.ToString() << "]";
      deltas_.push_back(d);
      current_delta_ += d;

      if (delta_to_beat.has_value() && current_delta_ > *delta_to_beat) {
        double max_error = MaxSumError(max_terms, initial_magnitude_sum,
                                       initial_delta, current_delta_);
        if (current_delta_ - max_error > *delta_to_beat) {
          return absl::CancelledError();
        }
      }
    }

    double cost_delta = SumDeltas(deltas_);
    VLOG(1) << "    = " << cost_delta;
    return cost_delta;
  }

  StatusOr<uint32_t> ComputeMergedPatchSize(const MergedConditionInfo& info) {
    uint32_t extra_raw =
        info.combined_incremental_size - info.largest_incremental_size;
    if constexpr (best_case) {
      return EstimateBestCaseMergedSize(merger_, info.largest_patch_size,
                                        extra_raw);
    }
    if (info.source_count == 1) {
      return info.largest_patch_size;
    }
    uint32_t size =
        TRY(merger_.Context().patch_size_cache->GetPatchSize(info.glyphs));
    MaybeRecordMergedSizeReduction(merger_, info.largest_patch_size, extra_raw,
                                   size);
    return size;
  }

  Status AddModifiedCondition(const ActivationCondition* condition,
                              const GlyphSet* glyphs) {
    ActivationCondition updated =
        condition->ReplaceSegments(rep_, merged_segments_);
    if (updated.IsExclusive()) {
      // Clear is exclusive flag so that de-dup happens properly.
      updated = ActivationCondition::clear_exclusive(std::move(updated));
    }
    if (unicode_range_mode_ && !updated.IsUnitary()) {
      // In unicode range mode non-exclusive patches are dropped from the final
      // segmentation and do not contribute to total cost. Any condition that
      // is not unitary after the merge was also non-exclusive before the merge.
      return absl::OkStatus();
    }

    auto [it, _] = new_conditions_.try_emplace(std::move(updated));
    return RecordSourceCondition(it->second, condition, glyphs);
  }

  Status AddExistingConditionIfPresent(const ActivationCondition& key,
                                       MergedConditionInfo& info) {
    const auto& conditions_and_glyphs = groupings_.ConditionsAndGlyphs();
    auto it = conditions_and_glyphs.find(key);
    if (it == conditions_and_glyphs.end()) {
      return absl::OkStatus();
    }
    return RecordSourceCondition(info, &it->first, &it->second);
  }

  Status RecordSourceCondition(MergedConditionInfo& info,
                               const ActivationCondition* condition,
                               const GlyphSet* glyphs) {
    info.source_count++;
    if (info.source_count == 1) {
      info.first_condition = condition;
      info.first_glyphs = glyphs;
      return absl::OkStatus();
    }
    if (info.source_count == 2) {
      // The previously collected first condition needs to be removed now that
      // it is merging with this condition.
      TRYV(AddRemovedCondition(info.first_condition, info.first_glyphs, info));
    }
    return AddRemovedCondition(condition, glyphs, info);
  }

  Status AddRemovedCondition(const ActivationCondition* condition,
                             const GlyphSet* glyphs,
                             MergedConditionInfo& info) {
    if (glyphs == nullptr) {
      const auto& conditions_and_glyphs = groupings_.ConditionsAndGlyphs();
      auto glyph_it = conditions_and_glyphs.find(*condition);
      if (glyph_it == conditions_and_glyphs.end()) {
        return absl::InternalError(
            "Condition which should be present wasn't found.");
      }
      glyphs = &glyph_it->second;
    }

    uint32_t patch_size =
        TRY(merger_.Context().patch_size_cache->GetPatchSize(*glyphs));
    if (!unicode_range_mode_ || condition->IsExclusive()) {
      double p = TRY(merger_.AggregateProbability(*condition));
      double d = p * (patch_size + per_request_overhead_);
      deltas_.push_back(-d);
      current_delta_ -= d;
      VLOG(1) << "    - (" << p << " * " << (patch_size + per_request_overhead_)
              << ") -> " << d << " [removed patch " << condition->ToString()
              << "]";
    }

    if constexpr (!best_case) {
      if (info.source_count > 1) {
        info.glyphs.union_set(*glyphs);
      }
    }

    uint32_t incremental_size =
        (patch_size > base_overhead_) ? (patch_size - base_overhead_) : 0;
    info.largest_patch_size = std::max(info.largest_patch_size, patch_size);
    info.largest_incremental_size =
        std::max(info.largest_incremental_size, incremental_size);
    info.combined_incremental_size += incremental_size;
    return absl::OkStatus();
  }

  Merger& merger_;
  const GlyphGroupings& groupings_;
  const SegmentSet& merged_segments_;
  segment_index_t rep_;
  bool unicode_range_mode_;
  uint32_t base_overhead_;
  uint32_t per_request_overhead_;

  // Maps post-merge conditions to the pre-merge conditions and patch metadata
  // that fold into them.
  flat_hash_map<ActivationCondition, MergedConditionInfo> new_conditions_;

  // Cost delta contributions are collected here and summed deterministically at
  // the end because `new_conditions_` is iterated in hash order. See
  // SumDeltas().
  std::vector<double> deltas_;
  double current_delta_ = 0.0;
};

}  // namespace

template <bool best_case>
StatusOr<double> CandidateMerge::ComputeCostDelta(
    Merger& merger, const SegmentSet& merged_segments,
    std::optional<double> delta_to_beat) {
  if (merged_segments.size() <= 1) {
    return 0;
  }

  VLOG(1) << "cost delta for merge of " << merged_segments.ToString() << " = ";

  const auto& context = merger.Context();
  if (merged_segments.size() == 2) {
    segment_index_t u = *merged_segments.min();
    segment_index_t v = *merged_segments.max();
    if (context.InertSegments().contains(u) &&
        context.InertSegments().contains(v)) {
      uint32_t size_u = TRY(merger.ExclusivePatchSize(u));
      uint32_t size_v = TRY(merger.ExclusivePatchSize(v));
      if (size_u > 0 && size_v > 0) {
        return ComputeInertPairCostDelta<best_case>(
            merger, merged_segments, u, v, size_u, size_v, delta_to_beat);
      }
    }
  }

  std::optional<segment_index_t> rep = SelectRepresentativeSegment<best_case>(
      context.glyph_groupings, merged_segments);
  if (!rep.has_value()) {
    return 0.0;
  }

  uint32_t base_overhead = TRY(merger.BasePatchOverhead());
  SegmentMergeCostEvaluator<best_case> evaluator(merger, merged_segments, *rep,
                                                 base_overhead);
  return evaluator.Evaluate(delta_to_beat);
}

template StatusOr<double> CandidateMerge::ComputeCostDelta<true>(
    Merger& merger, const SegmentSet& merged_segments,
    std::optional<double> delta_to_beat);
template StatusOr<double> CandidateMerge::ComputeCostDelta<false>(
    Merger& merger, const SegmentSet& merged_segments,
    std::optional<double> delta_to_beat);

StatusOr<CandidateMerge::PatchMergeDetails>
CandidateMerge::ComputePatchMergeDetails(
    const Merger& merger, const ActivationCondition& condition_a,
    const ActivationCondition& condition_b) {
  const auto& glyph_groupings = merger.Context().glyph_groupings;
  const auto& condition_and_glyphs = glyph_groupings.ConditionsAndGlyphs();

  auto glyphs_a_it = condition_and_glyphs.find(condition_a);
  auto glyphs_b_it = condition_and_glyphs.find(condition_b);

  if (glyphs_a_it == condition_and_glyphs.end() ||
      glyphs_b_it == condition_and_glyphs.end()) {
    return absl::InternalError(absl::StrCat(
        "Unable to find glyphs for conditions: ", condition_a.ToString(), ", ",
        condition_b.ToString()));
  }

  const GlyphSet& glyphs_a = glyphs_a_it->second;
  const GlyphSet& glyphs_b = glyphs_b_it->second;

  if (glyphs_a.empty() || glyphs_b.empty()) {
    return absl::InternalError(absl::StrCat(
        "Unexpected empty glyph sets for conditions: ", condition_a.ToString(),
        ", ", condition_b.ToString()));
  }

  GlyphSet merged_glyphs = glyphs_a;
  merged_glyphs.union_set(glyphs_b);

  // When computing the new merged condition utilize the original
  // pre-combination conditions. Doing it this way avoids applying
  // simplification multiple times which could result in a different merged
  // condition (since condition_a and condition_b may already be merged +
  // simplified).
  flat_hash_set<ActivationCondition> original_conditions;
  for (glyph_id_t g : merged_glyphs) {
    auto condition = glyph_groupings.GlyphToConditionPrecombination(g);
    if (!condition.has_value()) {
      return absl::InternalError(
          absl::StrCat("Cannot find precombination condition for g", g));
    }
    original_conditions.insert(*condition);
  }

  if (original_conditions.empty()) {
    return absl::InternalError(
        absl::StrCat("Unexpected empty original conditions."));
  }

  std::vector<const ActivationCondition*> cond_ptrs;
  cond_ptrs.reserve(original_conditions.size());
  for (const ActivationCondition& cond : original_conditions) {
    cond_ptrs.push_back(&cond);
  }
  ActivationCondition merged_condition =
      TRY(ActivationCondition::OrAll(cond_ptrs));

  if (merger.Context().GetConditionAnalysisMode() ==
      config::DEP_GRAPH_ONLY_WITH_SIMPLIFICATION) {
    // Merged condition will be simplified, so do the same here.
    merged_condition = merged_condition.NonCompositeSuperset();
  }

  GlyphSet existing_glyphs;
  auto existing_condition =
      merger.Context().glyph_groupings.ConditionsAndGlyphs().find(
          merged_condition);
  if (existing_condition !=
      merger.Context().glyph_groupings.ConditionsAndGlyphs().end()) {
    existing_glyphs = existing_condition->second;
  }

  // It's possible for the new condition to be equal to one of the input
  // conditions, in this case we don't want to record glyphs_existing since that
  // would cause double counting of the patches removal in cost assessment.
  if (condition_a == merged_condition || condition_b == merged_condition) {
    existing_glyphs.clear();
  }

  return PatchMergeDetails{.condition_a = condition_a,
                           .condition_b = condition_b,
                           .merged_condition = merged_condition,
                           .glyphs_a = glyphs_a,
                           .glyphs_b = glyphs_b,
                           .glyphs_existing = existing_glyphs,
                           .glyphs_merged = merged_glyphs};
}

template <bool best_case>
StatusOr<double> CandidateMerge::PatchMergeDetails::ComputePatchMergeCostDelta(
    const Merger& merger) const {
  // For a patch merge only three things are affected:
  // 1. Remove the patch with condition equal to condition_a.
  // 2. Remove the patch with condition equal to condition_b.
  // 3. Add a new combined patch that contains all of the glyphs of 1 + 2.
  //    New condition is (condition_a OR condition_b), with corresponding new
  //    probability.

  double network_overhead = merger.Strategy().NetworkOverheadCost();

  double size_a =
      TRY(merger.Context().patch_size_cache->GetPatchSize(glyphs_a));
  double probability_a = TRY(merger.AggregateProbability(condition_a));

  double size_b =
      TRY(merger.Context().patch_size_cache->GetPatchSize(glyphs_b));
  double probability_b = TRY(merger.AggregateProbability(condition_b));

  double size_existing = 0.0;
  if (!glyphs_existing.empty()) {
    size_existing =
        TRY(merger.Context().patch_size_cache->GetPatchSize(glyphs_existing));
  }

  double merged_patch_size = 0.0;
  if (best_case) {
    double best_case_reduction_fraction =
        merger.Strategy().BestCaseSizeReductionFraction();
    uint32_t total_size = size_a + size_b + size_existing;
    uint32_t max_size = std::max(std::max(size_a, size_b), size_existing);
    uint32_t extra = total_size - max_size;
    extra = std::max((uint32_t)(extra * best_case_reduction_fraction),
                     Merger::BEST_CASE_MERGE_SIZE_DELTA);
    merged_patch_size = max_size + extra;
  } else {
    GlyphSet merged_patch_glyphs = glyphs_merged;
    merged_patch_glyphs.union_set(glyphs_existing);
    merged_patch_size = TRY(
        merger.Context().patch_size_cache->GetPatchSize(merged_patch_glyphs));
  }

  size_a += network_overhead;
  size_b += network_overhead;
  size_existing += network_overhead;
  merged_patch_size += network_overhead;

  double merged_probability =
      TRY(merger.AggregateProbability(merged_condition));

  VLOG(1) << "cost_delta for patch merge of " << condition_a.ToString()
          << " with " << condition_b.ToString() << " =";
  double cost_delta = 0.0;

  double d = merged_probability * merged_patch_size;
  cost_delta += d;
  VLOG(1) << "    + (" << merged_probability << " * " << merged_patch_size
          << ") -> " << d << " [merged patch]";

  if (!glyphs_existing.empty()) {
    d = merged_probability * size_existing;
    cost_delta -= d;
    VLOG(1) << "    - (" << merged_probability << " * " << size_existing
            << ") -> " << d << " [removed existing patch]";
  }

  d = probability_a * size_a;
  cost_delta -= d;
  VLOG(1) << "    - (" << probability_a << " * " << size_a << ") -> " << d
          << " [removed patch A]";

  d = probability_b * size_b;
  cost_delta -= d;
  VLOG(1) << "    - (" << probability_b << " * " << size_b << ") -> " << d
          << " [removed patch B]";

  VLOG(1) << "    = " << cost_delta;
  return cost_delta;
}

static StatusOr<bool> ExceedsMaxPatchSize(const Merger& merger,
                                          segment_index_t base_segment_index,
                                          const SegmentSet& merged_segments,
                                          uint32_t& new_patch_size) {
  uint32_t max_patch_size = merger.Strategy().PatchSizeMaxBytes();

  flat_hash_map<const ActivationCondition*, const GlyphSet*>
      modified_conditions;
  TRYV(FindModifiedConditions(merger, merged_segments, modified_conditions));

  struct ConditionGlyphs {
    GlyphSet glyphs;
    uint32_t max_input_glyph_count = 0;
  };
  flat_hash_map<ActivationCondition, ConditionGlyphs> new_conditions;
  new_conditions.reserve(modified_conditions.size());

  for (const auto& [condition, glyphs] : modified_conditions) {
    ActivationCondition updated =
        condition->ReplaceSegments(base_segment_index, merged_segments);
    if (updated.IsExclusive()) {
      updated = ActivationCondition::clear_exclusive(std::move(updated));
    }
    auto& entry = new_conditions[std::move(updated)];
    entry.glyphs.union_set(*glyphs);
    entry.max_input_glyph_count =
        std::max<uint32_t>(entry.max_input_glyph_count, glyphs->size());
  }

  const auto& patch_size_cache = merger.Context().patch_size_cache;
  auto unitary_it = new_conditions.find(
      ActivationCondition::or_segments({base_segment_index}, 0));
  if (unitary_it != new_conditions.end()) {
    new_patch_size =
        TRY(patch_size_cache->GetPatchSize(unitary_it->second.glyphs));
    if (new_patch_size > max_patch_size) {
      VLOG(0) << "  Merged exclusive patch for " << base_segment_index
              << " is too big (" << new_patch_size << " > " << max_patch_size
              << "), skipping.";
      return true;
    }
  }

  for (auto& [condition, entry] : new_conditions) {
    if (condition.IsUnitary()) {
      continue;
    }

    if (entry.glyphs.size() <= entry.max_input_glyph_count) {
      // This non-exclusive condition did not grow in glyph count as a result
      // of the merge.
      continue;
    }

    uint32_t size = TRY(patch_size_cache->GetPatchSize(entry.glyphs));
    if (size > max_patch_size) {
      VLOG(0) << "  Merged non-exclusive patch " << condition.ToString()
              << " is too big (" << size << " > " << max_patch_size
              << "), skipping.";
      return true;
    }
  }

  return false;
}

StatusOr<std::optional<CandidateMerge>> CandidateMerge::AssessSegmentMerge(
    Merger& merger, segment_index_t base_segment_index,
    const SegmentSet& segments_to_merge_,
    const std::optional<CandidateMerge>& best_merge_candidate) {
  if (WouldMixFeaturesAndCodepoints(merger.Context().SegmentationInfo(),
                                    base_segment_index, segments_to_merge_)) {
    // With the heuristic merger if it doesn't find a previous merge candidate
    // will try to merge together segments that are composed of codepoints with
    // a segment that adds an optional feature. Since this feature segments are
    // likely rarely used this will inflate the size of the patches for those
    // codepoint segments unnecessarily.
    //
    // In the cost case we don't have proper frequency data for features and
    // likewise don't have co-occurence data with features + codepoints. This
    // makes it difficult to make proper merging decisions that involve
    // mixing codepoints and features.
    //
    // As a result, don't do merges that mix features and codepoints in any
    // cases.
    VLOG(0) << "  Merge would mix features into a codepoint only segment, "
               "skipping.";
    return std::nullopt;
  }

  // Create a merged segment, and remove all of the others
  SegmentSet segments_to_merge = segments_to_merge_;
  SegmentSet segments_to_merge_with_base = segments_to_merge_;
  segments_to_merge.erase(base_segment_index);
  segments_to_merge_with_base.insert(base_segment_index);

  bool segments_to_merge_are_inert =
      segments_to_merge.is_subset_of(merger.Context().InertSegments());
  bool segments_to_merge_and_base_are_inert =
      merger.Context().InertSegments().contains(base_segment_index) &&
      segments_to_merge_are_inert;

  std::optional<double> delta_to_beat = std::nullopt;
  if (best_merge_candidate.has_value()) {
    delta_to_beat = best_merge_candidate->CostDelta();
  }

  if (merger.Strategy().UseCosts() && delta_to_beat.has_value()) {
    // Before doing a full assessment check a "best case" cost delta.
    // If that doesn't beat the current smallest there's no need to
    // do more indepth analysis.
    auto best_case_delta_status = ComputeCostDelta<true>(
        merger, segments_to_merge_with_base, delta_to_beat);
    if (absl::IsCancelled(best_case_delta_status.status())) {
      // Delta computation was ended early since it couldn't beat
      // best_merge_candidate.
      return std::nullopt;
    }
    double best_case_delta = TRY(best_case_delta_status);
    if (best_case_delta >= *delta_to_beat) {
      // We can't possibly beat the current lowest delta.
      return std::nullopt;
    }
  }

  GlyphSet gid_conditions_to_update;
  for (segment_index_t segment_index : segments_to_merge) {
    // segments which are being removed/changed may appear in gid_conditions,
    // we need to update those (and the down stream and/or glyph groups) to
    // reflect the removal/change and allow recalculation during the
    // GroupGlyphs steps
    gid_conditions_to_update.union_set(
        merger.Context().glyph_condition_set.GlyphsWithSegment(segment_index));
  }

  uint32_t new_patch_size = 0;
  double cost_delta = 0.0;
  if (merger.Strategy().UseCosts()) {
    // Cost delta values are only needed when using cost based merge strategy.
    auto cost_delta_status = ComputeCostDelta<false>(
        merger, segments_to_merge_with_base, delta_to_beat);
    if (absl::IsCancelled(cost_delta_status.status())) {
      // Delta computation was ended early since it couldn't beat
      // best_merge_candidate.
      return std::nullopt;
    }
    cost_delta = TRY(cost_delta_status);
  } else if (TRY(ExceedsMaxPatchSize(merger, base_segment_index,
                                     segments_to_merge_with_base,
                                     new_patch_size))) {
    return std::nullopt;
  }

  if (delta_to_beat.has_value() && cost_delta >= *delta_to_beat) {
    // Our delta is not smaller, don't bother returning a candidate.
    return std::nullopt;
  }

  const auto& segments = merger.Context().SegmentationInfo().Segments();
  Segment merged_segment = segments[base_segment_index];
  MergeSegments(merger, segments_to_merge, merged_segment);

  CandidateMerge candidate(std::move(merged_segment));
  candidate.base_segment_index_ = base_segment_index;
  candidate.segments_to_merge_ = segments_to_merge;
  candidate.input_segments_are_inert_ = segments_to_merge_and_base_are_inert;
  candidate.new_patch_size_ = new_patch_size;
  candidate.cost_delta_ = cost_delta;
  candidate.invalidated_glyphs_ = gid_conditions_to_update;

  return candidate;
}

static bool HasInScopeClause(const ActivationCondition& condition,
                             const SegmentSet& inscope) {
  for (const SegmentSet& clause : condition.conditions()) {
    if (!clause.empty() && clause.is_subset_of(inscope)) {
      return true;
    }
  }
  return false;
}

static bool OutOfScopeClausesMatch(const ActivationCondition& a,
                                   const ActivationCondition& b,
                                   const SegmentSet& inscope) {
  const auto a_clauses = a.conditions();
  const auto b_clauses = b.conditions();
  auto a_it = a_clauses.begin();
  auto b_it = b_clauses.begin();

  auto advance_to_out_of_scope = [&](auto& it, auto end) {
    while (it != end && it->is_subset_of(inscope)) {
      ++it;
    }
  };

  while (true) {
    advance_to_out_of_scope(a_it, a_clauses.end());
    advance_to_out_of_scope(b_it, b_clauses.end());
    if (a_it == a_clauses.end() || b_it == b_clauses.end()) {
      return a_it == a_clauses.end() && b_it == b_clauses.end();
    }
    if (*a_it != *b_it) {
      return false;
    }
    ++a_it;
    ++b_it;
  }
}

StatusOr<std::optional<CandidateMerge>> CandidateMerge::AssessPatchMerge(
    Merger& merger, const ActivationCondition& condition_a,
    const ActivationCondition& condition_b,
    const std::optional<CandidateMerge>& best_merge_candidate) {
  if (condition_a == condition_b) {
    return absl::InternalError("Can't merge condition with itself");
  }

  // Only allow patch merging when both inputs have at least one purely
  // in-scope clause and identical out-of-scope clauses, so any out-of-group
  // clauses are factored out unchanged rather than merged across groups.
  const SegmentSet& inscope = merger.InscopeSegments();
  if (!HasInScopeClause(condition_a, inscope) ||
      !HasInScopeClause(condition_b, inscope) ||
      !OutOfScopeClausesMatch(condition_a, condition_b, inscope)) {
    return std::nullopt;
  }

  PatchMergeDetails details =
      TRY(ComputePatchMergeDetails(merger, condition_a, condition_b));

  // Simplification (e.g. NonCompositeSuperset) could drop the in-scope clause
  // or modify an out-of-scope clause; reject if that happens.
  if (!HasInScopeClause(details.merged_condition, inscope) ||
      !OutOfScopeClausesMatch(condition_a, details.merged_condition, inscope)) {
    return std::nullopt;
  }

  if (merger.Strategy().UseCosts() && best_merge_candidate.has_value()) {
    // Pre-filter, if possible, with a best case computation to avoid computing
    // the merged patch size.
    double cost_delta = TRY(details.ComputePatchMergeCostDelta<true>(merger));
    if (cost_delta >= best_merge_candidate->CostDelta()) {
      return std::nullopt;
    }
  }

  GlyphSet new_patch_glyphs = details.glyphs_merged;
  new_patch_glyphs.union_set(details.glyphs_existing);
  uint32_t new_patch_size =
      TRY(merger.Context().patch_size_cache->GetPatchSize(new_patch_glyphs));

  if (!merger.Strategy().UseCosts() &&
      new_patch_size > merger.Strategy().PatchSizeMaxBytes()) {
    return std::nullopt;
  }

  double cost_delta = 0.0;
  if (merger.Strategy().UseCosts()) {
    // Cost delta values are only needed when using cost based merge strategy.
    cost_delta = TRY(details.ComputePatchMergeCostDelta<false>(merger));
  }

  if (best_merge_candidate.has_value() &&
      cost_delta >= best_merge_candidate->CostDelta()) {
    // Our delta is not smaller, don't bother returning a candidate.
    return std::nullopt;
  }

  if (cost_delta >= 0) {
    // Only accept a non-negative delta if the merge would also increase group
    // size of at least one of the inputs
    uint32_t merged_group_size = details.merged_condition.EffectiveGroupSize();
    uint32_t a_group_size = details.condition_a.EffectiveGroupSize();
    uint32_t b_group_size = details.condition_b.EffectiveGroupSize();
    if (merged_group_size <= a_group_size &&
        merged_group_size <= b_group_size) {
      return std::nullopt;
    }
  }

  CandidateMerge candidate;
  candidate.base_segment_index_ = 0;  // Dummy value, not used for patch merges.
  candidate.segments_to_merge_ = condition_a.TriggeringSegments();
  candidate.segments_to_merge_.union_set(condition_b.TriggeringSegments());
  candidate.patch_merge_target_conditions_ = std::make_pair(
      std::move(details.condition_a), std::move(details.condition_b));
  candidate.patch_merge_glyphs_ = details.glyphs_merged;
  candidate.input_segments_are_inert_ = false;
  candidate.new_patch_size_ = new_patch_size;
  candidate.cost_delta_ = cost_delta;

  // Since patch merges trigger full recomputation (of the combined patches), no
  // glyphs need to be invalidated by this merge.
  candidate.invalidated_glyphs_ = {};

  return candidate;
}

}  // namespace ift::encoder
