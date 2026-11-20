#include "ift/encoder/requested_segmentation_information.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ift/common/font_helper.h"
#include "ift/common/int_set.h"
#include "ift/common/try.h"
#include "ift/encoder/glyph_closure_cache.h"
#include "ift/encoder/initial_font.h"
#include "ift/encoder/segment.h"
#include "ift/encoder/subset_definition.h"

using absl::Status;
using absl::StatusOr;
using ift::common::CodepointSet;
using ift::common::FontHelper;
using ift::common::GlyphSet;
using ift::common::SegmentSet;

namespace ift::encoder {

static bool CheckSegmentsAreDisjoint(const SubsetDefinition& init_segment,
                                     const std::vector<Segment>& segments) {
  bool segments_disjoint = true;
  SubsetDefinition full_definition = init_segment;
  for (const auto& s : segments) {
    const auto& def = s.Definition();
    if (segments_disjoint) {
      for (hb_tag_t tag : def.feature_tags) {
        if (full_definition.feature_tags.contains(tag)) {
          segments_disjoint = false;
        }
      }
      segments_disjoint =
          segments_disjoint &&
          !full_definition.codepoints.intersects(def.codepoints);
    }
    full_definition.Union(s.Definition());
  }
  return segments_disjoint;
}

StatusOr<std::unique_ptr<RequestedSegmentationInformation>>
RequestedSegmentationInformation::Create(std::vector<Segment> segments,
                                         SubsetDefinition init_font_segment,
                                         GlyphClosureCache& closure_cache) {
  std::unique_ptr<RequestedSegmentationInformation> info(
      new RequestedSegmentationInformation(segments, init_font_segment));
  TRYV(info->ReassignInitSubset(closure_cache, init_font_segment));
  info->segments_disjoint_ =
      CheckSegmentsAreDisjoint(init_font_segment, info->segments_);
  return info;
}

RequestedSegmentationInformation::RequestedSegmentationInformation(
    std::vector<Segment> segments, SubsetDefinition init_font_segment)
    : segments_(std::move(segments)) {
  // ReassignInitSubset expects full_definition_ is already populated.
  full_definition_ = init_font_segment;
  for (size_t i = 0; i < segments_.size(); ++i) {
    full_definition_.Union(segments_[i].Definition());
    AddToIndex(i, segments_[i].Definition().codepoints);
  }
}

void RequestedSegmentationInformation::ResetSegments(
    std::vector<Segment> segments) {
  segments_ = std::move(segments);
  init_font_.ResetClearedSegments();
  full_definition_ = init_font_.UnsegmentedDefinition();
  const CodepointSet& closed_init_codepoints =
      init_font_.ClosedInputDefinition().codepoints;
  codepoint_to_segments_.clear();
  for (size_t i = 0; i < segments_.size(); ++i) {
    auto& s = segments_[i];
    if (s.Definition().codepoints.intersects(closed_init_codepoints)) {
      s.Definition().codepoints.subtract(closed_init_codepoints);
    }
    full_definition_.Union(s.Definition());
    AddToIndex(i, s.Definition().codepoints);
  }
  segments_disjoint_ =
      CheckSegmentsAreDisjoint(init_font_.ClosedInputDefinition(), segments_);
}

Status RequestedSegmentationInformation::RefreshClosures(
    GlyphClosureCache& closure_cache) {
  full_closure_ = TRY(closure_cache.GlyphClosure(full_definition_));
  full_codepoint_closure_ = closure_cache.CodepointsForGlyphs(full_closure_);

  non_closed_input_glyphs_ = full_closure_;
  non_closed_input_glyphs_.subtract(init_font_.ClosedInputGlyphs());

  GlyphSet unchecked = non_closed_input_glyphs_;
  unchecked.subtract(checked_empty_glyphs_);
  if (!unchecked.empty()) {
    empty_glyphs_.union_set(
        TRY(FontHelper::EmptyGlyphs(closure_cache.OriginalFace(), unchecked)));
    checked_empty_glyphs_.union_set(unchecked);
  }

  non_init_font_glyphs_ = full_closure_;
  non_init_font_glyphs_.subtract(init_font_.GlyphKeyedGlyphs());
  non_init_font_non_empty_glyphs_ = non_init_font_glyphs_;
  non_init_font_non_empty_glyphs_.subtract(empty_glyphs_);
  return absl::OkStatus();
}

void RequestedSegmentationInformation::MoveGlyphsToInitFont(
    const GlyphSet& glyphs) {
  init_font_.AddGlyphKeyedGlyphs(glyphs);
  non_init_font_glyphs_.subtract(glyphs);
  non_init_font_non_empty_glyphs_.subtract(glyphs);
}

void RequestedSegmentationInformation::AddToInitFontSegment(
    const SubsetDefinition& def) {
  init_font_.AddToTableKeyed(def);
}

void RequestedSegmentationInformation::ClearUnusedInitSegments(
    const SegmentSet& unused_segments) {
  for (segment_index_t s : unused_segments) {
    auto& seg = segments_.at(s);
    init_font_.AddClearedSegment(seg.Definition());
    RemoveFromIndex(s, seg.Definition().codepoints);
    seg.Clear();
  }
}

Status RequestedSegmentationInformation::ReplayInitFontMoves(
    GlyphClosureCache& closure_cache, const InitialFont& init_font) {
  init_font_ = init_font;
  full_definition_.Union(init_font_.UnsegmentedDefinition());
  return RefreshClosures(closure_cache);
}

Status RequestedSegmentationInformation::ReassignInitSubset(
    GlyphClosureCache& closure_cache, const SubsetDefinition& new_def) {
  // TODO XXXXX Can we avoid aggressively expanding the init closure?
  // ie. run the init subsetting/closure operation purely in glyph form.
  TRYV(init_font_.ReassignClosedInput(closure_cache, new_def));
  const SubsetDefinition& closed_init = init_font_.ClosedInputDefinition();

  full_definition_.Union(closed_init);
  TRYV(RefreshClosures(closure_cache));

  // Changing the init font subset may have caused additional codepoints to be
  // moved to the init font. We need to update the segment definitions to
  // remove these.
  for (size_t i = 0; i < segments_.size(); ++i) {
    auto& s = segments_[i];
    if (s.Definition().codepoints.intersects(closed_init.codepoints)) {
      CodepointSet intersection = s.Definition().codepoints;
      intersection.intersect(closed_init.codepoints);
      RemoveFromIndex(i, intersection);
      s.Definition().codepoints.subtract(closed_init.codepoints);
    }
  }
  return absl::OkStatus();
}

SegmentSet RequestedSegmentationInformation::SegmentsForCodepoints(
    const CodepointSet& codepoints) const {
  SegmentSet result;
  for (hb_codepoint_t cp : codepoints) {
    auto it = codepoint_to_segments_.find(cp);
    if (it != codepoint_to_segments_.end()) {
      result.union_set(it->second);
    }
  }
  return result;
}

void RequestedSegmentationInformation::AddToIndex(
    segment_index_t s, const CodepointSet& codepoints) {
  for (hb_codepoint_t cp : codepoints) {
    codepoint_to_segments_[cp].insert(s);
  }
}

void RequestedSegmentationInformation::RemoveFromIndex(
    segment_index_t s, const CodepointSet& codepoints) {
  for (hb_codepoint_t cp : codepoints) {
    auto it = codepoint_to_segments_.find(cp);
    if (it == codepoint_to_segments_.end()) {
      continue;
    }
    it->second.erase(s);
    if (it->second.empty()) {
      codepoint_to_segments_.erase(it);
    }
  }
}

}  // namespace ift::encoder