#ifndef IFT_ENCODER_REQUESTED_SEGMENTATION_INFORMATION_H_
#define IFT_ENCODER_REQUESTED_SEGMENTATION_INFORMATION_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ift/common/int_set.h"
#include "ift/encoder/glyph_closure_cache.h"
#include "ift/encoder/initial_font.h"
#include "ift/encoder/segment.h"
#include "ift/encoder/subset_definition.h"
#include "ift/encoder/types.h"

namespace ift::encoder {

/*
 * Stores basic information about the configuration of a requested segmentation
 */
class RequestedSegmentationInformation {
 public:
  static absl::StatusOr<std::unique_ptr<RequestedSegmentationInformation>>
  Create(std::vector<Segment> segments, SubsetDefinition init_font_segment,
         GlyphClosureCache& closure_cache);

 private:
  RequestedSegmentationInformation() = default;

 public:
  // Merge all of the segments in to_merge into base, assigned it
  // a new subset definition "merged_segment".
  uint32_t AssignMergedSegment(segment_index_t base,
                               const ift::common::SegmentSet& to_merge,
                               const Segment& merged_segment) {
    auto& base_segment = segments_[base];
    RemoveFromIndex(base, base_segment.Definition().codepoints);
    base_segment = merged_segment;
    AddToIndex(base, base_segment.Definition().codepoints);

    for (segment_index_t s : to_merge) {
      // To avoid changing the indices of other segments set the ones we're
      // removing to empty sets. That effectively disables them.
      RemoveFromIndex(s, segments_[s].Definition().codepoints);
      segments_[s].Clear();
    }
    return base_segment.Definition().codepoints.size();
  }

  void ResetSegments(std::vector<Segment> segments);

  // Moves `glyphs` into the glyph-keyed initial font (removing them from
  // `non_init_font_glyphs_` and `non_init_font_non_empty_glyphs_`) without
  // running closure expansion or modifying `init_font_.ClosedInputDefinition()`
  // or `segments_`.
  void MoveGlyphsToInitFont(const ift::common::GlyphSet& glyphs);

  // Unions `def.codepoints` and `def.feature_tags` into the table-keyed
  // initial font without running closure expansion or removing items from
  // `segments_`.
  void AddToInitFontSegment(const SubsetDefinition& def);

  // Clears segments in `unused_segments`, recording their definitions in
  // `init_font_.UnsegmentedDefinition()` and updating `codepoint_to_segments_`.
  void ClearUnusedInitSegments(const ift::common::SegmentSet& unused_segments);

  // Replays moved initial-font state on top of the frozen closed input
  // baseline.
  absl::Status ReplayInitFontMoves(GlyphClosureCache& closure_cache,
                                   const InitialFont& init_font);

  const InitialFont& InitFont() const { return init_font_; }

  // Glyphs in `full_closure_` excluding `init_font_.ClosedInputGlyphs()`. Used
  // by `DependencyClosure` and `DependencyGraph`.
  const ift::common::GlyphSet& NonClosedInputGlyphs() const {
    return non_closed_input_glyphs_;
  }

  // Output initial font segment and outline glyphs (including items moved
  // directly without closure expansion).
  SubsetDefinition InitFontSegment() const {
    return init_font_.TableKeyedDefinition();
  }

  // Returns the table-keyed init font segment with all default always included
  // items removed.
  SubsetDefinition InitFontSegmentWithoutDefaults() const {
    return init_font_.TableKeyedDefinitionWithoutDefaults();
  }

  const ift::common::GlyphSet& InitFontGlyphs() const {
    return init_font_.GlyphKeyedGlyphs();
  }
  const ift::common::GlyphSet& NonInitFontGlyphs() const {
    return non_init_font_glyphs_;
  }
  const ift::common::GlyphSet& NonInitFontNonEmptyGlyphs() const {
    return non_init_font_non_empty_glyphs_;
  }

  const ift::common::GlyphSet& FullClosure() const { return full_closure_; }
  const ift::common::CodepointSet& FullCodepointClosure() const {
    return full_codepoint_closure_;
  }

  const SubsetDefinition& FullDefinition() const { return full_definition_; }

  bool SegmentsAreDisjoint() const { return segments_disjoint_; }

  const std::vector<Segment>& Segments() const { return segments_; }

  const std::vector<SubsetDefinition> SegmentSubsetDefinitions() const {
    std::vector<SubsetDefinition> out;
    for (const auto& s : segments_) {
      out.push_back(s.Definition());
    }
    return out;
  }

  ift::common::SegmentSet NonEmptySegments() const {
    // TODO(garretrieger): consider caching this value.
    ift::common::SegmentSet segments;
    segment_index_t index = 0;
    for (const auto& s : Segments()) {
      if (!s.Definition().Empty()) {
        segments.insert(index);
      }
      index++;
    }
    return segments;
  }

  SubsetDefinition CombinedDefinition(
      const ift::common::SegmentSet& segments) const {
    SubsetDefinition def;
    for (segment_index_t s : segments) {
      def.Union(Segments().at(s).Definition());
    }

    // Init font subset definition must be part of the closure input
    // since it contributes to reachability of things.
    def.Union(init_font_.ClosedInputDefinition());

    return def;
  }

  ift::common::SegmentSet SegmentsForCodepoints(
      const ift::common::CodepointSet& codepoints) const;

 private:
  absl::Status InitializeInitSubset(GlyphClosureCache& closure_cache,
                                    std::vector<Segment> segments,
                                    const SubsetDefinition& init_font_segment);
  void PopulateSegments(std::vector<Segment> segments);
  absl::Status RefreshClosures(GlyphClosureCache& closure_cache);
  void AddToIndex(segment_index_t seg_idx,
                  const ift::common::CodepointSet& codepoints);
  void RemoveFromIndex(segment_index_t seg_idx,
                       const ift::common::CodepointSet& codepoints);

  std::vector<Segment> segments_;
  InitialFont init_font_;
  SubsetDefinition full_definition_;
  ift::common::GlyphSet full_closure_;
  ift::common::CodepointSet full_codepoint_closure_;
  ift::common::GlyphSet empty_glyphs_;
  // Glyphs in `full_closure_` excluding `init_font_.ClosedInputGlyphs()`.
  ift::common::GlyphSet non_closed_input_glyphs_;
  // Glyphs in `full_closure_` excluding `init_font_.GlyphKeyedGlyphs()`.
  ift::common::GlyphSet non_init_font_glyphs_;
  // `non_init_font_glyphs_` excluding empty outline glyphs.
  ift::common::GlyphSet non_init_font_non_empty_glyphs_;

  bool segments_disjoint_ = true;
  absl::flat_hash_map<hb_codepoint_t, ift::common::SegmentSet>
      codepoint_to_segments_;
};

}  // namespace ift::encoder

#endif  // IFT_ENCODER_REQUESTED_SEGMENTATION_INFORMATION_H_