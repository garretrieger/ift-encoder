#ifndef IFT_ENCODER_INITIAL_FONT_H_
#define IFT_ENCODER_INITIAL_FONT_H_

#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "hb.h"
#include "ift/common/int_set.h"
#include "ift/encoder/glyph_closure_cache.h"
#include "ift/encoder/init_subset_defaults.h"
#include "ift/encoder/subset_definition.h"

namespace ift::encoder {

// Tracks the subset definitions and glyph/codepoint/feature sets that make up
// the initial font across dependency-graph analysis, segmentation tracking, and
// final table-keyed / glyph-keyed output.
class InitialFont {
 public:
  InitialFont() = default;

  bool operator==(const InitialFont& other) const = default;

  // Expands closure of `input_definition`, verifies it is a superset of any
  // previously assigned `closed_input_definition_` (returning
  // InvalidArgumentError without modifying state otherwise), stores it as
  // `closed_input_definition_`, and includes its codepoints, features, and
  // glyphs in the unsegmented, table-keyed, and glyph-keyed initial font sets.
  absl::Status ReassignClosedInput(GlyphClosureCache& closure_cache,
                                   const SubsetDefinition& input_definition);

  // Unions `glyphs` into the glyph-keyed initial font glyph set without running
  // closure expansion.
  void AddGlyphKeyedGlyphs(const ift::common::GlyphSet& glyphs) {
    glyph_keyed_glyphs_.union_set(glyphs);
  }

  // Unions `def.codepoints` and `def.feature_tags` into the table-keyed initial
  // font sets without running closure expansion.
  void AddToTableKeyed(const SubsetDefinition& def) {
    table_keyed_codepoints_.union_set(def.codepoints);
    table_keyed_features_.insert(def.feature_tags.begin(),
                                 def.feature_tags.end());
  }

  // Records the definition of a cleared segment that is no longer represented
  // in `Segments()`.
  void AddClearedSegment(const SubsetDefinition& def) {
    unsegmented_definition_.Union(def);
  }

  // Resets cleared segment definitions back to `closed_input_definition_`.
  void ResetClearedSegments() {
    unsegmented_definition_ = closed_input_definition_;
  }

  const SubsetDefinition& ClosedInputDefinition() const {
    return closed_input_definition_;
  }
  const ift::common::GlyphSet& ClosedInputGlyphs() const {
    return closed_input_definition_.gids;
  }

  const SubsetDefinition& UnsegmentedDefinition() const {
    return unsegmented_definition_;
  }

  const ift::common::CodepointSet& TableKeyedCodepoints() const {
    return table_keyed_codepoints_;
  }
  const absl::btree_set<hb_tag_t>& TableKeyedFeatures() const {
    return table_keyed_features_;
  }

  SubsetDefinition TableKeyedDefinition() const {
    SubsetDefinition result = closed_input_definition_;
    result.codepoints = table_keyed_codepoints_;
    result.feature_tags = table_keyed_features_;
    return result;
  }

  SubsetDefinition TableKeyedDefinitionWithoutDefaults() const {
    SubsetDefinition result = TableKeyedDefinition();
    SubsetDefinition defaults;
    AddInitSubsetDefaults(defaults);
    result.Subtract(defaults);
    return result;
  }

  const ift::common::GlyphSet& GlyphKeyedGlyphs() const {
    return glyph_keyed_glyphs_;
  }

  // Returns true if all codepoints, glyphs, features, and design space ranges
  // in `def` are already present in the initial font's output sets.
  bool ContainsSegment(const SubsetDefinition& def) const {
    if (!def.codepoints.is_subset_of(table_keyed_codepoints_) ||
        !def.gids.is_subset_of(glyph_keyed_glyphs_)) {
      return false;
    }
    for (hb_tag_t tag : def.feature_tags) {
      if (!table_keyed_features_.contains(tag)) {
        return false;
      }
    }
    for (const auto& [tag, range] : def.design_space) {
      auto it = closed_input_definition_.design_space.find(tag);
      if (it == closed_input_definition_.design_space.end() ||
          it->second != range) {
        return false;
      }
    }
    return true;
  }

 private:
  bool has_closed_input_ = false;
  // Subset definition provided by the input configuration, includes glyph
  // closure. Used by dep graph.
  SubsetDefinition closed_input_definition_;

  // `closed_input_definition_` plus definitions of any segments moved to the
  // initial font and cleared from `Segments()`. Used by
  // `GlyphClosureCache::ComputeExceptSegment` to represent all initial-font
  // items that are not in any active segment.
  SubsetDefinition unsegmented_definition_;

  // Output code point and feature set to be used by the table keyed initial
  // font.
  ift::common::CodepointSet table_keyed_codepoints_;
  absl::btree_set<hb_tag_t> table_keyed_features_;

  // Output glyph set to be used by the glyph keyed initial font.
  ift::common::GlyphSet glyph_keyed_glyphs_;
};

}  // namespace ift::encoder

#endif  // IFT_ENCODER_INITIAL_FONT_H_
