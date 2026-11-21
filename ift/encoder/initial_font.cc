#include "ift/encoder/initial_font.h"

#include <utility>

#include "ift/common/try.h"
#include "ift/encoder/glyph_closure_cache.h"
#include "ift/encoder/subset_definition.h"

namespace ift::encoder {

InitialFont::InitialFont(SubsetDefinition closed_input)
    : closed_input_definition_(std::move(closed_input)),
      unsegmented_definition_(closed_input_definition_),
      table_keyed_codepoints_(closed_input_definition_.codepoints),
      table_keyed_features_(closed_input_definition_.feature_tags.begin(),
                            closed_input_definition_.feature_tags.end()),
      glyph_keyed_glyphs_(closed_input_definition_.gids) {}

absl::StatusOr<InitialFont> InitialFont::Create(
    GlyphClosureCache& closure_cache,
    const SubsetDefinition& input_definition) {
  SubsetDefinition expanded =
      TRY(closure_cache.ExpandClosure(input_definition));
  return InitialFont(std::move(expanded));
}

}  // namespace ift::encoder
