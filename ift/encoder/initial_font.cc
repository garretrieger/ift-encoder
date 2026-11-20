#include "ift/encoder/initial_font.h"

#include <algorithm>
#include <utility>

#include "absl/status/status.h"
#include "ift/common/try.h"
#include "ift/encoder/glyph_closure_cache.h"
#include "ift/encoder/subset_definition.h"

namespace ift::encoder {
namespace {

bool IsSupersetOf(const SubsetDefinition& superset,
                  const SubsetDefinition& subset) {
  if (!subset.codepoints.is_subset_of(superset.codepoints)) {
    return false;
  }
  if (!subset.gids.is_subset_of(superset.gids)) {
    return false;
  }
  if (!std::includes(superset.feature_tags.begin(), superset.feature_tags.end(),
                     subset.feature_tags.begin(), subset.feature_tags.end())) {
    return false;
  }
  // Axes omitted from superset.design_space are unrestricted and therefore
  // cover any axis range present in subset.design_space.
  for (const auto& [tag, super_range] : superset.design_space) {
    auto it = subset.design_space.find(tag);
    if (it == subset.design_space.end()) {
      return false;
    }
    if (super_range.start() > it->second.start() ||
        super_range.end() < it->second.end()) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::Status InitialFont::ReassignClosedInput(
    GlyphClosureCache& closure_cache,
    const SubsetDefinition& input_definition) {
  SubsetDefinition expanded =
      TRY(closure_cache.ExpandClosure(input_definition));
  if (has_closed_input_ && !IsSupersetOf(expanded, closed_input_definition_)) {
    return absl::InvalidArgumentError(
        "ReassignClosedInput requires the closed input definition to be a "
        "superset of the previous closed input definition.");
  }

  has_closed_input_ = true;
  closed_input_definition_ = std::move(expanded);
  unsegmented_definition_.Union(closed_input_definition_);
  table_keyed_codepoints_.union_set(closed_input_definition_.codepoints);
  table_keyed_features_.insert(closed_input_definition_.feature_tags.begin(),
                               closed_input_definition_.feature_tags.end());
  glyph_keyed_glyphs_.union_set(closed_input_definition_.gids);
  return absl::OkStatus();
}

}  // namespace ift::encoder
