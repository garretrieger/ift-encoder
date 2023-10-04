#include "ift/iftb_binary_patch.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "ift/proto/ift_table.h"
#include "merger.h"
#include "patch_subset/font_data.h"

using absl::flat_hash_set;
using absl::Status;
using absl::StatusOr;
using ift::proto::IFTTable;
using iftb::merger;
using iftb::sfnt;
using patch_subset::FontData;

namespace ift {

Status IftbBinaryPatch::Patch(const FontData& font_base, const FontData& patch,
                              FontData* font_derived /* OUT */) const {
  std::vector<FontData> patches;
  patches.emplace_back();
  patches[0].shallow_copy(patch);
  return Patch(font_base, patches, font_derived);
}

StatusOr<uint32_t> get_chunk_index(const FontData& patch) {
  if (patch.size() < 28) {
    return absl::InvalidArgumentError(
        "Can't read chunk index in patch, too short.");
  }

  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(patch.data() + 24);
  return (((uint32_t)bytes[0]) << 24) + (((uint32_t)bytes[1]) << 16) +
         (((uint32_t)bytes[2]) << 8) + ((uint32_t)bytes[3]);
}

Status IftbBinaryPatch::Patch(const FontData& font_base,
                              const std::vector<FontData>& patches,
                              FontData* font_derived /* OUT */) const {
  // TODO(garretrieger): this makes many unnessecary copies of data.
  //   Optimize to avoid them.
  merger merger;
  auto ift_table = IFTTable::FromFont(font_base);

  if (!ift_table.ok()) {
    return ift_table.status();
  }

  uint32_t id[4];
  ift_table->GetId(id);
  merger.setID(id);

  flat_hash_set<uint32_t> patch_indices;
  for (const FontData& patch : patches) {
    auto idx = get_chunk_index(patch);
    if (!idx.ok()) {
      return idx.status();
    }
    patch_indices.insert(*idx);
    iftb::decodeBuffer(patch.data(), patch.size(), merger.stringForChunk(*idx));
  }

  if (!merger.unpackChunks()) {
    return absl::InvalidArgumentError("Failed to unpack the chunks.");
  }

  std::string font_data = font_base.string();
  sfnt sfnt;
  sfnt.setBuffer(font_data);
  if (!sfnt.read()) {
    return absl::InvalidArgumentError("Failed to read input font file.");
  }

  hb_face_t* face = font_base.reference_face();
  unsigned num_glyphs = hb_face_get_glyph_count(face);
  hb_face_destroy(face);

  uint32_t new_length = merger.calcLayout(
      sfnt, num_glyphs, 0 /* TODO: add CFF charstrings offset */);
  if (!new_length) {
    return absl::InvalidArgumentError(
        "Calculating layout before merge failed.");
  }
  std::string new_font_data;
  new_font_data.reserve(new_length);

  if (!merger.merge(sfnt, font_data.data(), new_font_data.data())) {
    return absl::InvalidArgumentError("IFTB Patch merging failed.");
  }

  auto s = ift_table->RemovePatches(patch_indices);
  if (!s.ok()) {
    return s;
  }

  hb_blob_t* blob = hb_blob_create(new_font_data.data(), new_font_data.size(),
                                   HB_MEMORY_MODE_READONLY, nullptr, nullptr);
  face = hb_face_create(blob, 0);
  hb_blob_destroy(blob);

  auto result = ift_table->AddToFont(face);
  hb_face_destroy(face);
  if (!result.ok()) {
    return result.status();
  }

  font_derived->shallow_copy(*result);
  return absl::OkStatus();
}

}  // namespace ift