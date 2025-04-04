#ifndef IFT_ENCODER_ENCODER_H_
#define IFT_ENCODER_ENCODER_H_

#include <cstdint>
#include <random>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "common/compat_id.h"
#include "common/font_data.h"
#include "hb-subset.h"
#include "ift/encoder/condition.h"
#include "ift/encoder/subset_definition.h"
#include "ift/proto/patch_map.h"
#include "ift/table_keyed_diff.h"

namespace ift::encoder {

/*
 * Implementation of an encoder which can convert non-IFT fonts to an IFT
 * font and a set of patches.
 */
class Encoder {
 public:
  // TODO(garretrieger): add api to configure brotli quality level (for glyph
  // and table keyed).
  //                     Default to 11 but in tests run lower quality.

  Encoder()
      : face_(common::make_hb_face(nullptr))

  {}

  Encoder(const Encoder&) = delete;
  Encoder(Encoder&& other) = delete;
  Encoder& operator=(const Encoder&) = delete;
  Encoder& operator=(Encoder&& other) = delete;

  /*
   * Configures how many graph levels can be reached from each node in the
   * encoded graph. Defaults to 1.
   */
  void SetJumpAhead(uint32_t count) { this->jump_ahead_ = count; }

  /*
   * Adds a segmentation of glyph data.
   *
   * In the generated encoding there will be one glyph keyed patch (containing
   * all data for all of the glyphs in the segment) per segment and unique
   * design space configuration.
   *
   * An id is provided which uniquely identifies this segment and can be used to
   * specify dependencies against this segment.
   */
  absl::Status AddGlyphDataPatch(uint32_t patch_id,
                                 const absl::btree_set<uint32_t>& gids);

  /*
   * Adds a condition which may trigger the inclusion of a glyph data patch.
   */
  absl::Status AddGlyphDataPatchCondition(Condition condition);

  void SetFace(hb_face_t* face) { face_.reset(hb_face_reference(face)); }

  /*
   * Configure the base subset to cover the provided codepoints, and the set of
   * layout features retained by default in the harfbuzz subsetter.
   */
  template <typename Set>
  absl::Status SetBaseSubset(const Set& base_codepoints) {
    if (!base_subset_.empty()) {
      return absl::FailedPreconditionError("Base subset has already been set.");
    }
    base_subset_.codepoints.insert(base_codepoints.begin(),
                                   base_codepoints.end());
    return absl::OkStatus();
  }

  /*
   * Adds a segment around which the non glyph data in the font will be split.
   */
  template <typename Set>
  void AddNonGlyphDataSegment(const Set& codepoints) {
    SubsetDefinition def;
    def.codepoints.insert(codepoints.begin(), codepoints.end());
    extension_subsets_.push_back(def);
  }

  /*
   * Marks the provided group offeature tags as optional. In the dependent
   * patch graph it will be possible to add support for the features at any
   * node via a patch. Once enabled data for all codepoints and those features
   * will always be available.
   */
  void AddFeatureGroupSegment(const absl::btree_set<hb_tag_t>& feature_tag);

  void AddDesignSpaceSegment(const design_space_t& space);

  struct Encoding {
    common::FontData init_font;
    absl::flat_hash_map<std::string, common::FontData> patches;
  };

  /*
   * Create an IFT encoded version of 'font' that initially supports
   * the configured base subset but can be extended via patches to support any
   * combination of of extension subsets.
   *
   * Returns: the IFT encoded initial font. Patches() will be populated with the
   * set of associated patch files.
   */
  absl::StatusOr<Encoding> Encode() const;

  // TODO(garretrieger): update handling of encoding for use in woff2,
  // see: https://w3c.github.io/IFT/Overview.html#ift-and-compression
  static absl::StatusOr<common::FontData> RoundTripWoff2(
      absl::string_view font, bool glyf_transform = true);

 public:
  absl::Status SetBaseSubsetFromDef(const SubsetDefinition& base_subset) {
    if (!base_subset_.empty()) {
      return absl::FailedPreconditionError("Base subset has already been set.");
    }
    base_subset_ = base_subset;
    return absl::OkStatus();
  }

  std::vector<SubsetDefinition> OutgoingEdges(const SubsetDefinition& base,
                                              uint32_t choose) const;

 private:
  struct ProcessingContext;

  // Returns the font subset which would be reach if all segments where added to
  // the font.
  absl::StatusOr<common::FontData> FullyExpandedSubset(
      const ProcessingContext& context) const;

  std::string UrlTemplate(uint32_t patch_set_id) const {
    if (patch_set_id == 0) {
      // patch_set_id 0 is always used for table keyed patches
      return "{id}.ift_tk";
    }

    // All other ids are for glyph keyed.
    return absl::StrCat(patch_set_id, "_{id}.ift_gk");
  }

  static void AddCombinations(const std::vector<const SubsetDefinition*>& in,
                              uint32_t number,
                              std::vector<SubsetDefinition>& out);

  SubsetDefinition Combine(const SubsetDefinition& s1,
                           const SubsetDefinition& s2) const;

  /*
   * Create an IFT encoded version of 'font' that initially supports
   * 'base_subset' but can be extended via patches to support any combination of
   * 'subsets'.
   *
   * Returns: the IFT encoded initial font. Patches() will be populated with the
   * set of associated patch files.
   */
  absl::StatusOr<common::FontData> Encode(ProcessingContext& context,
                                          const SubsetDefinition& base_subset,
                                          bool is_root = true) const;

  /*
   * Returns true if this encoding will contain both glyph keyed and table keyed
   * patches.
   */
  bool IsMixedMode() const { return !glyph_data_patches_.empty(); }

  absl::Status EnsureGlyphKeyedPatchesPopulated(
      ProcessingContext& context, const design_space_t& design_space,
      std::string& uri_template, common::CompatId& compat_id) const;

  absl::Status PopulateGlyphKeyedPatchMap(
      ift::proto::PatchMap& patch_map) const;

  absl::StatusOr<hb_subset_plan_t*> CreateSubsetPlan(
      const ProcessingContext& context, hb_face_t* font,
      const SubsetDefinition& def) const;

  absl::StatusOr<common::hb_face_unique_ptr> CutSubsetFaceBuilder(
      const ProcessingContext& context, hb_face_t* font,
      const SubsetDefinition& def) const;

  absl::StatusOr<common::FontData> GenerateBaseGvar(
      const ProcessingContext& context, hb_face_t* font,
      const design_space_t& design_space) const;

  absl::StatusOr<common::FontData> GenerateBaseCff2(
      const ProcessingContext& context, hb_face_t* font,
      const design_space_t& design_space) const;

  void SetMixedModeSubsettingFlagsIfNeeded(const ProcessingContext& context,
                                           hb_subset_input_t* input) const;

  absl::StatusOr<common::FontData> CutSubset(
      const ProcessingContext& context, hb_face_t* font,
      const SubsetDefinition& def, bool generate_glyph_keyed_bases) const;

  absl::StatusOr<common::FontData> Instance(
      const ProcessingContext& context, hb_face_t* font,
      const design_space_t& design_space) const;

  absl::StatusOr<std::unique_ptr<const common::BinaryDiff>> GetDifferFor(
      const common::FontData& font_data, common::CompatId compat_id,
      bool replace_url_template) const;

  static ift::TableKeyedDiff* FullFontTableKeyedDiff(
      common::CompatId base_compat_id) {
    return new TableKeyedDiff(base_compat_id);
  }

  static ift::TableKeyedDiff* MixedModeTableKeyedDiff(
      common::CompatId base_compat_id) {
    return new TableKeyedDiff(base_compat_id,
                              {"IFTX", "glyf", "loca", "gvar", "CFF ", "CFF2"});
  }

  static ift::TableKeyedDiff* ReplaceIftMapTableKeyedDiff(
      common::CompatId base_compat_id) {
    // the replacement differ is used during design space expansions, both
    // gvar and "IFT " are overwritten to be compatible with the new design
    // space. Glyph segment patches for all prev loaded glyphs will be
    // downloaded to repopulate variation data for any already loaded glyphs.
    return new TableKeyedDiff(base_compat_id, {"glyf", "loca", "CFF "},
                              {"IFTX", "gvar", "CFF2"});
  }

  bool AllocatePatchSet(ProcessingContext& context,
                        const design_space_t& design_space,
                        std::string& uri_template,
                        common::CompatId& compat_id) const;

  common::hb_face_unique_ptr face_;
  absl::btree_map<uint32_t, absl::btree_set<uint32_t>> glyph_data_patches_;
  std::vector<Condition> glyph_patch_conditions_;

  SubsetDefinition base_subset_;
  std::vector<SubsetDefinition> extension_subsets_;
  uint32_t jump_ahead_ = 1;
  uint32_t next_id_ = 0;

  struct ProcessingContext {
    ProcessingContext(uint32_t next_id)
        : gen_(),
          random_values_(0, std::numeric_limits<uint32_t>::max()),
          next_id_(next_id) {}

    std::mt19937 gen_;
    std::uniform_int_distribution<uint32_t> random_values_;

    common::FontData fully_expanded_subset_;
    bool force_long_loca_and_gvar_ = false;

    uint32_t next_id_ = 0;
    uint32_t next_patch_set_id_ =
        1;  // id 0 is reserved for table keyed patches.
    absl::flat_hash_map<design_space_t, std::string> patch_set_uri_templates_;
    absl::flat_hash_map<design_space_t, common::CompatId>
        glyph_keyed_compat_ids_;

    absl::flat_hash_map<SubsetDefinition, common::FontData> built_subsets_;
    absl::flat_hash_map<std::string, common::FontData> patches_;
    SubsetDefinition base_subset_;

    common::CompatId GenerateCompatId();
  };
};

}  // namespace ift::encoder

#endif  // IFT_ENCODER_ENCODER_H_
