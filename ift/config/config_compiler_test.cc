#include "ift/config/config_compiler.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "ift/common/font_data.h"
#include "ift/common/font_helper.h"
#include "ift/common/int_set.h"
#include "ift/common/test_font_loader.h"
#include "ift/config/segmentation_plan.pb.h"
#include "ift/encoder/compiler.h"

using ift::common::FontData;
using ift::common::FontHelper;
using ift::common::GlyphSet;
using ift::common::IntSet;
using ift::encoder::Compiler;

namespace ift::config {

TEST(ConfigCompilerTest, ConfigureOverrideUrlTemplatePrefix) {
  Compiler compiler;
  SegmentationPlan plan;
  std::string prefix_str = "https://example.com/";
  plan.mutable_advanced_settings()->set_override_url_template_prefix(
      prefix_str);

  absl::Status status = ConfigCompiler::Configure(plan, compiler);
  ASSERT_TRUE(status.ok()) << status;

  std::vector<uint8_t> expected(prefix_str.begin(), prefix_str.end());
  EXPECT_EQ(compiler.override_url_template_prefix(), expected);
}

TEST(ConfigCompilerTest, ConfigureMaxDepth) {
  Compiler compiler;
  SegmentationPlan plan;
  plan.set_max_depth(3);

  absl::Status status = ConfigCompiler::Configure(plan, compiler);
  ASSERT_TRUE(status.ok()) << status;

  EXPECT_EQ(compiler.max_depth(), 3);
}

TEST(ConfigCompilerTest, ConfigureIncludeAllSegmentPatches) {
  Compiler compiler;
  SegmentationPlan plan;
  plan.set_include_all_segment_patches(true);

  absl::Status status = ConfigCompiler::Configure(plan, compiler);
  ASSERT_TRUE(status.ok()) << status;

  EXPECT_TRUE(compiler.include_all_segment_patches());
}

TEST(ConfigCompilerTest, ConfigureMixedModeSeparatesInitialGlyphs) {
  auto loader = ift::common::TestFontLoader::Default().value();
  FontData roboto =
      loader->LoadFontData("ift/common/testdata/Roboto-Regular.ttf").value();
  auto orig_face = roboto.face();

  Compiler compiler;
  compiler.SetFace(orig_face.get());

  // In Roboto-Regular.ttf: 'f' = gid 74, 'i' = gid 77, 'fi' = gid 444, 'l' = 80
  SegmentationPlan plan;
  (*plan.mutable_segments())[0].mutable_codepoints()->add_values('f');
  (*plan.mutable_segments())[1].mutable_codepoints()->add_values('i');
  (*plan.mutable_glyph_patches())[1].add_values(444);
  auto* cond = plan.add_glyph_patch_conditions();
  cond->add_required_segments()->add_values(0);
  cond->add_required_segments()->add_values(1);
  cond->set_activated_patch(1);

  plan.mutable_initial_codepoints()->add_values('f');
  plan.mutable_initial_codepoints()->add_values('i');
  plan.mutable_initial_glyphs()->add_values(0);
  plan.mutable_initial_glyphs()->add_values(74);
  plan.mutable_initial_glyphs()->add_values(77);
  plan.add_non_glyph_codepoint_segmentation()->add_values('l');

  absl::Status status = ConfigCompiler::Configure(plan, compiler);
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_TRUE(compiler.init_glyphs().has_value());
  EXPECT_EQ(*compiler.init_glyphs(), (GlyphSet{0, 74, 77}));

  auto encoding = compiler.Compile();
  ASSERT_TRUE(encoding.ok()) << encoding.status();
  auto init_face = encoding->init_font.face();

  EXPECT_EQ(FontHelper::ToCodepointsSet(init_face.get()), (IntSet{'f', 'i'}));
  EXPECT_FALSE(FontHelper::GlyfData(init_face.get(), 74)->empty());
  EXPECT_FALSE(FontHelper::GlyfData(init_face.get(), 77)->empty());
  EXPECT_TRUE(FontHelper::GlyfData(init_face.get(), 444)->empty());
}

TEST(ConfigCompilerTest, ConfigureMixedModeWithoutInitialGlyphsUsesClosure) {
  auto loader = ift::common::TestFontLoader::Default().value();
  FontData roboto =
      loader->LoadFontData("ift/common/testdata/Roboto-Regular.ttf").value();
  auto orig_face = roboto.face();

  Compiler compiler;
  compiler.SetFace(orig_face.get());

  SegmentationPlan plan;
  (*plan.mutable_segments())[0].mutable_codepoints()->add_values('l');
  (*plan.mutable_glyph_patches())[1].add_values(80);
  auto* cond = plan.add_glyph_patch_conditions();
  cond->add_required_segments()->add_values(0);
  cond->set_activated_patch(1);

  plan.mutable_initial_codepoints()->add_values('f');
  plan.mutable_initial_codepoints()->add_values('i');
  plan.add_non_glyph_codepoint_segmentation()->add_values('l');

  absl::Status status = ConfigCompiler::Configure(plan, compiler);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_FALSE(compiler.init_glyphs().has_value());

  auto encoding = compiler.Compile();
  ASSERT_TRUE(encoding.ok()) << encoding.status();
  auto init_face = encoding->init_font.face();
  EXPECT_FALSE(FontHelper::GlyfData(init_face.get(), 444)->empty());
}

TEST(ConfigCompilerTest, ConfigureTableKeyedKeepsInitialGlyphsInInitSubset) {
  auto loader = ift::common::TestFontLoader::Default().value();
  // NotoSansJP-VF.empty-glyph.otf has 3 glyphs: 0 (.notdef), 1 (' '), 2 ('A').
  FontData font =
      loader->LoadFontData("ift/common/testdata/NotoSansJP-VF.empty-glyph.otf")
          .value();
  auto orig_face = font.face();

  Compiler compiler;
  compiler.SetFace(orig_face.get());

  SegmentationPlan plan;
  plan.mutable_initial_glyphs()->add_values(1);  // ' '
  plan.add_non_glyph_codepoint_segmentation()->add_values('A');

  absl::Status status = ConfigCompiler::Configure(plan, compiler);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_FALSE(compiler.init_glyphs().has_value());

  auto encoding = compiler.Compile();
  ASSERT_TRUE(encoding.ok()) << encoding.status();
  auto init_face = encoding->init_font.face();
  EXPECT_EQ(FontHelper::ToCodepointsSet(init_face.get()), (IntSet{' '}));
}

}  // namespace ift::config
