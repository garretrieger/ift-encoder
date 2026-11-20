#include "ift/encoder/requested_segmentation_information.h"

#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "ift/common/axis_range.h"
#include "ift/common/bazel_data_file_resolver.h"
#include "ift/common/font_data.h"
#include "ift/common/font_helper.h"
#include "ift/common/test_font_loader.h"
#include "ift/encoder/glyph_closure_cache.h"
#include "ift/encoder/init_subset_defaults.h"
#include "ift/encoder/subset_definition.h"
#include "ift/freq/probability_bound.h"

using ift::common::BazelDataFileResolver;
using ift::common::CodepointSet;
using ift::common::DataFileResolver;
using ift::common::FontData;
using ift::common::hb_face_unique_ptr;
using ift::common::make_hb_face;
using ift::common::SegmentSet;
using ift::freq::ProbabilityBound;

namespace ift::encoder {

class RequestedSegmentationInformationTest : public ::testing::Test {
 protected:
  RequestedSegmentationInformationTest()
      : roboto(make_hb_face(nullptr)),
        resolver(*BazelDataFileResolver::CreateForTest()) {
    roboto = from_file("ift/common/testdata/Roboto-Regular.ttf");
  }

  hb_face_unique_ptr from_file(const char* filename) {
    auto loader = ift::common::TestFontLoader::Default().value();
    return loader->LoadFace(filename).value();
  }

  hb_face_unique_ptr roboto;
  std::shared_ptr<DataFileResolver> resolver;
};

TEST_F(RequestedSegmentationInformationTest, SegmentsForCodepoints) {
  std::unique_ptr<GlyphClosureCache> cache =
      *GlyphClosureCache::Create(roboto.get(), *resolver);
  std::vector<Segment> segments{
      {{'a', 'b'}},
      {{'b', 'c'}},
      {{'d'}},
  };

  auto info_or = RequestedSegmentationInformation::Create(segments, {}, *cache);
  ASSERT_TRUE(info_or.ok());
  auto& info = *info_or;

  EXPECT_EQ(info->SegmentsForCodepoints({'a'}), (SegmentSet{0}));
  EXPECT_EQ(info->SegmentsForCodepoints({'b'}), (SegmentSet{0, 1}));
  EXPECT_EQ(info->SegmentsForCodepoints({'c'}), (SegmentSet{1}));
  EXPECT_EQ(info->SegmentsForCodepoints({'d'}), (SegmentSet{2}));
  EXPECT_EQ(info->SegmentsForCodepoints({'e'}), (SegmentSet{}));
  EXPECT_EQ(info->SegmentsForCodepoints({'a', 'c'}), (SegmentSet{0, 1}));
  EXPECT_EQ(info->SegmentsForCodepoints({'a', 'd'}), (SegmentSet{0, 2}));
}

TEST_F(RequestedSegmentationInformationTest, IndexUpdatesOnMerge) {
  std::unique_ptr<GlyphClosureCache> cache =
      *GlyphClosureCache::Create(roboto.get(), *resolver);
  std::vector<Segment> segments{
      {{'a'}},
      {{'b'}},
      {{'c'}},
  };

  auto info_or = RequestedSegmentationInformation::Create(segments, {}, *cache);
  ASSERT_TRUE(info_or.ok());
  auto& info = *info_or;

  EXPECT_EQ(info->SegmentsForCodepoints({'a'}), (SegmentSet{0}));
  EXPECT_EQ(info->SegmentsForCodepoints({'b'}), (SegmentSet{1}));
  EXPECT_EQ(info->SegmentsForCodepoints({'c'}), (SegmentSet{2}));

  // Merge segment 1 into segment 0, new definition is {'a', 'b'}
  Segment merged{{'a', 'b'}};
  info->AssignMergedSegment(0, {1}, merged);

  EXPECT_EQ(info->SegmentsForCodepoints({'a'}), (SegmentSet{0}));
  EXPECT_EQ(info->SegmentsForCodepoints({'b'}), (SegmentSet{0}));
  EXPECT_EQ(info->SegmentsForCodepoints({'c'}), (SegmentSet{2}));
}

TEST_F(RequestedSegmentationInformationTest, IndexUpdatesOnReassignInit) {
  std::unique_ptr<GlyphClosureCache> cache =
      *GlyphClosureCache::Create(roboto.get(), *resolver);
  std::vector<Segment> segments{
      {{'a', 'b'}},
      {{'b', 'c'}},
  };

  auto info_or = RequestedSegmentationInformation::Create(segments, {}, *cache);
  ASSERT_TRUE(info_or.ok());
  auto& info = *info_or;

  EXPECT_EQ(info->SegmentsForCodepoints({'a'}), (SegmentSet{0}));
  EXPECT_EQ(info->SegmentsForCodepoints({'b'}), (SegmentSet{0, 1}));
  EXPECT_EQ(info->SegmentsForCodepoints({'c'}), (SegmentSet{1}));

  // Reassign init subset to include 'b'. This should remove 'b' from all
  // segments.
  SubsetDefinition new_init;
  new_init.codepoints = {'b'};
  ASSERT_TRUE(info->ReassignInitSubset(*cache, new_init).ok());

  EXPECT_EQ(info->SegmentsForCodepoints({'a'}), (SegmentSet{0}));
  EXPECT_EQ(info->SegmentsForCodepoints({'b'}), (SegmentSet{}));
  EXPECT_EQ(info->SegmentsForCodepoints({'c'}), (SegmentSet{1}));
}

TEST_F(RequestedSegmentationInformationTest, NonInitFontNonEmptyGlyphs_Glyf) {
  std::unique_ptr<GlyphClosureCache> cache =
      *GlyphClosureCache::Create(roboto.get(), *resolver);
  // ' ' -> gid 4 (empty in glyf), 'a' -> gid 69, 'b' -> gid 70.
  std::vector<Segment> segments{
      {{' '}},
      {{'a'}},
      {{'b'}},
  };

  auto info_or = RequestedSegmentationInformation::Create(segments, {}, *cache);
  ASSERT_TRUE(info_or.ok()) << info_or.status();
  auto& info = *info_or;

  EXPECT_EQ(info->NonInitFontGlyphs(), (ift::common::GlyphSet{4, 69, 70}));
  EXPECT_EQ(info->NonInitFontNonEmptyGlyphs(), (ift::common::GlyphSet{69, 70}));

  // Moving 'a' into the init font removes gid 69 from both sets.
  SubsetDefinition new_init;
  new_init.codepoints = {'a'};
  ASSERT_TRUE(info->ReassignInitSubset(*cache, new_init).ok());
  EXPECT_EQ(info->NonInitFontGlyphs(), (ift::common::GlyphSet{4, 70}));
  EXPECT_EQ(info->NonInitFontNonEmptyGlyphs(), (ift::common::GlyphSet{70}));
}

TEST_F(RequestedSegmentationInformationTest,
       NonInitFontNonEmptyGlyphs_GlyfAndGvar) {
  hb_face_unique_ptr roboto_vf =
      from_file("ift/common/testdata/Roboto[wdth,wght].ttf");
  std::unique_ptr<GlyphClosureCache> cache =
      *GlyphClosureCache::Create(roboto_vf.get(), *resolver);
  // 0x0002 -> gid 2 (empty in both glyf and gvar)
  // ' ' -> gid 4 (empty in glyf, non-empty in gvar)
  // 'a' -> gid 69 (non-empty in both)
  std::vector<Segment> segments{
      {{0x0002}},
      {{' '}},
      {{'a'}},
  };

  auto info_or = RequestedSegmentationInformation::Create(segments, {}, *cache);
  ASSERT_TRUE(info_or.ok()) << info_or.status();
  auto& info = *info_or;

  EXPECT_EQ(info->NonInitFontGlyphs(), (ift::common::GlyphSet{2, 4, 69}));
  EXPECT_EQ(info->NonInitFontNonEmptyGlyphs(), (ift::common::GlyphSet{4, 69}));
}

TEST_F(RequestedSegmentationInformationTest, NonInitFontNonEmptyGlyphs_Cff2) {
  hb_face_unique_ptr noto_vf =
      from_file("ift/common/testdata/NotoSansJP-VF.subset.otf");
  std::unique_ptr<GlyphClosureCache> cache =
      *GlyphClosureCache::Create(noto_vf.get(), *resolver);
  SubsetDefinition empty_gid_seg;
  empty_gid_seg.gids = {1};  // gid 1 is 0-byte charstring in CFF2
  std::vector<Segment> segments{
      Segment{empty_gid_seg},
      Segment{{'A'}},  // gid 34
  };

  auto info_or = RequestedSegmentationInformation::Create(segments, {}, *cache);
  ASSERT_TRUE(info_or.ok()) << info_or.status();
  auto& info = *info_or;

  EXPECT_EQ(info->NonInitFontGlyphs(), (ift::common::GlyphSet{1, 34}));
  EXPECT_EQ(info->NonInitFontNonEmptyGlyphs(), (ift::common::GlyphSet{34}));
}

TEST_F(RequestedSegmentationInformationTest,
       MoveGlyphsToInitFontDoesNotExpandClosureOrMutateBaseline) {
  std::unique_ptr<GlyphClosureCache> cache =
      *GlyphClosureCache::Create(roboto.get(), *resolver);
  // ' ' -> gid 4 (empty), 'f' -> gid 74, 'i' -> gid 77, 'fi' -> gid 444.
  std::vector<Segment> segments{
      {{' '}},
      {{'f'}},
      {{'i'}},
  };

  SubsetDefinition init;
  AddInitSubsetDefaults(init);
  auto info_or =
      RequestedSegmentationInformation::Create(segments, init, *cache);
  ASSERT_TRUE(info_or.ok()) << info_or.status();
  auto& info = *info_or;

  const InitialFont& init_font = info->InitFont();
  const SubsetDefinition initial_closed_input =
      init_font.ClosedInputDefinition();
  EXPECT_EQ(initial_closed_input, init_font.TableKeyedDefinition());
  EXPECT_EQ(init_font.ClosedInputGlyphs(), (ift::common::GlyphSet{0}));
  EXPECT_EQ(init_font.GlyphKeyedGlyphs(), (ift::common::GlyphSet{0}));
  EXPECT_TRUE(info->NonClosedInputGlyphs().contains(4));
  EXPECT_TRUE(info->NonClosedInputGlyphs().contains(74));
  EXPECT_TRUE(info->NonClosedInputGlyphs().contains(77));
  EXPECT_TRUE(info->NonClosedInputGlyphs().contains(444));

  // Move 'f' (gid 74) and ' ' (gid 4) directly without running ExpandClosure.
  info->MoveGlyphsToInitFont({74, 4});

  EXPECT_EQ(init_font.GlyphKeyedGlyphs(), (ift::common::GlyphSet{0, 4, 74}));
  EXPECT_EQ(info->InitFontGlyphs(), (ift::common::GlyphSet{0, 4, 74}));
  EXPECT_FALSE(info->NonInitFontGlyphs().contains(4));
  EXPECT_FALSE(info->NonInitFontGlyphs().contains(74));
  EXPECT_TRUE(info->NonInitFontGlyphs().contains(77));
  EXPECT_TRUE(info->NonInitFontGlyphs().contains(444));
  EXPECT_FALSE(info->NonInitFontNonEmptyGlyphs().contains(74));
  EXPECT_TRUE(info->NonInitFontNonEmptyGlyphs().contains(77));
  EXPECT_TRUE(info->NonInitFontNonEmptyGlyphs().contains(444));

  // Frozen closed input baseline and segment definitions remain untouched.
  EXPECT_EQ(init_font.ClosedInputDefinition(), initial_closed_input);
  EXPECT_EQ(init_font.ClosedInputGlyphs(), (ift::common::GlyphSet{0}));
  EXPECT_TRUE(info->NonClosedInputGlyphs().contains(4));
  EXPECT_TRUE(info->NonClosedInputGlyphs().contains(74));
  EXPECT_TRUE(init_font.TableKeyedCodepoints().empty());
  EXPECT_TRUE(
      init_font.TableKeyedDefinitionWithoutDefaults().codepoints.empty());
  EXPECT_TRUE(
      init_font.TableKeyedDefinitionWithoutDefaults().feature_tags.empty());
  EXPECT_EQ(info->Segments().at(0).Definition().codepoints,
            (CodepointSet{' '}));
  EXPECT_EQ(info->Segments().at(1).Definition().codepoints,
            (CodepointSet{'f'}));
  EXPECT_EQ(info->Segments().at(2).Definition().codepoints,
            (CodepointSet{'i'}));
}

TEST_F(RequestedSegmentationInformationTest,
       AddToInitFontSegmentClearUnusedAndReplay) {
  std::unique_ptr<GlyphClosureCache> cache =
      *GlyphClosureCache::Create(roboto.get(), *resolver);
  SubsetDefinition smcp;
  smcp.feature_tags.insert(HB_TAG('s', 'm', 'c', 'p'));
  std::vector<Segment> segments{
      {{'a'}},
      {{'f'}},
      {{'i'}},
      Segment{smcp},
  };

  auto info_or = RequestedSegmentationInformation::Create(segments, {}, *cache);
  ASSERT_TRUE(info_or.ok()) << info_or.status();
  auto& info = *info_or;

  const SubsetDefinition expected_full_def = info->FullDefinition();
  const ift::common::GlyphSet expected_full_closure = info->FullClosure();

  info->MoveGlyphsToInitFont({69, 74});
  info->AddToInitFontSegment({'a', 'f'});
  info->AddToInitFontSegment(smcp);

  const InitialFont& init_font = info->InitFont();
  // Table-keyed sets are updated without ExpandClosure or removing items from
  // active segments.
  EXPECT_EQ(init_font.TableKeyedCodepoints(), (CodepointSet{'a', 'f'}));
  EXPECT_EQ(init_font.TableKeyedFeatures(), smcp.feature_tags);
  EXPECT_EQ(init_font.TableKeyedDefinitionWithoutDefaults().codepoints,
            (CodepointSet{'a', 'f'}));
  EXPECT_EQ(init_font.TableKeyedDefinitionWithoutDefaults().feature_tags,
            smcp.feature_tags);
  EXPECT_TRUE(init_font.ClosedInputDefinition().codepoints.empty());
  EXPECT_TRUE(init_font.ClosedInputDefinition().feature_tags.empty());
  EXPECT_EQ(info->SegmentsForCodepoints({'a'}), (SegmentSet{0}));
  EXPECT_EQ(info->SegmentsForCodepoints({'f'}), (SegmentSet{1}));
  EXPECT_EQ(init_font.UnsegmentedDefinition(),
            init_font.ClosedInputDefinition());

  EXPECT_TRUE(init_font.ContainsSegment({'a'}));
  EXPECT_TRUE(init_font.ContainsSegment(smcp));
  SubsetDefinition a_with_gid({'a'});
  a_with_gid.gids.insert(69);
  EXPECT_TRUE(init_font.ContainsSegment(a_with_gid));
  EXPECT_FALSE(init_font.ContainsSegment({'i'}));
  SubsetDefinition dlig;
  dlig.feature_tags.insert(HB_TAG('d', 'l', 'i', 'g'));
  EXPECT_FALSE(init_font.ContainsSegment(dlig));
  SubsetDefinition a_with_unmoved_gid({'a'});
  a_with_unmoved_gid.gids.insert(77);
  EXPECT_FALSE(init_font.ContainsSegment(a_with_unmoved_gid));

  // Clearing s0 ('a') moves its definition into UnsegmentedDefinition and drops
  // it from the segment map while leaving active segment s1 ('f') intact.
  info->ClearUnusedInitSegments({0});
  EXPECT_TRUE(info->Segments().at(0).Definition().Empty());
  EXPECT_EQ(info->Segments().at(1).Definition().codepoints,
            (CodepointSet{'f'}));
  EXPECT_EQ(info->SegmentsForCodepoints({'a'}), (SegmentSet{}));
  EXPECT_EQ(info->SegmentsForCodepoints({'f'}), (SegmentSet{1}));
  EXPECT_EQ(init_font.UnsegmentedDefinition().codepoints, (CodepointSet{'a'}));
  EXPECT_EQ(info->FullDefinition(), expected_full_def);

  // Replaying InitialFont on a fresh instance built from remaining Segments()
  // and ClosedInputDefinition() restores identical state.
  auto replayed_or = RequestedSegmentationInformation::Create(
      info->Segments(), init_font.ClosedInputDefinition(), *cache);
  ASSERT_TRUE(replayed_or.ok()) << replayed_or.status();
  auto& replayed = *replayed_or;
  ASSERT_TRUE(replayed->ReplayInitFontMoves(*cache, init_font).ok());

  EXPECT_EQ(replayed->FullDefinition(), info->FullDefinition());
  EXPECT_EQ(replayed->FullClosure(), expected_full_closure);
  EXPECT_EQ(replayed->InitFont(), init_font);
  EXPECT_EQ(replayed->InitFontSegment(), info->InitFontSegment());
  EXPECT_EQ(replayed->InitFontGlyphs(), info->InitFontGlyphs());
  EXPECT_EQ(replayed->NonInitFontGlyphs(), info->NonInitFontGlyphs());
  EXPECT_EQ(replayed->NonInitFontNonEmptyGlyphs(),
            info->NonInitFontNonEmptyGlyphs());
  EXPECT_EQ(replayed->NonClosedInputGlyphs(), info->NonClosedInputGlyphs());
}

TEST_F(RequestedSegmentationInformationTest,
       ReassignClosedInputRequiresSupersetAfterClosure) {
  {
    std::unique_ptr<GlyphClosureCache> cache =
        *GlyphClosureCache::Create(roboto.get(), *resolver);
    SubsetDefinition init({'a'});
    init.feature_tags.insert(HB_TAG('s', 'm', 'c', 'p'));
    init.gids.insert(444);  // unencoded 'fi' ligature in Roboto-Regular.ttf
    auto info_or = RequestedSegmentationInformation::Create({}, init, *cache);
    ASSERT_TRUE(info_or.ok()) << info_or.status();
    auto& info = *info_or;

    auto gid_a =
        ift::common::FontHelper::GetNominalGlyph(roboto.get(), 'a').value();
    auto gid_b =
        ift::common::FontHelper::GetNominalGlyph(roboto.get(), 'b').value();

    // Closure-implied superset succeeds: passing {gid_a, gid_b, 'c'} plus
    // smcp, dlig, and gid 444 reverse-maps gid_a -> 'a' and gid_b -> 'b' in
    // ExpandClosure.
    SubsetDefinition closure_superset({'c'});
    closure_superset.gids = {gid_a, gid_b, 444};
    closure_superset.feature_tags = {HB_TAG('s', 'm', 'c', 'p'),
                                     HB_TAG('d', 'l', 'i', 'g')};
    EXPECT_TRUE(info->ReassignInitSubset(*cache, closure_superset).ok());
    EXPECT_TRUE(
        info->InitFont().ClosedInputDefinition().codepoints.contains('a'));
    EXPECT_TRUE(
        info->InitFont().ClosedInputDefinition().codepoints.contains('b'));
    EXPECT_TRUE(
        info->InitFont().ClosedInputDefinition().codepoints.contains('c'));

    const InitialFont before_failures = info->InitFont();

    // Missing codepoint ('b') fails and leaves state unchanged.
    SubsetDefinition missing_cp({'a', 'c'});
    missing_cp.gids = {444};
    missing_cp.feature_tags = {HB_TAG('s', 'm', 'c', 'p'),
                               HB_TAG('d', 'l', 'i', 'g')};
    EXPECT_TRUE(
        absl::IsInvalidArgument(info->ReassignInitSubset(*cache, missing_cp)));
    EXPECT_EQ(info->InitFont(), before_failures);

    // Missing feature ('smcp') fails and leaves state unchanged.
    SubsetDefinition missing_feature({'a', 'b', 'c'});
    missing_feature.gids = {444};
    missing_feature.feature_tags = {HB_TAG('d', 'l', 'i', 'g')};
    EXPECT_TRUE(absl::IsInvalidArgument(
        info->ReassignInitSubset(*cache, missing_feature)));
    EXPECT_EQ(info->InitFont(), before_failures);

    // Missing unencoded glyph (444) fails and leaves state unchanged.
    SubsetDefinition missing_gid({'a', 'b', 'c'});
    missing_gid.feature_tags = {HB_TAG('s', 'm', 'c', 'p'),
                                HB_TAG('d', 'l', 'i', 'g')};
    EXPECT_TRUE(
        absl::IsInvalidArgument(info->ReassignInitSubset(*cache, missing_gid)));
    EXPECT_EQ(info->InitFont(), before_failures);
  }

  {
    hb_face_unique_ptr roboto_vf =
        from_file("ift/common/testdata/Roboto[wdth,wght].ttf");
    std::unique_ptr<GlyphClosureCache> cache =
        *GlyphClosureCache::Create(roboto_vf.get(), *resolver);
    SubsetDefinition init({'a'});
    init.design_space[HB_TAG('w', 'g', 'h', 't')] =
        *ift::common::AxisRange::Range(400.0f, 700.0f);
    auto info_or = RequestedSegmentationInformation::Create({}, init, *cache);
    ASSERT_TRUE(info_or.ok()) << info_or.status();
    auto& info = *info_or;
    const InitialFont before_failures = info->InitFont();

    // Narrowing wght range fails and leaves state unchanged.
    SubsetDefinition narrowed({'a'});
    narrowed.design_space[HB_TAG('w', 'g', 'h', 't')] =
        *ift::common::AxisRange::Range(400.0f, 600.0f);
    EXPECT_TRUE(
        absl::IsInvalidArgument(info->ReassignInitSubset(*cache, narrowed)));
    EXPECT_EQ(info->InitFont(), before_failures);

    // Restricting a previously unrestricted axis (wdth) fails.
    SubsetDefinition restricted_wdth({'a'});
    restricted_wdth.design_space[HB_TAG('w', 'g', 'h', 't')] =
        *ift::common::AxisRange::Range(400.0f, 700.0f);
    restricted_wdth.design_space[HB_TAG('w', 'd', 't', 'h')] =
        *ift::common::AxisRange::Range(75.0f, 100.0f);
    EXPECT_TRUE(absl::IsInvalidArgument(
        info->ReassignInitSubset(*cache, restricted_wdth)));
    EXPECT_EQ(info->InitFont(), before_failures);

    // Widening wght range succeeds.
    SubsetDefinition widened({'a', 'b'});
    widened.design_space[HB_TAG('w', 'g', 'h', 't')] =
        *ift::common::AxisRange::Range(300.0f, 800.0f);
    EXPECT_TRUE(info->ReassignInitSubset(*cache, widened).ok());

    // Dropping the wght restriction (full unrestricted axis) succeeds.
    SubsetDefinition unrestricted({'a', 'b'});
    EXPECT_TRUE(info->ReassignInitSubset(*cache, unrestricted).ok());
  }
}

}  // namespace ift::encoder
