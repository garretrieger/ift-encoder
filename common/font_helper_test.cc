#include "common/font_helper.h"

#include "gtest/gtest.h"

using absl::string_view;

namespace common {

class FontHelperTest : public ::testing::Test {
 protected:
  FontHelperTest() {
    hb_blob_t* blob =
        hb_blob_create_from_file("patch_subset/testdata/Roboto-Regular.ab.ttf");
    roboto_ab = hb_face_create(blob, 0);
    hb_blob_destroy(blob);

    blob = hb_blob_create_from_file(
        "patch_subset/testdata/NotoSansJP-Regular.otf");
    noto_sans_jp_otf = hb_face_create(blob, 0);
    hb_blob_destroy(blob);
  }

  ~FontHelperTest() { hb_face_destroy(roboto_ab); }

  hb_face_t* noto_sans_jp_otf;
  hb_face_t* roboto_ab;
};

TEST_F(FontHelperTest, ReadUInt32) {
  uint8_t input[] = {0x12, 0x34, 0x56, 0x78};
  auto s = FontHelper::ReadUInt32(string_view((const char*)input, 4));
  ASSERT_TRUE(s.ok()) << s.status();
  ASSERT_EQ(*s, 0x12345678);

  s = FontHelper::ReadUInt32(string_view((const char*)input, 3));
  ASSERT_FALSE(s.ok());
}

TEST_F(FontHelperTest, Loca) {
  auto s = FontHelper::Loca(roboto_ab);
  ASSERT_TRUE(s.ok()) << s.status();
  hb_blob_t* loca_blob =
      hb_face_reference_table(roboto_ab, HB_TAG('l', 'o', 'c', 'a'));
  uint32_t length = 0;
  EXPECT_EQ(s->data(), hb_blob_get_data(loca_blob, &length));
  EXPECT_EQ(s->size(), length);
  hb_blob_destroy(loca_blob);

  s = FontHelper::Loca(noto_sans_jp_otf);
  ASSERT_TRUE(absl::IsNotFound(s.status())) << s.status();
}

TEST_F(FontHelperTest, GetTags) {
  auto s = FontHelper::GetTags(roboto_ab);
  ASSERT_TRUE(s.contains(FontHelper::kLoca));
  ASSERT_TRUE(s.contains(FontHelper::kGlyf));
  ASSERT_FALSE(s.contains(FontHelper::kCFF));

  s = FontHelper::GetTags(noto_sans_jp_otf);
  ASSERT_FALSE(s.contains(FontHelper::kLoca));
  ASSERT_FALSE(s.contains(FontHelper::kGlyf));
  ASSERT_TRUE(s.contains(FontHelper::kCFF));
}

TEST_F(FontHelperTest, GetOrderedTags) {
  auto s = FontHelper::ToStrings(FontHelper::GetOrderedTags(roboto_ab));
  EXPECT_EQ(s[0], "gasp");
  EXPECT_EQ(s[1], "maxp");
  EXPECT_EQ(s[16], "glyf");
  EXPECT_EQ(s[17], "fpgm");
}

}  // namespace common