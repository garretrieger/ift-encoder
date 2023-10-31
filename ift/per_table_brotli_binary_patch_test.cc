#include "ift/per_table_brotli_binary_patch.h"

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "common/font_helper.h"
#include "gtest/gtest.h"
#include "hb.h"
#include "ift/proto/IFT.pb.h"
#include "patch_subset/brotli_binary_diff.h"
#include "patch_subset/brotli_binary_patch.h"
#include "patch_subset/font_data.h"

using absl::StatusOr;
using absl::string_view;
using common::FontHelper;
using ift::proto::PerTablePatch;
using patch_subset::BrotliBinaryDiff;
using patch_subset::BrotliBinaryPatch;
using patch_subset::FontData;

namespace ift {

class PerTableBrotliBinaryPatchTest : public ::testing::Test {
 protected:
  PerTableBrotliBinaryPatchTest() {
    BrotliBinaryDiff differ;

    FontData foo, bar, abc, def;
    foo.copy("foo");
    bar.copy("bar");
    abc.copy("abc");
    def.copy("def");

    assert(differ.Diff(foo, bar, &foo_to_bar).ok());
    assert(differ.Diff(abc, def, &abc_to_def).ok());
  }

  hb_tag_t tag1 = HB_TAG('t', 'a', 'g', '1');
  hb_tag_t tag2 = HB_TAG('t', 'a', 'g', '2');
  hb_tag_t tag3 = HB_TAG('t', 'a', 'g', '3');

  std::string tag1_str = FontHelper::ToString(tag1);
  std::string tag2_str = FontHelper::ToString(tag2);
  std::string tag3_str = FontHelper::ToString(tag3);

  FontData foo_to_bar;
  FontData abc_to_def;

  PerTableBrotliBinaryPatch patcher;
};

TEST_F(PerTableBrotliBinaryPatchTest, BasicPatch) {
  FontData before = FontHelper::BuildFont({
      {tag1, "foo"},
      {tag2, "abc"},
  });
  FontData after = FontHelper::BuildFont({
      {tag1, "bar"},
      {tag2, "def"},
  });

  PerTablePatch patch_proto;
  (*patch_proto.mutable_table_patches())[tag1_str] = foo_to_bar.string();
  (*patch_proto.mutable_table_patches())[tag2_str] = abc_to_def.string();
  std::string patch = patch_proto.SerializeAsString();
  FontData patch_data;
  patch_data.copy(patch.data(), patch.size());

  FontData result;
  auto sc = patcher.Patch(before, patch_data, &result);
  ASSERT_TRUE(sc.ok()) << sc;

  ASSERT_EQ(after.str(), result.str());
}

// TODO test more advanced cases.

}  // namespace ift