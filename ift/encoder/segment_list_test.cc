#include "ift/encoder/segment_list.h"

#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "ift/common/int_set.h"
#include "ift/encoder/segment.h"
#include "ift/encoder/subset_definition.h"

using ift::common::CodepointSet;
using ift::common::SegmentSet;

namespace ift::encoder {
namespace {

TEST(SegmentListTest, ConstructionAndInitialEpochs) {
  SegmentList from_vec(std::vector<Segment>{{{'a'}}, {{'b'}}, {{'c'}}});
  ASSERT_EQ(from_vec.size(), 3u);
  EXPECT_FALSE(from_vec.empty());
  EXPECT_EQ(from_vec.CurrentEpoch(), 1u);
  EXPECT_EQ(from_vec.Epoch(0), 1u);
  EXPECT_EQ(from_vec.Epoch(1), 1u);
  EXPECT_EQ(from_vec.Epoch(2), 1u);
  EXPECT_EQ(from_vec.at(1).Definition(), (SubsetDefinition{'b'}));

  std::vector<SubsetDefinition> defs = {{'x'}, {'y'}};
  SegmentList from_span(defs);
  ASSERT_EQ(from_span.size(), 2u);
  EXPECT_EQ(from_span.Epoch(0), 1u);
  EXPECT_EQ(from_span.Epoch(1), 1u);
  EXPECT_EQ(from_span[0].Definition(), (SubsetDefinition{'x'}));
  EXPECT_EQ(from_span[1].Definition(), (SubsetDefinition{'y'}));

  SegmentList from_init_list = {{{'m'}}, {{'n'}}};
  ASSERT_EQ(from_init_list.size(), 2u);
  EXPECT_EQ(from_init_list.Epoch(0), 1u);
  EXPECT_EQ(from_init_list.Epoch(1), 1u);

  SegmentList empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0u);

  auto shared_empty = SegmentList::Create();
  EXPECT_TRUE(shared_empty->empty());

  auto shared_vec = SegmentList::Create(std::vector<Segment>{{{'a'}}, {{'b'}}});
  ASSERT_EQ(shared_vec->size(), 2u);
  EXPECT_EQ(shared_vec->Epoch(0), 1u);
  EXPECT_EQ(shared_vec->at(1).Definition(), (SubsetDefinition{'b'}));

  auto shared_defs = SegmentList::FromDefinitions(defs);
  ASSERT_EQ(shared_defs->size(), 2u);
  EXPECT_EQ(shared_defs->Epoch(0), 1u);
  EXPECT_EQ(shared_defs->at(0).Definition(), (SubsetDefinition{'x'}));
  EXPECT_EQ(shared_defs->at(1).Definition(), (SubsetDefinition{'y'}));
}

TEST(SegmentListTest, SetAndClearBumpOnlyTargetedEpoch) {
  SegmentList segments = {{{'a'}}, {{'b'}}, {{'c'}}};
  EXPECT_EQ(segments.Epoch(0), 1u);
  EXPECT_EQ(segments.Epoch(1), 1u);
  EXPECT_EQ(segments.Epoch(2), 1u);

  segments.Set(1, Segment{{'d'}});
  EXPECT_EQ(segments.at(1).Definition(), (SubsetDefinition{'d'}));
  EXPECT_EQ(segments.Epoch(0), 1u);
  EXPECT_EQ(segments.Epoch(1), 2u);
  EXPECT_EQ(segments.Epoch(2), 1u);
  EXPECT_EQ(segments.CurrentEpoch(), 2u);

  segments.Clear(0);
  EXPECT_TRUE(segments.at(0).Definition().Empty());
  EXPECT_EQ(segments.Epoch(0), 3u);
  EXPECT_EQ(segments.Epoch(1), 2u);
  EXPECT_EQ(segments.Epoch(2), 1u);
  EXPECT_EQ(segments.CurrentEpoch(), 3u);
}

TEST(SegmentListTest, AssignMergedBumpsBaseAndToMergeOnly) {
  SegmentList segments = {{{'a'}}, {{'b'}}, {{'c'}}, {{'d'}}};

  segments.AssignMerged(0, SegmentSet{1, 3}, Segment{{'a', 'b', 'd'}});

  EXPECT_EQ(segments.at(0).Definition(), (SubsetDefinition{'a', 'b', 'd'}));
  EXPECT_TRUE(segments.at(1).Definition().Empty());
  EXPECT_EQ(segments.at(2).Definition(), (SubsetDefinition{'c'}));
  EXPECT_TRUE(segments.at(3).Definition().Empty());

  EXPECT_EQ(segments.Epoch(0), 2u);
  EXPECT_EQ(segments.Epoch(1), 2u);
  EXPECT_EQ(segments.Epoch(2), 1u);
  EXPECT_EQ(segments.Epoch(3), 2u);
  EXPECT_EQ(segments.CurrentEpoch(), 2u);
}

TEST(SegmentListTest, SubtractCodepointsBumpsOnlyWhenIntersecting) {
  SegmentList segments = {{{'a', 'b'}}, {{'c'}}};

  segments.SubtractCodepoints(0, CodepointSet{'z'});
  EXPECT_EQ(segments.Epoch(0), 1u);
  EXPECT_EQ(segments.Epoch(1), 1u);
  EXPECT_EQ(segments.CurrentEpoch(), 1u);
  EXPECT_EQ(segments.at(0).Definition(), (SubsetDefinition{'a', 'b'}));

  segments.SubtractCodepoints(0, CodepointSet{'b', 'z'});
  EXPECT_EQ(segments.Epoch(0), 2u);
  EXPECT_EQ(segments.Epoch(1), 1u);
  EXPECT_EQ(segments.CurrentEpoch(), 2u);
  EXPECT_EQ(segments.at(0).Definition(), (SubsetDefinition{'a'}));
}

TEST(SegmentListTest, ResetMonotonicallyIncreasesEpochsAcrossResizes) {
  SegmentList segments = {{{'a'}}, {{'b'}}, {{'c'}}};
  segments.Set(1, Segment{{'x'}});
  uint64_t epoch_before_reset = segments.CurrentEpoch();
  EXPECT_EQ(epoch_before_reset, 2u);

  // Shrink to 1.
  segments.Reset({Segment{{'p'}}});
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_GT(segments.Epoch(0), epoch_before_reset);
  uint64_t shrink_epoch = segments.Epoch(0);

  // Same size reset.
  segments.Reset({Segment{{'q'}}});
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_GT(segments.Epoch(0), shrink_epoch);
  uint64_t same_size_epoch = segments.Epoch(0);

  // Grow back to 3 — previously out-of-range indices receive the new epoch.
  segments.Reset({Segment{{'r'}}, Segment{{'s'}}, Segment{{'t'}}});
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_GT(segments.Epoch(0), same_size_epoch);
  EXPECT_EQ(segments.Epoch(1), segments.Epoch(0));
  EXPECT_EQ(segments.Epoch(2), segments.Epoch(0));
}

TEST(SegmentListTest, OutOfBoundsThrows) {
  SegmentList segments = {{{'a'}}, {{'b'}}};

  EXPECT_THROW(segments.at(2), std::out_of_range);
  EXPECT_THROW(segments.Epoch(2), std::out_of_range);
  EXPECT_THROW(segments.Set(2, Segment{{'c'}}), std::out_of_range);
  EXPECT_THROW(segments.Clear(2), std::out_of_range);
  EXPECT_THROW(segments.SubtractCodepoints(2, CodepointSet{'a'}),
               std::out_of_range);
  EXPECT_THROW(segments.AssignMerged(2, SegmentSet{0}, Segment{{'c'}}),
               std::out_of_range);
  EXPECT_THROW(segments.AssignMerged(0, SegmentSet{2}, Segment{{'a'}}),
               std::out_of_range);
}

}  // namespace
}  // namespace ift::encoder
