#include "ift/freq/bigram_probability_calculator.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "ift/common/int_set.h"
#include "ift/encoder/segment.h"
#include "ift/encoder/segment_list.h"
#include "ift/encoder/subset_definition.h"
#include "ift/freq/probability_bound.h"
#include "ift/freq/probability_calculator.h"
#include "ift/freq/unicode_frequencies.h"

using ift::common::SegmentSet;
using ift::encoder::Segment;
using ift::encoder::SegmentList;
using ift::encoder::SubsetDefinition;

namespace ift::freq {

TEST(BigramProbabilityCalculatorTest, ComputeProbability) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 70}, {{'b', 'b'}, 60}, {{'c', 'c'}, 100}, {{'d', 'd'}, 50},

      {{'a', 'b'}, 40}, {{'a', 'c'}, 50}, {{'b', 'c'}, 60},  {{'a', 'd'}, 45},
      {{'b', 'd'}, 17},
  };

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));

  auto segments = SegmentList::Create({
      Segment{{'a'}},
      Segment{{'b'}},
      Segment{{'c'}},
      Segment{{'a', 'b'}},
      Segment{{'a', 'b', 'd'}},
      Segment{{}},
  });
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  ASSERT_EQ(seg_calc->ComputeProbability(5), (ProbabilityBound{1, 1}));

  ASSERT_EQ(seg_calc->ComputeProbability(0), (ProbabilityBound{0.7, 0.7}));
  ASSERT_EQ(seg_calc->ComputeProbability(1), (ProbabilityBound{0.6, 0.6}));
  ASSERT_EQ(seg_calc->ComputeProbability(2), (ProbabilityBound{1.0, 1.0}));

  double Pab = 0.70 + 0.60 - 0.40;  // 0.9
  ASSERT_EQ(seg_calc->ComputeProbability(3), (ProbabilityBound{Pab, Pab}));

  double Pbd = 0.60 + 0.50 - 0.17;
  double Pabd_upper =
      0.70 + 0.60 + 0.50 - 0.40 - 0.45;  // sum(Pi) - P(a and b) - P(a and d)
  auto b = seg_calc->ComputeProbability(4);
  ASSERT_DOUBLE_EQ(b.Min(), Pbd);
  ASSERT_DOUBLE_EQ(b.Max(), Pabd_upper);

  // Standalone SubsetDefinition overload matches bound segment computations.
  EXPECT_EQ(calc->ComputeProbability(SubsetDefinition{}),
            (ProbabilityBound{1, 1}));
  EXPECT_EQ(calc->ComputeProbability(SubsetDefinition{'a'}),
            (ProbabilityBound{0.7, 0.7}));
  EXPECT_EQ(calc->ComputeProbability(SubsetDefinition{'c'}),
            (ProbabilityBound{1.0, 1.0}));
  EXPECT_EQ(calc->ComputeProbability(SubsetDefinition{'a', 'b'}),
            (ProbabilityBound{Pab, Pab}));
  auto b_standalone = calc->ComputeProbability(SubsetDefinition{'a', 'b', 'd'});
  EXPECT_DOUBLE_EQ(b_standalone.Min(), Pbd);
  EXPECT_DOUBLE_EQ(b_standalone.Max(), Pabd_upper);
}

TEST(BigramProbabilityCalculatorTest, ComputeProbability_Cached) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 70}, {{'b', 'b'}, 60}, {{'c', 'c'}, 100},
      {{'a', 'b'}, 40}, {{'a', 'c'}, 50}, {{'b', 'c'}, 60},
  };

  auto segments = SegmentList::Create({
      Segment{{'a', 'b'}},
  });

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  // Call once to populate cache
  seg_calc->ComputeProbability(0);

  // Call again to use cache
  double Pab = 0.70 + 0.60 - 0.40;
  ASSERT_EQ(seg_calc->ComputeProbability(0), (ProbabilityBound{Pab, Pab}));
}

TEST(BigramProbabilityCalculatorTest, ComputeMergedProbability) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 70}, {{'b', 'b'}, 60}, {{'c', 'c'}, 100},

      {{'a', 'b'}, 40}, {{'a', 'c'}, 50}, {{'b', 'c'}, 60},
  };

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));

  auto segments = SegmentList::Create({
      Segment{{'a'}},
      Segment{{'b'}},
  });
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  double Pab = 0.70 + 0.60 - 0.40;
  ASSERT_EQ(seg_calc->ComputeMergedProbability({0, 1}),
            (ProbabilityBound{Pab, Pab}));
}

TEST(BigramProbabilityCalculatorTest, ComputeMergedProbability_Complex) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 70}, {{'b', 'b'}, 60}, {{'c', 'c'}, 100}, {{'d', 'd'}, 55},
      {{'e', 'e'}, 65}, {{'a', 'b'}, 40}, {{'a', 'c'}, 50},  {{'b', 'c'}, 60},
      {{'a', 'd'}, 30}, {{'b', 'd'}, 20}, {{'c', 'd'}, 35},  {{'a', 'e'}, 5},
      {{'b', 'e'}, 10}, {{'c', 'e'}, 15}, {{'d', 'e'}, 20},
  };

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));

  auto segments = SegmentList::Create({
      Segment{{'a', 'b'}},
      Segment{{'c', 'd'}},
      Segment{{'a', 'b', 'c', 'd'}},
  });
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  ProbabilityBound expected = seg_calc->ComputeProbability(2);
  ASSERT_EQ(seg_calc->ComputeMergedProbability({0, 1}), expected);

  expected = seg_calc->ComputeProbability(0);
  ASSERT_EQ(seg_calc->ComputeMergedProbability({0}), expected);

  segments->Reset({
      {{'a', 'd'}},
      {{'b', 'e'}},
      {{'c'}},
      {{'a', 'b', 'c', 'd', 'e'}},
  });

  expected = seg_calc->ComputeProbability(3);
  ProbabilityBound actual = seg_calc->ComputeMergedProbability({0, 1, 2});
  ASSERT_NEAR(actual.Min(), expected.Min(), 1e-9);
  ASSERT_NEAR(actual.Max(), expected.Max(), 1e-9);
}

TEST(BigramProbabilityCalculatorTest, ComputeProbability_Clamped) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 10}, {{'b', 'b'}, 20}, {{'c', 'c'}, 100},

      {{'a', 'b'}, 40}, {{'a', 'c'}, 50}, {{'b', 'c'}, 60},
  };

  auto segments = SegmentList::Create({
      Segment{{'a', 'b'}},
  });

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  auto b = seg_calc->ComputeProbability(0);
  ASSERT_DOUBLE_EQ(b.Min(), 0.2);  // P(b) sets a lower bound in this case
  ASSERT_DOUBLE_EQ(b.Max(), 0.2);
}

TEST(BigramProbabilityCalculatorTest, ComputeProbability_ClampedUpper) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 80}, {{'b', 'b'}, 90}, {{'c', 'c'}, 100},

      {{'a', 'b'}, 10}, {{'a', 'c'}, 10}, {{'b', 'c'}, 10},
  };

  auto segments = SegmentList::Create({
      Segment{{'a', 'b'}},
  });

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  auto b = seg_calc->ComputeProbability(0);
  ASSERT_DOUBLE_EQ(b.Min(), 1.0);  // P(b) sets a lower bound in this case
  ASSERT_DOUBLE_EQ(b.Max(), 1.0);
}

TEST(BigramProbabilityCalculatorTest, ComputeProbability_WithLayoutTags) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 70},
      {{'c', 'c'}, 100},
  };

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));

  SubsetDefinition def;
  def.feature_tags.insert(HB_TAG('l', 'i', 'g', 'a'));

  auto segments = SegmentList::Create({Segment{def}});
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  // P(liga) = 0.001
  // Codepoints are empty, so codepoints_bound = {0, 0}
  // Combined: Lower = max(0, 0.001) = 0.001
  //           Upper = min(1.0, 0 + 0.001) = 0.001
  ASSERT_EQ(seg_calc->ComputeProbability(0), (ProbabilityBound{0.001, 0.001}));
  ASSERT_EQ(calc->ComputeProbability(def), (ProbabilityBound{0.001, 0.001}));

  // Add another tag
  def.feature_tags.insert(HB_TAG('c', 'c', 'm', 'p'));
  segments->Set(0, Segment{def});

  // P(ccmp) = 0.001
  // Combined tags probability:
  // features_min = 0.001
  // features_sum = 0.002
  // Lower = max(0, 0.001) = 0.001
  // Upper = min(1.0, 0 + 0.002) = 0.002
  ASSERT_EQ(seg_calc->ComputeProbability(0), (ProbabilityBound{0.001, 0.002}));
  ASSERT_EQ(calc->ComputeProbability(def), (ProbabilityBound{0.001, 0.002}));

  // Add a codepoint
  def.codepoints.insert('a');
  segments->Set(0, Segment{def});
  // P(a) = 0.7
  // codepoints_bound = {0.7, 0.7}
  // Lower = max(0.7, 0.001) = 0.7
  // Upper = min(1.0, 0.7 + 0.002) = 0.702
  ASSERT_EQ(seg_calc->ComputeProbability(0), (ProbabilityBound{0.7, 0.702}));
  ASSERT_EQ(calc->ComputeProbability(def), (ProbabilityBound{0.7, 0.702}));
}

TEST(BigramProbabilityCalculatorTest, ComputeConjunctiveProbability) {
  UnicodeFrequencies freqs;
  BigramProbabilityCalculator calculator(std::move(freqs));

  ProbabilityBound b1(0.8, 0.9);
  ProbabilityBound b2(0.7, 0.8);

  std::vector<ProbabilityBound> bounds = {b1, b2};

  // sum(min) = 0.8 + 0.7 = 1.5
  // n = 2
  // min = 1.5 - 2 + 1 = 0.5
  // max = min(0.9, 0.8) = 0.8
  ProbabilityBound result = calculator.ComputeConjunctiveProbability(bounds);
  ASSERT_DOUBLE_EQ(result.Min(), 0.5);
  ASSERT_DOUBLE_EQ(result.Max(), 0.8);
}

TEST(BigramProbabilityCalculatorTest, ComputeConjunctiveProbability_Clamped) {
  UnicodeFrequencies freqs;
  BigramProbabilityCalculator calculator(std::move(freqs));

  ProbabilityBound b1(0.1, 0.2);
  ProbabilityBound b2(0.3, 0.4);
  ProbabilityBound b3(0.5, 0.6);

  std::vector<ProbabilityBound> bounds = {b1, b2, b3};

  // sum(min) = 0.1 + 0.3 + 0.5 = 0.9
  // n = 3
  // min = max(0.0, 0.9 - 3 + 1) = 0.0
  // max = min(0.2, 0.4, 0.6) = 0.2
  ProbabilityBound result = calculator.ComputeConjunctiveProbability(bounds);
  ASSERT_DOUBLE_EQ(result.Min(), 0.0);
  ASSERT_DOUBLE_EQ(result.Max(), 0.2);
}

TEST(BigramProbabilityCalculatorTest,
     ComputeConjunctiveProbabilitySingleSegment) {
  UnicodeFrequencies freqs;
  BigramProbabilityCalculator calculator(std::move(freqs));

  ProbabilityBound b1(0.1, 0.2);

  std::vector<ProbabilityBound> bounds = {b1};

  // sum(min) = 0.1
  // n = 1
  // min = 0.1 - 1 + 1 = 0.1
  // max = 0.2
  ProbabilityBound result = calculator.ComputeConjunctiveProbability(bounds);
  ASSERT_DOUBLE_EQ(result.Min(), 0.1);
  ASSERT_DOUBLE_EQ(result.Max(), 0.2);
}

TEST(BigramProbabilityCalculatorTest, ComputeConjunctiveProbabilityNoSegments) {
  UnicodeFrequencies freqs;
  BigramProbabilityCalculator calculator(std::move(freqs));

  std::vector<ProbabilityBound> bounds;

  // sum(min) = 0
  // n = 0
  // min = 0 - 0 + 1 = 1.0
  // max = 1.0
  ProbabilityBound result = calculator.ComputeConjunctiveProbability(bounds);
  ASSERT_DOUBLE_EQ(result.Min(), 1.0);
  ASSERT_DOUBLE_EQ(result.Max(), 1.0);
}

TEST(BigramProbabilityCalculatorTest, ComputeProbability_LruEviction) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 70}, {{'b', 'b'}, 60}, {{'c', 'c'}, 100},
      {{'a', 'b'}, 40}, {{'a', 'c'}, 50}, {{'b', 'c'}, 60},
  };

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies), 2);
  auto segments = SegmentList::Create({
      Segment{{'a'}},
      Segment{{'b'}},
      Segment{{'c'}},
  });
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  seg_calc->ComputeProbability(0);
  seg_calc->ComputeProbability(1);
  // Third entry will cause eviction
  seg_calc->ComputeProbability(2);
  // Recompute evicted entry
  ASSERT_EQ(seg_calc->ComputeProbability(0), (ProbabilityBound{0.7, 0.7}));
}

TEST(BigramProbabilityCalculatorTest, SegmentCacheAutoInvalidates) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 70}, {{'b', 'b'}, 60}, {{'c', 'c'}, 100},
      {{'a', 'b'}, 40}, {{'a', 'c'}, 50}, {{'b', 'c'}, 60},
  };

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));

  auto segments =
      SegmentList::Create({Segment{{'a'}}, Segment{{'b'}}, Segment{{'c'}}});
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);
  // Populate segment 0 with {'a'} (prob 0.7)
  EXPECT_EQ(seg_calc->ComputeProbability(0), (ProbabilityBound{0.7, 0.7}));

  // Mutate segment 0: cache automatically invalidates via epoch bump.
  segments->Set(0, Segment{{'b'}});
  EXPECT_EQ(seg_calc->ComputeProbability(0), (ProbabilityBound{0.6, 0.6}));

  // Reset segments: cache automatically invalidates and resizes.
  segments->Reset({Segment{{'a'}}, Segment{{'b'}}});
  double Pab = 0.70 + 0.60 - 0.40;
  EXPECT_EQ(seg_calc->ComputeMergedProbability(SegmentSet{0, 1}),
            (ProbabilityBound{Pab, Pab}));
  EXPECT_THROW(seg_calc->ComputeProbability(2), std::out_of_range);
}

TEST(BigramProbabilityCalculatorTest,
     ComputeMergedProbability_SegmentIndices_Complex) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 70}, {{'b', 'b'}, 60}, {{'c', 'c'}, 100}, {{'d', 'd'}, 55},
      {{'e', 'e'}, 65}, {{'a', 'b'}, 40}, {{'a', 'c'}, 50},  {{'b', 'c'}, 60},
      {{'a', 'd'}, 30}, {{'b', 'd'}, 20}, {{'c', 'd'}, 35},  {{'a', 'e'}, 5},
      {{'b', 'e'}, 10}, {{'c', 'e'}, 15}, {{'d', 'e'}, 20},
  };

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));

  auto segments = SegmentList::Create({
      Segment{{'a', 'b'}},
      Segment{{'c', 'd'}},
      Segment{{'e'}},
      Segment{{'a', 'b', 'c', 'd'}},
      Segment{{'a', 'b', 'e'}},
  });
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  // Empty segment set returns [1.0, 1.0].
  EXPECT_EQ(seg_calc->ComputeMergedProbability(SegmentSet{}),
            (ProbabilityBound{1.0, 1.0}));

  // Single segment delegates to ComputeProbability.
  ProbabilityBound expected_single = seg_calc->ComputeProbability(0);
  EXPECT_EQ(seg_calc->ComputeMergedProbability(SegmentSet{0}), expected_single);

  // Two multi-codepoint segments.
  ProbabilityBound expected_ab_cd = seg_calc->ComputeProbability(3);
  ProbabilityBound actual_ab_cd =
      seg_calc->ComputeMergedProbability(SegmentSet{0, 1});
  EXPECT_NEAR(actual_ab_cd.Min(), expected_ab_cd.Min(), 1e-9);
  EXPECT_NEAR(actual_ab_cd.Max(), expected_ab_cd.Max(), 1e-9);

  // Asymmetric sizes (2 codepoints + 1 codepoint, testing both orderings).
  ProbabilityBound expected_ab_e = seg_calc->ComputeProbability(4);
  ProbabilityBound actual_ab_e =
      seg_calc->ComputeMergedProbability(SegmentSet{0, 2});
  EXPECT_NEAR(actual_ab_e.Min(), expected_ab_e.Min(), 1e-9);
  EXPECT_NEAR(actual_ab_e.Max(), expected_ab_e.Max(), 1e-9);

  // Three segments with interleaved codepoints.
  segments->Reset({
      {{'a', 'd'}},
      {{'b', 'e'}},
      {{'c'}},
      {{'a', 'b', 'c', 'd', 'e'}},
  });
  ProbabilityBound expected_all = seg_calc->ComputeProbability(3);
  ProbabilityBound actual_all =
      seg_calc->ComputeMergedProbability(SegmentSet{0, 1, 2});
  EXPECT_NEAR(actual_all.Min(), expected_all.Min(), 1e-9);
  EXPECT_NEAR(actual_all.Max(), expected_all.Max(), 1e-9);
}

TEST(BigramProbabilityCalculatorTest,
     ComputeMergedProbability_AutoInvalidatesOnMutation) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 70}, {{'b', 'b'}, 60}, {{'c', 'c'}, 100}, {{'d', 'd'}, 50},
      {{'a', 'b'}, 40}, {{'a', 'c'}, 50}, {{'b', 'c'}, 60},  {{'a', 'd'}, 45},
      {{'b', 'd'}, 17},
  };

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));
  auto segments = SegmentList::Create({
      Segment{{'a'}},
      Segment{{'b'}},
      Segment{{'c'}},
  });
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  double Pab = 0.70 + 0.60 - 0.40;
  EXPECT_DOUBLE_EQ(seg_calc->ComputeMergedProbability(SegmentSet{0, 1}).Min(),
                   Pab);

  // Mutate segment 0 via AssignMerged: ComputeMergedProbability automatically
  // refreshes segment 0 while keeping segment 1 cached.
  segments->AssignMerged(0, SegmentSet{2}, Segment{{'d'}});
  double Pdb = 0.60 + 0.50 - 0.17;
  EXPECT_DOUBLE_EQ(seg_calc->ComputeMergedProbability(SegmentSet{0, 1}).Min(),
                   Pdb);
}

TEST(BigramProbabilityCalculatorTest,
     ComputeMergedProbability_SegmentIndices_ClampedAndWithFeatureTags) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 80}, {{'b', 'b'}, 90}, {{'c', 'c'}, 100},
      {{'a', 'b'}, 10}, {{'a', 'c'}, 10}, {{'b', 'c'}, 10},
  };

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));

  // Cross-segment pair saturation: P(a) + P(b) - P(a, b) = 0.8 + 0.9 - 0.1 >= 1
  auto segments = SegmentList::Create({
      Segment{{'a'}},
      Segment{{'b'}},
      Segment{{'c'}},
  });
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);

  EXPECT_EQ(seg_calc->ComputeMergedProbability(SegmentSet{0, 1}),
            (ProbabilityBound{1.0, 1.0}));
  // Single segment saturation ('c' has P = 1.0).
  EXPECT_EQ(seg_calc->ComputeMergedProbability(SegmentSet{0, 2}),
            (ProbabilityBound{1.0, 1.0}));

  // Segments with feature tags.
  SubsetDefinition def0{'a'};
  def0.feature_tags.insert(HB_TAG('l', 'i', 'g', 'a'));

  SubsetDefinition def1{};
  def1.feature_tags.insert(HB_TAG('c', 'c', 'm', 'p'));

  SubsetDefinition union_def = def0;
  union_def.Union(def1);

  segments->Reset({Segment{def0}, Segment{def1}, Segment{union_def}});

  // expected =
  // (0.8, 0.8) # codepoints only bound
  // (0.001, 0.001) # liga bound
  // (0.001, 0.001) # ccmp bound
  // = (max(0.8, 0.001, 0.001), sum(0.8, 0.001, 0.001)) = (0.8, 0.802)
  ProbabilityBound prob_union = seg_calc->ComputeProbability(2);
  EXPECT_EQ(prob_union, ProbabilityBound(0.8, 0.802));
  EXPECT_EQ(seg_calc->ComputeMergedProbability(SegmentSet{2}), prob_union);
  ProbabilityBound merged_prob =
      seg_calc->ComputeMergedProbability(SegmentSet{0, 1});
  EXPECT_EQ(merged_prob, prob_union);
}

TEST(BigramProbabilityCalculatorTest, BindToSegmentsValidationAndLifetime) {
  UnicodeFrequencies frequencies{
      {{'a', 'a'}, 70},
      {{'b', 'b'}, 60},
      {{'c', 'c'}, 100},
      {{'a', 'b'}, 40},
  };

  auto calc =
      std::make_shared<BigramProbabilityCalculator>(std::move(frequencies));
  auto segments = SegmentList::Create({Segment{{'a'}}, Segment{{'b'}}});

  EXPECT_TRUE(absl::IsInvalidArgument(
      ProbabilityCalculator::BindToSegments(nullptr, segments).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      ProbabilityCalculator::BindToSegments(calc, nullptr).status()));

  auto seg_calc = *ProbabilityCalculator::BindToSegments(calc, segments);
  calc.reset();
  segments.reset();

  double Pab = 0.70 + 0.60 - 0.40;
  EXPECT_EQ(seg_calc->ComputeProbability(0), (ProbabilityBound{0.7, 0.7}));
  EXPECT_EQ(seg_calc->ComputeMergedProbability({0, 1}),
            (ProbabilityBound{Pab, Pab}));
}

}  // namespace ift::freq
