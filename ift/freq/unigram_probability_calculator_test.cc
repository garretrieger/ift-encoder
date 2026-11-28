#include "ift/freq/unigram_probability_calculator.h"

#include <memory>
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

TEST(UnigramProbabilityCalculatorTest, ComputeProbability) {
  UnicodeFrequenciesBuilder builder;
  builder.Add(1, 1, 10);
  builder.Add(2, 2, 20);
  builder.Add(3, 3, 5);

  auto calculator =
      std::make_shared<UnigramProbabilityCalculator>(builder.Build());

  auto segments = SegmentList::Create({Segment{{1, 2}}, Segment{{1, 3}}});
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calculator, segments);

  double p1 = 10.0 / 20.0;
  double p2 = 20.0 / 20.0;
  double expected_prob1 = 1.0 - (1.0 - p1) * (1.0 - p2);

  ProbabilityBound bound1 = seg_calc->ComputeProbability(0);
  EXPECT_DOUBLE_EQ(bound1.Min(), expected_prob1);
  EXPECT_DOUBLE_EQ(bound1.Max(), expected_prob1);
  EXPECT_DOUBLE_EQ(calculator->ComputeProbability(SubsetDefinition{1, 2}).Min(),
                   expected_prob1);

  double p3 = 5.0 / 20.0;
  double expected_prob2 = 1.0 - (1.0 - p1) * (1.0 - p3);

  ProbabilityBound bound2 = seg_calc->ComputeProbability(1);
  EXPECT_DOUBLE_EQ(bound2.Min(), expected_prob2);
  EXPECT_DOUBLE_EQ(bound2.Max(), expected_prob2);
  EXPECT_DOUBLE_EQ(calculator->ComputeProbability(SubsetDefinition{1, 3}).Min(),
                   expected_prob2);
}

TEST(UnigramProbabilityCalculatorTest, ComputeMergedProbability) {
  UnicodeFrequenciesBuilder builder;
  builder.Add(1, 1, 10);
  builder.Add(2, 2, 20);
  builder.Add(3, 3, 5);

  auto calculator =
      std::make_shared<UnigramProbabilityCalculator>(builder.Build());

  auto segments = SegmentList::Create({Segment{{1}}, Segment{{3}}});
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calculator, segments);

  double p1 = 10.0 / 20.0;
  double p3 = 5.0 / 20.0;
  double expected_prob1 = 1.0 - (1.0 - p1) * (1.0 - p3);

  ProbabilityBound bound = seg_calc->ComputeMergedProbability({0, 1});
  EXPECT_DOUBLE_EQ(bound.Min(), expected_prob1);
  EXPECT_DOUBLE_EQ(bound.Max(), expected_prob1);
}

TEST(UnigramProbabilityCalculatorTest, ComputeConjunctiveProbability) {
  UnicodeFrequenciesBuilder builder;
  builder.Add(1, 1, 10);

  UnigramProbabilityCalculator calculator(builder.Build());

  std::vector<ProbabilityBound> bounds{ProbabilityBound{0.2, 0.2}};
  ProbabilityBound bound = calculator.ComputeConjunctiveProbability(bounds);
  EXPECT_DOUBLE_EQ(bound.Min(), 0.2);
  EXPECT_DOUBLE_EQ(bound.Max(), 0.2);

  bounds = {ProbabilityBound{0.5, 0.5}, ProbabilityBound{0.7, 0.7}};
  bound = calculator.ComputeConjunctiveProbability(bounds);
  EXPECT_DOUBLE_EQ(bound.Min(), 0.5 * 0.7);
  EXPECT_DOUBLE_EQ(bound.Max(), 0.5 * 0.7);

  bounds = {ProbabilityBound{0.5, 0.5}, ProbabilityBound{0.2, 0.2},
            ProbabilityBound{0.7, 0.7}};
  bound = calculator.ComputeConjunctiveProbability(bounds);
  EXPECT_DOUBLE_EQ(bound.Min(), 0.5 * 0.7 * 0.2);
  EXPECT_DOUBLE_EQ(bound.Max(), 0.5 * 0.7 * 0.2);

  bounds = {};
  bound = calculator.ComputeConjunctiveProbability(bounds);
  EXPECT_DOUBLE_EQ(bound.Min(), 1.0);
  EXPECT_DOUBLE_EQ(bound.Max(), 1.0);
}

TEST(UnigramProbabilityCalculatorTest, SegmentCacheAutoInvalidates) {
  UnicodeFrequenciesBuilder builder;
  builder.Add(1, 1, 10);
  builder.Add(2, 2, 20);
  builder.Add(3, 3, 5);

  auto calculator =
      std::make_shared<UnigramProbabilityCalculator>(builder.Build());

  auto segments =
      SegmentList::Create({Segment{{1}}, Segment{{2}}, Segment{{3}}});
  auto seg_calc = *ProbabilityCalculator::BindToSegments(calculator, segments);
  double p1 = 10.0 / 20.0;
  double p3 = 5.0 / 20.0;

  // Populate segment 0 with {1}
  EXPECT_DOUBLE_EQ(seg_calc->ComputeProbability(0).Min(), p1);

  // Mutating segment 0 automatically invalidates its cache entry.
  segments->Set(0, Segment{{3}});
  EXPECT_DOUBLE_EQ(seg_calc->ComputeProbability(0).Min(), p3);

  // Resetting the SegmentList automatically invalidates and resizes.
  segments->Reset({Segment{{1}}, Segment{{2}}, Segment{{3}}});
  EXPECT_DOUBLE_EQ(seg_calc->ComputeProbability(0).Min(), p1);

  ProbabilityBound merged =
      seg_calc->ComputeMergedProbability(SegmentSet{0, 2});
  EXPECT_DOUBLE_EQ(merged.Min(), 1.0 - (1.0 - p1) * (1.0 - p3));
}

TEST(UnigramProbabilityCalculatorTest, BindToSegmentsValidationAndLifetime) {
  UnicodeFrequenciesBuilder builder;
  builder.Add(1, 1, 10);
  builder.Add(2, 2, 20);
  auto calculator =
      std::make_shared<UnigramProbabilityCalculator>(builder.Build());
  auto segments = SegmentList::Create({Segment{{1}}, Segment{{2}}});

  EXPECT_TRUE(absl::IsInvalidArgument(
      ProbabilityCalculator::BindToSegments(nullptr, segments).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      ProbabilityCalculator::BindToSegments(calculator, nullptr).status()));

  auto seg_calc = *ProbabilityCalculator::BindToSegments(calculator, segments);
  calculator.reset();
  segments.reset();

  double p1 = 10.0 / 20.0;
  double p2 = 20.0 / 20.0;
  EXPECT_DOUBLE_EQ(seg_calc->ComputeProbability(0).Min(), p1);
  EXPECT_DOUBLE_EQ(seg_calc->ComputeMergedProbability({0, 1}).Min(),
                   1.0 - (1.0 - p1) * (1.0 - p2));
}

}  // namespace ift::freq
