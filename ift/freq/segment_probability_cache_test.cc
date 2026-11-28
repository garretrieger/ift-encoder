#include "ift/freq/segment_probability_cache.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "ift/common/int_set.h"
#include "ift/encoder/segment.h"
#include "ift/encoder/segment_list.h"
#include "ift/freq/probability_bound.h"

using ift::common::CodepointSet;
using ift::common::SegmentSet;
using ift::encoder::Segment;
using ift::encoder::SegmentList;
using ift::freq::ProbabilityBound;
using ift::freq::SegmentProbabilityCache;

namespace {

TEST(SegmentProbabilityCacheTest, CreateRejectsNullSegments) {
  auto cache = SegmentProbabilityCache<ProbabilityBound>::Create(nullptr);
  EXPECT_TRUE(absl::IsInvalidArgument(cache.status()));
}

TEST(SegmentProbabilityCacheTest, HitsAndAutomaticMutationInvalidation) {
  auto segments = SegmentList::Create(
      {Segment{{'a'}}, Segment{{'b'}}, Segment{{'c'}}, Segment{{'d', 'e'}}});
  auto cache_or = SegmentProbabilityCache<ProbabilityBound>::Create(segments);
  ASSERT_TRUE(cache_or.ok()) << cache_or.status();
  auto cache = std::move(*cache_or);

  int compute_calls = 0;
  auto compute = [&](const Segment& seg) {
    compute_calls++;
    double p = static_cast<double>(*seg.Definition().codepoints.min()) / 100.0;
    return ProbabilityBound(p, p);
  };

  // Initial miss then hit for segment 0 and 1.
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(0, compute).Min(), 0.97);
  EXPECT_EQ(compute_calls, 1);
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(0, compute).Min(), 0.97);
  EXPECT_EQ(compute_calls, 1);

  EXPECT_DOUBLE_EQ(cache.GetOrCompute(1, compute).Min(), 0.98);
  EXPECT_EQ(compute_calls, 2);

  // Set invalidates only the mutated segment.
  segments->Set(0, Segment{{'z'}});
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(0, compute).Min(), 1.22);
  EXPECT_EQ(compute_calls, 3);
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(1, compute).Min(), 0.98);
  EXPECT_EQ(compute_calls, 3);

  // AssignMerged invalidates base and to_merge, leaving untouched segments hot.
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(2, compute).Min(), 0.99);
  EXPECT_EQ(compute_calls, 4);
  segments->AssignMerged(1, SegmentSet{3}, Segment{{'x'}});
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(1, compute).Min(), 1.20);
  EXPECT_EQ(compute_calls, 5);
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(2, compute).Min(), 0.99);
  EXPECT_EQ(compute_calls, 5);

  // SubtractCodepoints invalidates only when intersecting.
  segments->Set(3, Segment{{'d', 'e'}});
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(3, compute).Min(), 1.00);
  EXPECT_EQ(compute_calls, 6);

  segments->SubtractCodepoints(3, CodepointSet{'q'});
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(3, compute).Min(), 1.00);
  EXPECT_EQ(compute_calls, 6);

  segments->SubtractCodepoints(3, CodepointSet{'d'});
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(3, compute).Min(), 1.01);
  EXPECT_EQ(compute_calls, 7);

  // Clear invalidates only the cleared segment.
  segments->Set(0, Segment{{'a'}});
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(0, compute).Min(), 0.97);
  EXPECT_EQ(compute_calls, 8);
  segments->Clear(0);
  int empty_calls = 0;
  auto compute_or_empty = [&](const Segment& seg) {
    empty_calls++;
    return seg.Definition().Empty() ? ProbabilityBound(1.0, 1.0)
                                    : ProbabilityBound(0.0, 0.0);
  };
  EXPECT_EQ(cache.GetOrCompute(0, compute_or_empty),
            ProbabilityBound(1.0, 1.0));
  EXPECT_EQ(empty_calls, 1);
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(2, compute).Min(), 0.99);
  EXPECT_EQ(compute_calls, 8);
}

TEST(SegmentProbabilityCacheTest, ResetAndOutOfBounds) {
  auto segments =
      SegmentList::Create({Segment{{'a'}}, Segment{{'b'}}, Segment{{'c'}}});
  auto cache_or = SegmentProbabilityCache<ProbabilityBound>::Create(segments);
  ASSERT_TRUE(cache_or.ok()) << cache_or.status();
  auto cache = std::move(*cache_or);

  int compute_calls = 0;
  auto compute = [&](const Segment& seg) {
    compute_calls++;
    double p = static_cast<double>(*seg.Definition().codepoints.min());
    return ProbabilityBound(p, p);
  };

  EXPECT_DOUBLE_EQ(cache.GetOrCompute(0, compute).Min(), 'a');
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(2, compute).Min(), 'c');
  EXPECT_EQ(compute_calls, 2);

  EXPECT_THROW(cache.GetOrCompute(3, compute), std::out_of_range);

  // Shrink via Reset: old entries are invalidated and out-of-bounds throws.
  segments->Reset({Segment{{'x'}}});
  EXPECT_THROW(cache.GetOrCompute(1, compute), std::out_of_range);
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(0, compute).Min(), 'x');
  EXPECT_EQ(compute_calls, 3);

  // Grow back via Reset: previously populated slot 2 must not return stale 'c'.
  segments->Reset({Segment{{'m'}}, Segment{{'n'}}, Segment{{'o'}}});
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(0, compute).Min(), 'm');
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(1, compute).Min(), 'n');
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(2, compute).Min(), 'o');
  EXPECT_EQ(compute_calls, 6);
}

TEST(SegmentProbabilityCacheTest, KeepsSegmentListAliveAfterCallerDropsRef) {
  auto segments = SegmentList::Create({Segment{{'a'}}, Segment{{'b'}}});
  auto cache = *SegmentProbabilityCache<ProbabilityBound>::Create(segments);
  segments.reset();

  int compute_calls = 0;
  auto compute = [&](const Segment& seg) {
    compute_calls++;
    double p = static_cast<double>(*seg.Definition().codepoints.min());
    return ProbabilityBound(p, p);
  };

  EXPECT_DOUBLE_EQ(cache.GetOrCompute(0, compute).Min(), 'a');
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(0, compute).Min(), 'a');
  EXPECT_DOUBLE_EQ(cache.GetOrCompute(1, compute).Min(), 'b');
  EXPECT_EQ(compute_calls, 2);
}

}  // namespace
