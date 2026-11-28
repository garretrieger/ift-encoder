#include "ift/freq/unigram_probability_calculator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "hb.h"
#include "ift/common/int_set.h"
#include "ift/common/try.h"
#include "ift/encoder/segment.h"
#include "ift/encoder/segment_list.h"
#include "ift/encoder/subset_definition.h"
#include "ift/encoder/types.h"
#include "ift/freq/probability_bound.h"
#include "ift/freq/probability_calculator.h"
#include "ift/freq/segment_probability_cache.h"

using absl::StatusOr;
using ift::common::SegmentSet;
using ift::encoder::Segment;
using ift::encoder::segment_index_t;
using ift::encoder::SegmentList;
using ift::encoder::SubsetDefinition;

namespace ift::freq {

namespace {

class UnigramSegmentProbabilityCalculator final
    : public SegmentProbabilityCalculator {
 public:
  UnigramSegmentProbabilityCalculator(
      std::shared_ptr<const UnigramProbabilityCalculator> calculator,
      SegmentProbabilityCache<ProbabilityBound> segment_cache)
      : calculator_(std::move(calculator)),
        segment_cache_(std::move(segment_cache)) {}

  ProbabilityBound ComputeProbability(
      segment_index_t segment_index) const override {
    return segment_cache_.GetOrCompute(
        segment_index, [this](const Segment& segment) {
          return calculator_->ComputeProbability(segment.Definition());
        });
  }

  ProbabilityBound ComputeMergedProbability(
      const SegmentSet& segment_indices) const override {
    // Note: this assumes that all segments are disjoint. Which we enforce for
    // the inputs to cost based merging.
    double probability_of_none = 1.0;
    for (segment_index_t s : segment_indices) {
      probability_of_none *= (1.0 - ComputeProbability(s).Min());
    }
    double p = 1.0 - probability_of_none;
    return {p, p};
  }

  ProbabilityBound ComputeConjunctiveProbability(
      const std::vector<ProbabilityBound>& bounds) const override {
    return calculator_->ComputeConjunctiveProbability(bounds);
  }

 private:
  std::shared_ptr<const UnigramProbabilityCalculator> calculator_;
  mutable SegmentProbabilityCache<ProbabilityBound> segment_cache_;
};

}  // namespace

UnigramProbabilityCalculator::UnigramProbabilityCalculator(
    UnicodeFrequencies frequencies, size_t max_cache_size)
    : frequencies_(std::move(frequencies)),
      cache_("unigram probability", max_cache_size) {}

ProbabilityBound UnigramProbabilityCalculator::ComputeProbability(
    const SubsetDefinition& definition) const {
  std::optional<double>& cached_bound = cache_[definition.codepoints];
  if (cached_bound.has_value()) {
    double value = *cached_bound;
    return ProbabilityBound(value, value);
  }

  double probability_of_none = 1.0;
  for (uint32_t cp : definition.codepoints) {
    probability_of_none *= (1.0 - frequencies_.ProbabilityFor(cp));
  }

  for (hb_tag_t tag : definition.feature_tags) {
    probability_of_none *= (1.0 - frequencies_.ProbabilityForLayoutTag(tag));
  }

  double probability = 1.0 - probability_of_none;
  cached_bound = probability;
  return {probability, probability};
}

ProbabilityBound UnigramProbabilityCalculator::ComputeConjunctiveProbability(
    const std::vector<ProbabilityBound>& bounds) const {
  double probability = 1.0;
  for (const auto& bound : bounds) {
    probability *= bound.Value();
  }
  return {probability, probability};
}

StatusOr<std::unique_ptr<SegmentProbabilityCalculator>>
UnigramProbabilityCalculator::BindToSegmentsImpl(
    std::shared_ptr<const ProbabilityCalculator> self,
    std::shared_ptr<const SegmentList> segments) const {
  auto cache = TRY(
      SegmentProbabilityCache<ProbabilityBound>::Create(std::move(segments)));
  return std::make_unique<UnigramSegmentProbabilityCalculator>(
      std::static_pointer_cast<const UnigramProbabilityCalculator>(
          std::move(self)),
      std::move(cache));
}

}  // namespace ift::freq
