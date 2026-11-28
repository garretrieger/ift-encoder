#ifndef IFT_FREQ_MOCK_PROBABILITY_CALCULATOR_H_
#define IFT_FREQ_MOCK_PROBABILITY_CALCULATOR_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ift/common/int_set.h"
#include "ift/encoder/segment.h"
#include "ift/encoder/segment_list.h"
#include "ift/encoder/subset_definition.h"
#include "ift/encoder/types.h"
#include "ift/freq/probability_bound.h"
#include "ift/freq/probability_calculator.h"

namespace ift::freq {

class MockProbabilityCalculator : public ProbabilityCalculator {
 public:
  explicit MockProbabilityCalculator(
      std::vector<std::pair<ift::encoder::Segment, double>> segments)
      : segments_(std::move(segments)) {}

  absl::string_view Name() const override {
    return "MockProbabilityCalculator";
  }

  ift::common::CodepointSet CoveredCodepoints() const override {
    ift::common::CodepointSet out;
    for (const auto& [segment, prob] : segments_) {
      if (prob > 0.0) {
        out.union_set(segment.Definition().codepoints);
      }
    }
    return out;
  }

  ProbabilityBound ComputeProbability(uint32_t codepoint) const override {
    return ComputeProbability(ift::encoder::SubsetDefinition{codepoint});
  }

  ProbabilityBound ComputeProbability(
      const ift::encoder::SubsetDefinition& definition) const override {
    for (const auto& [segment, prob] : segments_) {
      if (segment.Definition() == definition) {
        return {prob, prob};
      }
    }
    return {0.0, 0.0};
  }

  ProbabilityBound ComputeConjunctiveProbability(
      const std::vector<ProbabilityBound>& bounds) const override {
    double probability = 1.0;
    for (const auto& bound : bounds) {
      probability *= bound.Value();
    }
    return {probability, probability};
  }

 protected:
  absl::StatusOr<std::unique_ptr<SegmentProbabilityCalculator>>
  BindToSegmentsImpl(
      std::shared_ptr<const ProbabilityCalculator> self,
      std::shared_ptr<const ift::encoder::SegmentList> segments) const override;

 private:
  std::vector<std::pair<ift::encoder::Segment, double>> segments_;
};

class MockSegmentProbabilityCalculator final
    : public SegmentProbabilityCalculator {
 public:
  MockSegmentProbabilityCalculator(
      std::shared_ptr<const MockProbabilityCalculator> calculator,
      std::shared_ptr<const ift::encoder::SegmentList> segments)
      : calculator_(std::move(calculator)), segments_(std::move(segments)) {}

  ProbabilityBound ComputeProbability(
      ift::encoder::segment_index_t segment_index) const override {
    return calculator_->ComputeProbability(
        segments_->at(segment_index).Definition());
  }

  ProbabilityBound ComputeMergedProbability(
      const ift::common::SegmentSet& segment_indices) const override {
    ift::encoder::SubsetDefinition merged;
    for (ift::encoder::segment_index_t s : segment_indices) {
      merged.Union(segments_->at(s).Definition());
    }
    return calculator_->ComputeProbability(merged);
  }

  ProbabilityBound ComputeConjunctiveProbability(
      const std::vector<ProbabilityBound>& bounds) const override {
    return calculator_->ComputeConjunctiveProbability(bounds);
  }

 private:
  std::shared_ptr<const MockProbabilityCalculator> calculator_;
  std::shared_ptr<const ift::encoder::SegmentList> segments_;
};

inline absl::StatusOr<std::unique_ptr<SegmentProbabilityCalculator>>
MockProbabilityCalculator::BindToSegmentsImpl(
    std::shared_ptr<const ProbabilityCalculator> self,
    std::shared_ptr<const ift::encoder::SegmentList> segments) const {
  if (!self) {
    return absl::InvalidArgumentError("calculator must not be null.");
  }
  if (!segments) {
    return absl::InvalidArgumentError("segments must not be null.");
  }
  return std::make_unique<MockSegmentProbabilityCalculator>(
      std::static_pointer_cast<const MockProbabilityCalculator>(
          std::move(self)),
      std::move(segments));
}

}  // namespace ift::freq

#endif  // IFT_FREQ_MOCK_PROBABILITY_CALCULATOR_H_
