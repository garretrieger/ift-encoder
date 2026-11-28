#ifndef IFT_FREQ_PROBABILITY_CALCULATOR_H_
#define IFT_FREQ_PROBABILITY_CALCULATOR_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ift/common/int_set.h"
#include "ift/encoder/segment_list.h"
#include "ift/encoder/subset_definition.h"
#include "ift/encoder/types.h"
#include "ift/freq/probability_bound.h"

namespace ift::freq {

// Computes segment and merged-segment probabilities for a specific SegmentList,
// automatically tracking per-segment cache validity via SegmentList epochs.
class SegmentProbabilityCalculator {
 public:
  virtual ~SegmentProbabilityCalculator() = default;

  virtual ProbabilityBound ComputeProbability(
      ift::encoder::segment_index_t segment_index) const = 0;

  virtual ProbabilityBound ComputeMergedProbability(
      const ift::common::SegmentSet& segment_indices) const = 0;

  // Compute and return the probability bounds on a page intersecting
  // all of the input segments.
  virtual ProbabilityBound ComputeConjunctiveProbability(
      const std::vector<ProbabilityBound>& bounds) const = 0;
};

class ProbabilityCalculator {
 public:
  virtual ~ProbabilityCalculator() = default;

  virtual absl::string_view Name() const = 0;

  virtual ift::common::CodepointSet CoveredCodepoints() const = 0;

  virtual ProbabilityBound ComputeProbability(uint32_t codepoint) const = 0;

  virtual ProbabilityBound ComputeProbability(
      const ift::encoder::SubsetDefinition& definition) const = 0;

  // Compute and return the probability bounds on a page intersecting
  // all of the input segments.
  virtual ProbabilityBound ComputeConjunctiveProbability(
      const std::vector<ProbabilityBound>& bounds) const = 0;

  // Returns a calculator bound to `calculator` and `segments`, holding shared
  // ownership of both.
  static absl::StatusOr<std::unique_ptr<SegmentProbabilityCalculator>>
  BindToSegments(std::shared_ptr<const ProbabilityCalculator> calculator,
                 std::shared_ptr<const ift::encoder::SegmentList> segments) {
    if (!calculator) {
      return absl::InvalidArgumentError("calculator must not be null.");
    }
    if (!segments) {
      return absl::InvalidArgumentError("segments must not be null.");
    }
    const ProbabilityCalculator* raw = calculator.get();
    return raw->BindToSegmentsImpl(std::move(calculator), std::move(segments));
  }

 protected:
  virtual absl::StatusOr<std::unique_ptr<SegmentProbabilityCalculator>>
  BindToSegmentsImpl(
      std::shared_ptr<const ProbabilityCalculator> self,
      std::shared_ptr<const ift::encoder::SegmentList> segments) const = 0;
};

}  // namespace ift::freq

#endif  // IFT_FREQ_PROBABILITY_CALCULATOR_H_
