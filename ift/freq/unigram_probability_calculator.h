#ifndef IFT_FREQ_UNIGRAM_PROBABILITY_CALCULATOR_H_
#define IFT_FREQ_UNIGRAM_PROBABILITY_CALCULATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ift/common/int_set.h"
#include "ift/encoder/segment_list.h"
#include "ift/encoder/subset_definition.h"
#include "ift/freq/bigram_probability_calculator.h"
#include "ift/freq/lru_cache.h"
#include "ift/freq/probability_bound.h"
#include "ift/freq/probability_calculator.h"
#include "ift/freq/unicode_frequencies.h"

namespace ift::freq {

constexpr size_t UNIGRAM_PROBABILITY_CACHE_SIZE = 300000;

// The UnigramProbabilityCalculator calculates segment probabilites of occurence
// using unigram's (ie. one probability per codepoint). Because no additional
// probability data is present (such as co-occurrence probabilities) the
// calculations assume that these unigram probabilities are fully independent.
class UnigramProbabilityCalculator : public ProbabilityCalculator {
 public:
  explicit UnigramProbabilityCalculator(
      UnicodeFrequencies frequencies,
      size_t max_cache_size = UNIGRAM_PROBABILITY_CACHE_SIZE);

  absl::string_view Name() const override { return frequencies_.Name(); }

  ift::common::CodepointSet CoveredCodepoints() const override {
    return frequencies_.CoveredCodepoints();
  }

  ProbabilityBound ComputeProbability(uint32_t codepoint) const override {
    double p = frequencies_.ProbabilityFor(codepoint);
    return {p, p};
  }

  ProbabilityBound ComputeProbability(
      const ift::encoder::SubsetDefinition& definition) const override;

  ProbabilityBound ComputeConjunctiveProbability(
      const std::vector<ProbabilityBound>& bounds) const override;

  std::shared_ptr<BigramProbabilityCalculator> ToBigramCalculator() && {
    return std::make_shared<BigramProbabilityCalculator>(
        std::move(frequencies_));
  }

 protected:
  absl::StatusOr<std::unique_ptr<SegmentProbabilityCalculator>>
  BindToSegmentsImpl(
      std::shared_ptr<const ProbabilityCalculator> self,
      std::shared_ptr<const ift::encoder::SegmentList> segments) const override;

 private:
  UnicodeFrequencies frequencies_;
  mutable LruCache<ift::common::CodepointSet, std::optional<double>> cache_;
};

}  // namespace ift::freq

#endif  // IFT_FREQ_UNIGRAM_PROBABILITY_CALCULATOR_H_
