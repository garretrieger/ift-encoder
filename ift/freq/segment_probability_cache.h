#ifndef IFT_FREQ_SEGMENT_PROBABILITY_CACHE_H_
#define IFT_FREQ_SEGMENT_PROBABILITY_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ift/encoder/segment.h"
#include "ift/encoder/segment_list.h"
#include "ift/encoder/types.h"
#include "ift/freq/probability_bound.h"

namespace ift::freq {

// Caches per-segment probability entries indexed by segment_index_t for a
// bound SegmentList. Entries automatically invalidate whenever the
// SegmentList's per-segment epoch advances. Out-of-bounds accesses fail.
template <typename Entry = ProbabilityBound>
class SegmentProbabilityCache {
 public:
  static absl::StatusOr<SegmentProbabilityCache<Entry>> Create(
      std::shared_ptr<const ift::encoder::SegmentList> segments) {
    if (!segments) {
      return absl::InvalidArgumentError("segments must not be null.");
    }
    return SegmentProbabilityCache<Entry>(std::move(segments));
  }

  void SyncSize() {
    size_t num_segments = segments_->size();
    if (entries_.size() != num_segments) {
      entries_.resize(num_segments);
    }
  }

  template <typename ComputeFn>
  const Entry& GetOrCompute(ift::encoder::segment_index_t segment_index,
                            ComputeFn&& compute_fn) {
    SyncSize();
    uint64_t current_epoch = segments_->Epoch(segment_index);
    Slot& slot = entries_.at(segment_index);
    if (slot.epoch != current_epoch) {
      slot.value =
          std::forward<ComputeFn>(compute_fn)(segments_->at(segment_index));
      slot.epoch = current_epoch;
    }
    return slot.value;
  }

  const ift::encoder::SegmentList& Segments() const { return *segments_; }

 private:
  explicit SegmentProbabilityCache(
      std::shared_ptr<const ift::encoder::SegmentList> segments)
      : segments_(std::move(segments)), entries_(segments_->size()) {}

  struct Slot {
    // Valid SegmentList epochs start at 1; 0 marks an unpopulated slot.
    uint64_t epoch = 0;
    Entry value{};
  };

  std::shared_ptr<const ift::encoder::SegmentList> segments_;
  std::vector<Slot> entries_;
};

}  // namespace ift::freq

#endif  // IFT_FREQ_SEGMENT_PROBABILITY_CACHE_H_
