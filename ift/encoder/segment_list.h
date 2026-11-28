#ifndef IFT_ENCODER_SEGMENT_LIST_H_
#define IFT_ENCODER_SEGMENT_LIST_H_

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "ift/common/int_set.h"
#include "ift/encoder/segment.h"
#include "ift/encoder/subset_definition.h"
#include "ift/encoder/types.h"

namespace ift::encoder {

// Stores the sequence of segments and tracks a strictly increasing per-segment
// modification epoch so bound segment probability caches invalidate
// automatically without manual invalidation calls.
class SegmentList {
 public:
  static std::shared_ptr<SegmentList> Create() {
    return std::make_shared<SegmentList>();
  }

  static std::shared_ptr<SegmentList> Create(std::vector<Segment> segments) {
    return std::make_shared<SegmentList>(std::move(segments));
  }

  static std::shared_ptr<SegmentList> FromDefinitions(
      absl::Span<const SubsetDefinition> definitions) {
    return std::make_shared<SegmentList>(definitions);
  }

  SegmentList() = default;

  explicit SegmentList(std::vector<Segment> segments)
      : segments_(std::move(segments)),
        epochs_(segments_.size(), 1),
        current_epoch_(1) {}

  explicit SegmentList(absl::Span<const SubsetDefinition> definitions)
      : epochs_(definitions.size(), 1), current_epoch_(1) {
    segments_.reserve(definitions.size());
    for (const SubsetDefinition& def : definitions) {
      segments_.emplace_back(def);
    }
  }

  SegmentList(std::initializer_list<Segment> segments)
      : segments_(segments), epochs_(segments_.size(), 1), current_epoch_(1) {}

  // Non-copyable and non-movable so pointers held by bound caches remain
  // valid for the lifetime of the SegmentList.
  SegmentList(const SegmentList&) = delete;
  SegmentList& operator=(const SegmentList&) = delete;
  SegmentList(SegmentList&&) = delete;
  SegmentList& operator=(SegmentList&&) = delete;

  size_t size() const { return segments_.size(); }
  bool empty() const { return segments_.empty(); }

  const Segment& at(segment_index_t index) const { return segments_.at(index); }
  const Segment& operator[](segment_index_t index) const {
    return segments_[index];
  }

  auto begin() const { return segments_.begin(); }
  auto end() const { return segments_.end(); }

  std::vector<Segment> CopySegments() const { return segments_; }

  // Valid epochs start at 1 (0 is reserved for unpopulated cache slots).
  uint64_t Epoch(segment_index_t index) const { return epochs_.at(index); }
  uint64_t CurrentEpoch() const { return current_epoch_; }

  void Set(segment_index_t index, Segment new_segment) {
    segments_.at(index) = std::move(new_segment);
    epochs_.at(index) = ++current_epoch_;
  }

  void Clear(segment_index_t index) {
    segments_.at(index).Clear();
    epochs_.at(index) = ++current_epoch_;
  }

  void AssignMerged(segment_index_t base,
                    const ift::common::SegmentSet& to_merge,
                    Segment merged_segment) {
    segments_.at(base) = std::move(merged_segment);
    for (segment_index_t s : to_merge) {
      segments_.at(s).Clear();
    }
    uint64_t epoch = ++current_epoch_;
    epochs_.at(base) = epoch;
    for (segment_index_t s : to_merge) {
      epochs_.at(s) = epoch;
    }
  }

  void SubtractCodepoints(segment_index_t index,
                          const ift::common::CodepointSet& codepoints) {
    Segment& segment = segments_.at(index);
    if (!segment.Definition().codepoints.intersects(codepoints)) {
      return;
    }
    segment.Definition().codepoints.subtract(codepoints);
    epochs_.at(index) = ++current_epoch_;
  }

  void Reset(std::vector<Segment> segments) {
    ++current_epoch_;
    segments_ = std::move(segments);
    epochs_.assign(segments_.size(), current_epoch_);
  }

 private:
  std::vector<Segment> segments_;
  std::vector<uint64_t> epochs_;
  uint64_t current_epoch_ = 1;
};

}  // namespace ift::encoder

#endif  // IFT_ENCODER_SEGMENT_LIST_H_
