#ifndef IFT_ENCODER_INVALIDATION_SET_H_
#define IFT_ENCODER_INVALIDATION_SET_H_

#include <optional>
#include <utility>

#include "ift/common/int_set.h"
#include "ift/encoder/types.h"

namespace ift::encoder {

struct InvalidationSet {
  static InvalidationSet Empty() { return InvalidationSet(); }

  explicit InvalidationSet(ift::common::GlyphSet glyphs_,
                           ift::common::SegmentSet segments_,
                           segment_index_t base_segment_index)
      : glyphs(std::move(glyphs_)),
        segments(std::move(segments_)),
        base_segment(base_segment_index) {
    segments.insert(base_segment_index);
  }

  ift::common::GlyphSet glyphs;
  ift::common::SegmentSet segments;
  std::optional<segment_index_t> base_segment;

 private:
  InvalidationSet() = default;
};

}  // namespace ift::encoder

#endif  // IFT_ENCODER_INVALIDATION_SET_H_