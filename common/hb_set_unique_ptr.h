#ifndef COMMON_HB_SET_UNIQUE_PTR_H_
#define COMMON_HB_SET_UNIQUE_PTR_H_

#include <memory>

#include "hb.h"

namespace common {

typedef std::unique_ptr<hb_set_t, decltype(&hb_set_destroy)> hb_set_unique_ptr;

hb_set_unique_ptr make_hb_set();

// Takes ownership of set
hb_set_unique_ptr make_hb_set(hb_set_t* set);

hb_set_unique_ptr make_hb_set(int length, ...);

hb_set_unique_ptr make_hb_set_from_ranges(int number_of_ranges, ...);

hb_set_unique_ptr make_hb_set(int length, ...);

}  // namespace common

#endif  // COMMON_HB_SET_UNIQUE_PTR_H_
