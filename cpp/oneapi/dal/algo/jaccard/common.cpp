/*******************************************************************************
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "oneapi/dal/algo/jaccard/common.hpp"

namespace oneapi::dal::preview {
namespace jaccard {

class detail::descriptor_impl : public base {
public:
    std::int64_t row_begin    = 0;
    std::int64_t row_end      = 0;
    std::int64_t column_begin = 0;
    std::int64_t column_end   = 0;
};

using detail::descriptor_impl;

descriptor_base::descriptor_base() : impl_(new descriptor_impl{}) {}

std::int64_t descriptor_base::get_row_begin() const {
    return impl_->row_begin;
}

std::int64_t descriptor_base::get_row_end() const {
    return impl_->row_end;
}

std::int64_t descriptor_base::get_column_begin() const {
    return impl_->column_begin;
}

std::int64_t descriptor_base::get_column_end() const {
    return impl_->column_end;
}

void descriptor_base::set_row_range_impl(const int64_t& begin, const int64_t& end) {
    impl_->row_begin = begin;
    impl_->row_end   = end;
}

void descriptor_base::set_column_range_impl(const int64_t& begin, const int64_t& end) {
    impl_->column_begin = begin;
    impl_->column_end   = end;
}

void descriptor_base::set_block_impl(const std::initializer_list<int64_t>& row_range,
                                     const std::initializer_list<int64_t>& column_range) {
    impl_->row_begin    = *row_range.begin();
    impl_->row_end      = *(row_range.begin() + 1);
    impl_->column_begin = *column_range.begin();
    impl_->column_end   = *(column_range.begin() + 1);
}

} // namespace jaccard
} // namespace oneapi::dal::preview
