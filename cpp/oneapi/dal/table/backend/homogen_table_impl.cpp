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

#include "oneapi/dal/table/backend/homogen_table_impl.hpp"
#include "oneapi/dal/table/backend/convert.hpp"

#include <cstring>

namespace oneapi::dal::backend {

using std::int32_t;

table_metadata create_homogen_metadata(int64_t feature_count, data_type dtype) {
    auto default_ftype =
        detail::is_floating_point(dtype) ? feature_type::ratio : feature_type::ordinal;

    auto dtypes = array<data_type>::full(feature_count, dtype);
    auto ftypes = array<feature_type>::full(feature_count, default_ftype);
    return table_metadata{ dtypes, ftypes };
}

template <typename Policy, typename Data>
void make_mutable_data(const Policy& policy, array<Data>& array) {
    if constexpr (std::is_same_v<Policy, detail::default_host_policy>) {
        array.need_mutable_data();
    }
#ifdef ONEAPI_DAL_DATA_PARALLEL
    else if constexpr (std::is_same_v<Policy, detail::data_parallel_policy>) {
        auto queue = policy.get_queue();
        auto kind = sycl::get_pointer_type(array.get_data(), queue.get_context());
        array.need_mutable_data(queue, kind);
    }
#endif
    else {
        static_assert("make_mutable_data(): undefined policy type");
    }
}

template <typename Policy, typename Data, typename Alloc>
void reset_array(const Policy& policy, array<Data>& array, int64_t count, const Alloc& kind) {
    if constexpr (std::is_same_v<Policy, detail::default_host_policy>) {
        array.reset(count);
    }
#ifdef ONEAPI_DAL_DATA_PARALLEL
    else if constexpr (std::is_same_v<Policy, detail::data_parallel_policy>) {
        array.reset(policy.get_queue(), count, kind);
    }
#endif
    else {
        static_assert("reset_array(): undefined policy type");
    }
}

template <typename Policy, typename Data, typename Alloc>
bool has_array_data_kind(const Policy& policy, const array<Data>& array, const Alloc& kind) {
    if (array.get_count() <= 0) {
        return false;
    }

    if constexpr (std::is_same_v<Policy, detail::default_host_policy>) {
        // We assume that no sycl::usm::alloc::device pointers used with host policies.
        // It is responsibility of user to pass right pointers because we cannot check
        // the right pointer type with the host policy.
        return true;
    }
#ifdef ONEAPI_DAL_DATA_PARALLEL
    else if constexpr (std::is_same_v<Policy, detail::data_parallel_policy>) {
        static_assert(std::is_same_v<Alloc, sycl::usm::alloc>);
        auto array_data_kind =
            sycl::get_pointer_type(array.get_data(), policy.get_queue().get_context());
        return array_data_kind == kind;
    }
#endif
    else {
        static_assert("has_array_data_kind(): undefined policy type");
    }

    return false;
}

template <typename DataSrc, typename DataDest>
void refer_source_data(const array<DataSrc>& src,
                       std::int64_t src_start_index,
                       std::int64_t dst_count,
                       array<DataDest>& dst) {
    if (src.has_mutable_data()) {
        auto start_pointer = reinterpret_cast<DataDest*>(src.get_mutable_data() + src_start_index);
        dst.reset(src, start_pointer, dst_count);
    }
    else {
        auto start_pointer = reinterpret_cast<const DataDest*>(src.get_data() + src_start_index);
        dst.reset(src, start_pointer, dst_count);
    }
}

class block_access_provider {
public:
    block_access_provider(int64_t origin_row_count,
                          int64_t origin_column_count,
                          data_type origin_data_type,
                          const range& block_row_range,
                          const range& block_column_range)
        : block_row_count_(block_row_range.get_element_count(origin_row_count)),
          block_column_count_(block_column_range.get_element_count(origin_column_count)),
          block_start_row_idx_(block_row_range.start_idx),
          block_start_column_idx_(block_column_range.start_idx),
          block_size_(block_row_count_ * block_column_count_),
          origin_dtype_(origin_data_type),
          origin_row_count_(origin_row_count),
          origin_column_count_(origin_column_count)
        {}

    template <typename Policy, typename BlockData, typename Alloc>
    void pull_by_row_major(const Policy& policy,
                           const array<byte_t>& origin,
                           array<BlockData>& block,
                           const Alloc& kind) const {
        const auto block_dtype = detail::make_data_type<BlockData>();
        const int64_t origin_offset =
            (block_start_row_idx_ * origin_column_count_ + block_start_column_idx_);
        const bool contiguous_block_requested =
            block_column_count_ == origin_column_count_ || block_row_count_ == 1;

        if (block_dtype == origin_dtype_ && contiguous_block_requested == true &&
            has_array_data_kind(policy, origin, kind)) {
            refer_source_data(origin, origin_offset * sizeof(BlockData), block_size_, block);
        }
        else {
            const auto origin_dtype_size = detail::get_data_type_size(origin_dtype_);

            if (block.get_count() < block_size_ || block.has_mutable_data() == false ||
                has_array_data_kind(policy, block, kind) == false) {
                reset_array(policy, block, block_size_, kind);
            }

            auto src_data = origin.get_data() + origin_offset * origin_dtype_size;
            auto dst_data = block.get_mutable_data();

            if (block_column_count_ > 1) {
                const int64_t subblocks_count = contiguous_block_requested ? 1 : block_row_count_;
                const int64_t subblock_size = contiguous_block_requested ? block_size_ : block_column_count_;

                for (int64_t subblock_idx = 0; subblock_idx < subblocks_count; subblock_idx++) {
                    backend::convert_vector(policy,
                                            src_data + subblock_idx * origin_column_count_ * origin_dtype_size,
                                            dst_data + subblock_idx * block_column_count_,
                                            origin_dtype_,
                                            block_dtype,
                                            subblock_size);
                }
            }
            else {
                backend::convert_vector(policy,
                                        src_data,
                                        dst_data,
                                        origin_dtype_,
                                        block_dtype,
                                        origin_dtype_size * origin_column_count_,
                                        sizeof(BlockData),
                                        block_size_);
            }
        }
    }

    template <typename Policy, typename BlockData, typename Alloc>
    void pull_by_column_major(const Policy& policy,
                              const array<byte_t>& origin,
                              array<BlockData>& block,
                              const Alloc& kind) const {
        const auto block_dtype = detail::make_data_type<BlockData>();
        const int64_t origin_offset =
            block_start_row_idx_ + block_start_column_idx_ * origin_row_count_;
        const auto type_size = detail::get_data_type_size(origin_dtype_);

        if (block.get_count() < block_size_ || block.has_mutable_data() == false ||
            has_array_data_kind(policy, block, kind) == false) {
            reset_array(policy, block, block_size_, kind);
        }

        auto src_data = origin.get_data() + origin_offset * type_size;
        auto dst_data = block.get_mutable_data();

        for(int64_t row_idx = 0; row_idx < block_row_count_; row_idx++) {
            backend::convert_vector(policy,
                                    src_data + row_idx * type_size,
                                    dst_data + row_idx * block_column_count_,
                                    origin_dtype_,
                                    block_dtype,
                                    type_size * origin_row_count_,
                                    sizeof(BlockData),
                                    block_column_count_);
        }
    }

    template <typename Policy, typename BlockData>
    void push_by_row_major(const Policy& policy,
                           array<byte_t>& origin,
                           const array<BlockData>& block) const {
        make_mutable_data(policy, origin);
        const auto dst_dtype = detail::make_data_type<BlockData>();
        const int64_t origin_offset =
            block_start_row_idx_ * origin_column_count_ + block_start_column_idx_;
        const bool contiguous_block_requested =
            block_column_count_ == origin_column_count_ || block_row_count_ == 1;

        if (origin_dtype_ == dst_dtype && contiguous_block_requested == true) {
            auto row_data = reinterpret_cast<BlockData*>(origin.get_mutable_data());
            auto row_start_pointer = row_data + origin_offset;

            if (row_start_pointer == block.get_data()) {
                return;
            }
            else {
                detail::memcpy(policy, row_start_pointer, block.get_data(), block_size_ * sizeof(BlockData));
            }
        }
        else {
            const auto origin_dtype_size = detail::get_data_type_size(origin_dtype_);

            auto src_data = block.get_data();
            auto dst_data =
                origin.get_mutable_data() + origin_offset * origin_dtype_size;

            if (block_column_count_ > 1) {
                const int64_t blocks_count = contiguous_block_requested ? 1 : block_row_count_;
                const int64_t block_size = contiguous_block_requested ? block_size_ : block_column_count_;

                for (int64_t block_idx = 0; block_idx < blocks_count; block_idx++) {
                    backend::convert_vector(policy,
                                            src_data + block_idx * block_column_count_,
                                            dst_data + block_idx * origin_column_count_ * origin_dtype_size,
                                            dst_dtype,
                                            origin_dtype_,
                                            block_size);
                }
            }
            else {
                backend::convert_vector(policy,
                                        src_data,
                                        dst_data,
                                        dst_dtype,
                                        origin_dtype_,
                                        sizeof(BlockData),
                                        origin_dtype_size * origin_column_count_,
                                        block_size_);
            }
        }
    }

    template <typename Policy, typename BlockData>
    void push_by_column_major(const Policy& policy,
                              array<byte_t>& origin,
                              const array<BlockData>& block) const {
        make_mutable_data(policy, origin);
        const auto dst_dtype = detail::make_data_type<BlockData>();
        const auto origin_dtype_size = detail::get_data_type_size(origin_dtype_);
        const int64_t origin_offset =
            block_start_row_idx_ + block_start_column_idx_ * origin_row_count_;

        auto src_data = block.get_data();
        auto dst_data = origin.get_mutable_data() + origin_offset * origin_dtype_size;

        for(int64_t row_idx = 0; row_idx < block_row_count_; row_idx++) {
            backend::convert_vector(policy,
                                    src_data + row_idx * block_column_count_,
                                    dst_data + row_idx * origin_dtype_size,
                                    dst_dtype,
                                    origin_dtype_,
                                    sizeof(BlockData),
                                    origin_dtype_size * origin_row_count_,
                                    block_column_count_);
        }
    }

private:
    const int64_t block_row_count_;
    const int64_t block_column_count_;
    const int64_t block_start_row_idx_;
    const int64_t block_start_column_idx_;
    const int64_t block_size_;
    const data_type origin_dtype_;
    const int64_t origin_row_count_;
    const int64_t origin_column_count_;
};

template <typename Policy, typename Data, typename Alloc>
void homogen_table_impl::pull_rows_impl(const Policy& policy,
                                        array<Data>& block,
                                        const range& rows,
                                        const Alloc& kind) const {
    // TODO: check range correctness
    // TODO: check array size if non-zero
    const block_access_provider provider { row_count_, col_count_, meta_.get_data_type(0), rows, {0, -1} };

    switch(layout_) {
        case data_layout::row_major:
            provider.pull_by_row_major(policy, data_, block, kind);
            break;
        case data_layout::column_major:
            provider.pull_by_column_major(policy, data_, block, kind);
            break;
        default:
            throw internal_error("homogen_table_impl::pull_rows_impl: unsupported layout");
    }
}

template <typename Policy, typename Data, typename Alloc>
void homogen_table_impl::pull_column_impl(const Policy& policy,
                                          array<Data>& block,
                                          std::int64_t column_index,
                                          const range& rows,
                                          const Alloc& kind) const {
    // TODO: check range correctness
    // TODO: check array size if non-zero
    const block_access_provider provider { col_count_, row_count_, meta_.get_data_type(0), {column_index, column_index + 1}, rows };

    switch(layout_) {
        case data_layout::row_major:
            provider.pull_by_column_major(policy, data_, block, kind);
            break;
        case data_layout::column_major:
            provider.pull_by_row_major(policy, data_, block, kind);
            break;
        default:
            throw internal_error("homogen_table_impl::pull_column_impl: unsupported layout");
    }
}

template <typename Policy, typename Data>
void homogen_table_impl::push_rows_impl(const Policy& policy,
                                        const array<Data>& block,
                                        const range& rows) {
    // TODO: check range correctness
    // TODO: check array size if non-zero
    const block_access_provider provider { row_count_, col_count_, meta_.get_data_type(0), rows, {0, -1} };

    switch(layout_) {
        case data_layout::row_major:
            provider.push_by_row_major(policy, data_, block);
            break;
        case data_layout::column_major:
            provider.push_by_column_major(policy, data_, block);
            break;
        default:
            throw internal_error("homogen_table_impl::push_rows_impl: unsupported layout");
    }
}

template <typename Policy, typename Data>
void homogen_table_impl::push_column_impl(const Policy& policy,
                        const array<Data>& block,
                        std::int64_t column_index,
                        const range& rows) {
    // TODO: check range correctness
    // TODO: check array size if non-zero
    const block_access_provider provider { col_count_, row_count_, meta_.get_data_type(0), {column_index, column_index + 1}, rows };

    switch(layout_) {
        case data_layout::row_major:
            provider.push_by_column_major(policy, data_, block);
            break;
        case data_layout::column_major:
            provider.push_by_row_major(policy, data_, block);
            break;
        default:
            throw internal_error("homogen_table_impl::push_column_impl: unsupported layout");
    }
}

#define INSTANTIATE_IMPL(Policy, Data, Alloc)                                     \
    template void homogen_table_impl::pull_rows_impl(const Policy& policy,        \
                                                     array<Data>& block,          \
                                                     const range& rows,           \
                                                     const Alloc& kind) const;    \
    template void homogen_table_impl::pull_column_impl(const Policy& policy,      \
                                                       array<Data>& block,        \
                                                       int64_t column_index,      \
                                                       const range& rows,         \
                                                       const Alloc& kind) const;  \
    template void homogen_table_impl::push_rows_impl(const Policy& policy,        \
                                                     const array<Data>& block,    \
                                                     const range& rows);          \
    template void homogen_table_impl::push_column_impl(const Policy& policy,      \
                                                       const array<Data>& block,  \
                                                       int64_t column_index,      \
                                                       const range& rows);

#ifdef ONEAPI_DAL_DATA_PARALLEL
#define INSTANTIATE_IMPL_ALL_POLICIES(Data)                                               \
    INSTANTIATE_IMPL(detail::default_host_policy, Data, homogen_table_impl::host_alloc_t) \
    INSTANTIATE_IMPL(detail::data_parallel_policy, Data, sycl::usm::alloc)
#else
#define INSTANTIATE_IMPL_ALL_POLICIES(Data) \
    INSTANTIATE_IMPL(detail::default_host_policy, Data, homogen_table_impl::host_alloc_t)
#endif

INSTANTIATE_IMPL_ALL_POLICIES(float)
INSTANTIATE_IMPL_ALL_POLICIES(double)
INSTANTIATE_IMPL_ALL_POLICIES(int32_t)

#undef INSTANTIATE_IMPL_ALL_POLICIES
#undef INSTANTIATE_IMPL

} // namespace oneapi::dal::backend
