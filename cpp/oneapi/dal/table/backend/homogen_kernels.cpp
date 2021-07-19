/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/table/backend/homogen_kernels.hpp"
#include "oneapi/dal/table/backend/convert.hpp"
#include "oneapi/dal/backend/common.hpp"

namespace oneapi::dal::backend {

struct block_info {
    block_info(std::int64_t row_offset,
               std::int64_t row_count,
               std::int64_t column_offset,
               std::int64_t column_count)
            : row_offset_(row_offset),
              row_count_(row_count),
              column_offset_(column_offset),
              column_count_(column_count) {
        ONEDAL_ASSERT(row_offset >= 0);
        ONEDAL_ASSERT(row_count > 0);
        ONEDAL_ASSERT(column_offset >= 0);
        ONEDAL_ASSERT(column_count > 0);
        ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, row_count, column_count);
    }

    std::int64_t get_row_offset() const {
        return row_offset_;
    }

    std::int64_t get_row_count() const {
        return row_count_;
    }

    std::int64_t get_column_offset() const {
        return column_offset_;
    }

    std::int64_t get_column_count() const {
        return column_count_;
    }

    std::int64_t get_element_count() const {
        return row_count_ * column_count_;
    }

private:
    std::int64_t row_offset_;
    std::int64_t row_count_;
    std::int64_t column_offset_;
    std::int64_t column_count_;
};

ONEDAL_FORCEINLINE void check_block_row_range(const range& rows, std::int64_t origin_row_count) {
    const std::int64_t range_row_count = rows.get_element_count(origin_row_count);
    detail::check_sum_overflow(rows.start_idx, range_row_count);
    if (rows.start_idx + range_row_count > origin_row_count) {
        throw range_error{ detail::error_messages::invalid_range_of_rows() };
    }
}

ONEDAL_FORCEINLINE void check_block_column_index(std::int64_t column_index,
                                                 std::int64_t origin_col_count) {
    if (column_index >= origin_col_count) {
        throw range_error{ detail::error_messages::column_index_out_of_range() };
    }
}

ONEDAL_FORCEINLINE void check_origin_data(const homogen_info& origin_info,
                                          const array<byte_t>& origin_data,
                                          std::int64_t origin_dtype_size,
                                          std::int64_t block_dtype_size) {
    detail::check_mul_overflow(origin_info.get_element_count(),
                               std::max(origin_dtype_size, block_dtype_size));
    ONEDAL_ASSERT(origin_data.get_count() >= origin_info.get_element_count() * origin_dtype_size);
}

template <typename Policy, typename BlockData>
static void pull_row_major_impl(const Policy& policy,
                                const homogen_info& origin_info,
                                const block_info& block_info,
                                const array<byte_t>& origin_data,
                                array<BlockData>& block_data,
                                alloc_kind requested_alloc_kind,
                                bool preserve_mutability) {
    constexpr std::int64_t block_dtype_size = sizeof(BlockData);
    const auto origin_dtype_size = origin_info.get_data_type_size();
    const auto block_dtype = detail::make_data_type<BlockData>();

    // Overflows are checked here
    check_origin_data(origin_info, origin_data, origin_dtype_size, block_dtype_size);

    // Arithmetic operations are safe because block offsets do not exceed origin element count
    const std::int64_t origin_offset =
        block_info.get_row_offset() * origin_info.get_column_count() +
        block_info.get_column_offset();

    const bool contiguous_block_requested =
        (block_info.get_column_count() == origin_info.get_column_count()) ||
        (block_info.get_row_count() == 1);
    const bool nocopy_alloc_kind =
        !alloc_kind_requires_copy(get_alloc_kind(origin_data), requested_alloc_kind);
    const bool same_data_type = (block_dtype == origin_info.get_data_type());
    const bool block_has_enough_space = (block_data.get_count() >= block_info.get_element_count());
    const bool block_has_mutable_data = block_data.has_mutable_data();

    if (contiguous_block_requested && same_data_type && nocopy_alloc_kind) {
        refer_origin_data(origin_data,
                          origin_offset * block_dtype_size,
                          block_info.get_element_count(),
                          block_data,
                          preserve_mutability);
    }
    else {
        if (!block_has_enough_space || !block_has_mutable_data || !nocopy_alloc_kind) {
            reset_array(policy, block_data, block_info.get_element_count(), requested_alloc_kind);
        }

        auto src_data = origin_data.get_data() + origin_offset * origin_dtype_size;
        auto dst_data = block_data.get_mutable_data();

        if (block_info.get_column_count() > 1) {
            const std::int64_t subblocks_count =
                contiguous_block_requested ? 1 : block_info.get_row_count();
            const std::int64_t subblock_size = contiguous_block_requested
                                                   ? block_info.get_element_count()
                                                   : block_info.get_column_count();

            for (std::int64_t i = 0; i < subblocks_count; i++) {
                backend::convert_vector(
                    policy,
                    src_data + i * origin_info.get_column_count() * origin_dtype_size,
                    dst_data + i * block_info.get_column_count(),
                    origin_info.get_data_type(),
                    block_dtype,
                    subblock_size);
            }
        }
        else {
            backend::convert_vector(policy,
                                    src_data,
                                    dst_data,
                                    origin_info.get_data_type(),
                                    block_dtype,
                                    origin_info.get_column_count(),
                                    1,
                                    block_info.get_element_count());
        }
    }
}

template <typename Policy, typename BlockData>
static void pull_column_major_impl(const Policy& policy,
                                   const homogen_info& origin_info,
                                   const block_info& block_info,
                                   const array<byte_t>& origin_data,
                                   array<BlockData>& block_data,
                                   alloc_kind requested_alloc_kind,
                                   bool preserve_mutability) {
    constexpr std::int64_t block_dtype_size = sizeof(BlockData);
    const auto origin_dtype_size = origin_info.get_data_type_size();
    const auto block_dtype = detail::make_data_type<BlockData>();

    // overflows checked here
    check_origin_data(origin_info, origin_data, origin_dtype_size, block_dtype_size);

    // operation is safe because block offsets do not exceed origin element count
    const std::int64_t origin_offset =
        block_info.get_row_offset() + block_info.get_column_offset() * origin_info.get_row_count();

    const bool nocopy_alloc_kind =
        !alloc_kind_requires_copy(get_alloc_kind(origin_data), requested_alloc_kind);
    const bool block_has_enough_space = (block_data.get_count() >= block_info.get_element_count());
    const bool block_has_mutable_data = block_data.has_mutable_data();

    if (!block_has_enough_space || !block_has_mutable_data || !nocopy_alloc_kind) {
        reset_array(policy, block_data, block_info.get_element_count(), requested_alloc_kind);
    }

    auto src_data = origin_data.get_data() + origin_offset * origin_dtype_size;
    auto dst_data = block_data.get_mutable_data();

    for (std::int64_t i = 0; i < block_info.get_row_count(); i++) {
        backend::convert_vector(policy,
                                src_data + i * origin_dtype_size,
                                dst_data + i * block_info.get_column_count(),
                                origin_info.get_data_type(),
                                block_dtype,
                                origin_info.get_row_count(),
                                1,
                                block_info.get_column_count());
    }
}

template <typename Policy, typename BlockData>
static void push_row_major_impl(const Policy& policy,
                                const homogen_info& origin_info,
                                const block_info& block_info,
                                array<byte_t>& origin_data,
                                const array<BlockData>& block_data) {
    constexpr std::int64_t block_dtype_size = sizeof(BlockData);
    const auto origin_dtype_size = origin_info.get_data_type_size();
    const auto block_dtype = detail::make_data_type<BlockData>();

    // overflows checked here
    check_origin_data(origin_info, origin_data, origin_dtype_size, block_dtype_size);
    if (block_data.get_count() != block_info.get_element_count()) {
        throw range_error{ detail::error_messages::small_data_block() };
    }

    origin_data.need_mutable_data();

    // operation is safe because block offsets do not exceed origin element count
    const std::int64_t origin_offset =
        block_info.get_row_offset() * origin_info.get_column_count() +
        block_info.get_column_offset();

    const bool contiguous_block_requested =
        block_info.get_column_count() == origin_info.get_column_count() ||
        block_info.get_row_count() == 1;

    if (origin_info.get_data_type() == block_dtype && contiguous_block_requested == true) {
        auto row_data = reinterpret_cast<BlockData*>(origin_data.get_mutable_data());
        auto row_start_pointer = row_data + origin_offset;

        if (row_start_pointer == block_data.get_data()) {
            return;
        }
        else {
            detail::memcpy(policy,
                           row_start_pointer,
                           block_data.get_data(),
                           block_info.get_element_count() * block_dtype_size);
        }
    }
    else {
        auto src_data = block_data.get_data();
        auto dst_data = origin_data.get_mutable_data() + origin_offset * origin_dtype_size;

        if (block_info.get_column_count() > 1) {
            const std::int64_t blocks_count =
                contiguous_block_requested ? 1 : block_info.get_row_count();
            const std::int64_t block_size = contiguous_block_requested
                                                ? block_info.get_element_count()
                                                : block_info.get_column_count();

            for (std::int64_t block_idx = 0; block_idx < blocks_count; block_idx++) {
                backend::convert_vector(
                    policy,
                    src_data + block_idx * block_info.get_column_count(),
                    dst_data + block_idx * origin_info.get_column_count() * origin_dtype_size,
                    block_dtype,
                    origin_info.get_data_type(),
                    block_size);
            }
        }
        else {
            backend::convert_vector(policy,
                                    src_data,
                                    dst_data,
                                    block_dtype,
                                    origin_info.get_data_type(),
                                    1,
                                    origin_info.get_column_count(),
                                    block_info.get_element_count());
        }
    }
}

template <typename Policy, typename BlockData>
static void push_column_major_impl(const Policy& policy,
                                   const homogen_info& origin_info,
                                   const block_info& block_info,
                                   array<byte_t>& origin_data,
                                   const array<BlockData>& block_data) {
    constexpr std::int64_t block_dtype_size = sizeof(BlockData);
    const auto origin_dtype_size = origin_info.get_data_type_size();
    const auto block_dtype = detail::make_data_type<BlockData>();

    // overflows checked here
    check_origin_data(origin_info, origin_data, origin_dtype_size, block_dtype_size);

    if (block_data.get_count() != block_info.get_element_count()) {
        throw range_error{ detail::error_messages::small_data_block() };
    }

    origin_data.need_mutable_data();

    // operation is safe because block offsets do not exceed origin element count
    const std::int64_t origin_offset =
        block_info.get_row_offset() + block_info.get_column_offset() * origin_info.get_row_count();

    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, origin_info.get_element_count(), origin_dtype_size);
    auto src_data = block_data.get_data();
    auto dst_data = origin_data.get_mutable_data() + origin_offset * origin_dtype_size;

    for (std::int64_t row_idx = 0; row_idx < block_info.get_row_count(); row_idx++) {
        backend::convert_vector(policy,
                                src_data + row_idx * block_info.get_column_count(),
                                dst_data + row_idx * origin_dtype_size,
                                block_dtype,
                                origin_info.get_data_type(),
                                1,
                                origin_info.get_row_count(),
                                block_info.get_column_count());
    }
}

/// The function tries to select correct policy for pull/push implementation
/// depending on the queues stored in the table or requested data block
template <typename Policy, typename Data, typename Body>
ONEDAL_FORCEINLINE void override_policy(const Policy& policy,
                                        const array<byte_t>& origin_data,
                                        const array<Data>& block_data,
                                        Body&& body) {
#ifdef ONEDAL_DATA_PARALLEL
    const auto origin_queue_opt = origin_data.get_queue();
    const auto block_queue_opt = block_data.get_queue();

    if constexpr (detail::is_data_parallel_policy_v<Policy>) {
        is_same_context_ignore_nullopt(policy.get_queue(), block_queue_opt, origin_queue_opt);
        body(policy);
    }
    else if (block_queue_opt.has_value()) {
        is_same_context_ignore_nullopt(*block_queue_opt, origin_queue_opt);
        body(detail::data_parallel_policy{ *block_queue_opt });
    }
    else if (origin_queue_opt.has_value()) {
        body(detail::data_parallel_policy{ *origin_queue_opt });
    }
    else {
        body(detail::default_host_policy{});
    }
#else
    body(policy);
#endif
}

template <typename... Args>
ONEDAL_FORCEINLINE void pull_rows_impl(data_layout layout, Args&&... args) {
    switch (layout) {
        case data_layout::row_major: pull_row_major_impl(std::forward<Args>(args)...); break;
        case data_layout::column_major: pull_column_major_impl(std::forward<Args>(args)...); break;
        default: throw domain_error{ detail::error_messages::unsupported_data_layout() };
    }
}

template <typename... Args>
ONEDAL_FORCEINLINE void pull_column_impl(data_layout layout, Args&&... args) {
    switch (layout) {
        case data_layout::row_major: pull_column_major_impl(std::forward<Args>(args)...); break;
        case data_layout::column_major: pull_row_major_impl(std::forward<Args>(args)...); break;
        default: throw domain_error{ detail::error_messages::unsupported_data_layout() };
    }
}

template <typename... Args>
ONEDAL_FORCEINLINE void push_rows_impl(data_layout layout, Args&&... args) {
    switch (layout) {
        case data_layout::row_major: push_row_major_impl(std::forward<Args>(args)...); break;
        case data_layout::column_major: push_column_major_impl(std::forward<Args>(args)...); break;
        default: throw domain_error{ detail::error_messages::unsupported_data_layout() };
    }
}

template <typename... Args>
ONEDAL_FORCEINLINE void push_column_impl(data_layout layout, Args&&... args) {
    switch (layout) {
        case data_layout::row_major: push_column_major_impl(std::forward<Args>(args)...); break;
        case data_layout::column_major: push_row_major_impl(std::forward<Args>(args)...); break;
        default: throw domain_error{ detail::error_messages::unsupported_data_layout() };
    }
}

template <typename Policy, typename BlockData>
void homogen_pull_rows(const Policy& policy,
                       const homogen_info& origin_info,
                       const array<byte_t>& origin_data,
                       array<BlockData>& block_data,
                       const range& rows_range,
                       alloc_kind requested_alloc_kind,
                       bool preserve_mutability) {
    check_block_row_range(rows_range, origin_info.get_row_count());

    const block_info b_info{ rows_range.start_idx,
                             rows_range.get_element_count(origin_info.get_row_count()),
                             0,
                             origin_info.get_column_count() };

    override_policy(policy, origin_data, block_data, [&](auto overriden_policy) {
        pull_rows_impl(origin_info.get_layout(),
                       overriden_policy,
                       origin_info,
                       b_info,
                       origin_data,
                       block_data,
                       requested_alloc_kind,
                       preserve_mutability);
    });
}

template <typename Policy, typename BlockData>
void homogen_pull_column(const Policy& policy,
                         const homogen_info& origin_info,
                         const array<byte_t>& origin_data,
                         array<BlockData>& block_data,
                         std::int64_t column_index,
                         const range& rows_range,
                         alloc_kind requested_alloc_kind,
                         bool preserve_mutability) {
    check_block_row_range(rows_range, origin_info.get_row_count());
    check_block_column_index(column_index, origin_info.get_column_count());

    const homogen_info o_info_transposed{ origin_info.get_column_count(),
                                          origin_info.get_row_count(),
                                          origin_info.get_data_type(),
                                          origin_info.get_layout() };

    const block_info b_info{ column_index,
                             1,
                             rows_range.start_idx,
                             rows_range.get_element_count(origin_info.get_row_count()) };

    override_policy(policy, origin_data, block_data, [&](auto overriden_policy) {
        pull_column_impl(o_info_transposed.get_layout(),
                         overriden_policy,
                         o_info_transposed,
                         b_info,
                         origin_data,
                         block_data,
                         requested_alloc_kind,
                         preserve_mutability);
    });
}

template <typename Policy, typename BlockData>
void homogen_push_rows(const Policy& policy,
                       const homogen_info& origin_info,
                       array<byte_t>& origin_data,
                       const array<BlockData>& block_data,
                       const range& rows_range) {
    check_block_row_range(rows_range, origin_info.get_row_count());

    const block_info b_info{ rows_range.start_idx,
                             rows_range.get_element_count(origin_info.get_row_count()),
                             0,
                             origin_info.get_column_count() };

    override_policy(policy, origin_data, block_data, [&](auto overriden_policy) {
        push_rows_impl(origin_info.get_layout(),
                       overriden_policy,
                       origin_info,
                       b_info,
                       origin_data,
                       block_data);
    });
}

template <typename Policy, typename BlockData>
void homogen_push_column(const Policy& policy,
                         const homogen_info& origin_info,
                         array<byte_t>& origin_data,
                         const array<BlockData>& block_data,
                         std::int64_t column_index,
                         const range& rows_range) {
    check_block_row_range(rows_range, origin_info.get_row_count());
    check_block_column_index(column_index, origin_info.get_column_count());

    const homogen_info o_info_transposed{ origin_info.get_column_count(),
                                          origin_info.get_row_count(),
                                          origin_info.get_data_type(),
                                          origin_info.get_layout() };

    const block_info b_info{ column_index,
                             1,
                             rows_range.start_idx,
                             rows_range.get_element_count(origin_info.get_row_count()) };

    override_policy(policy, origin_data, block_data, [&](auto overriden_policy) {
        push_column_impl(o_info_transposed.get_layout(),
                         overriden_policy,
                         o_info_transposed,
                         b_info,
                         origin_data,
                         block_data);
    });
}

#define INSTANTIATE(Policy, BlockData)                                    \
    template void homogen_pull_rows(const Policy& policy,                 \
                                    const homogen_info& origin_info,      \
                                    const array<byte_t>& origin_data,     \
                                    array<BlockData>& block_data,         \
                                    const range& rows_range,              \
                                    alloc_kind requested_alloc_kind,      \
                                    bool preserve_mutability);            \
    template void homogen_pull_column(const Policy& policy,               \
                                      const homogen_info& origin_info,    \
                                      const array<byte_t>& origin_data,   \
                                      array<BlockData>& block_data,       \
                                      std::int64_t column_index,          \
                                      const range& rows_range,            \
                                      alloc_kind requested_alloc_kind,    \
                                      bool preserve_mutability);          \
    template void homogen_push_rows(const Policy& policy,                 \
                                    const homogen_info& origin_info,      \
                                    array<byte_t>& origin_data,           \
                                    const array<BlockData>& block_data,   \
                                    const range& rows_range);             \
    template void homogen_push_column(const Policy& policy,               \
                                      const homogen_info& origin_info,    \
                                      array<byte_t>& origin_data,         \
                                      const array<BlockData>& block_data, \
                                      std::int64_t column_index,          \
                                      const range& rows_range);

#ifdef ONEDAL_DATA_PARALLEL
#define INSTANTIATE_ALL_POLICIES(Data)             \
    INSTANTIATE(detail::default_host_policy, Data) \
    INSTANTIATE(detail::data_parallel_policy, Data)
#else
#define INSTANTIATE_ALL_POLICIES(Data) INSTANTIATE(detail::default_host_policy, Data)
#endif

INSTANTIATE_ALL_POLICIES(float)
INSTANTIATE_ALL_POLICIES(double)
INSTANTIATE_ALL_POLICIES(std::int32_t)

} // namespace oneapi::dal::backend
