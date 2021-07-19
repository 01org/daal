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

#pragma once

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend {

enum class alloc_kind {
    host, /// Non-USM pointer allocated on host
    usm_host, /// USM pointer allocated by sycl::alloc_host
    usm_device, /// USM pointer allocated by sycl::alloc_device
    usm_shared /// USM pointer allocated by sycl::alloc_shared
};

#ifdef ONEDAL_DATA_PARALLEL
inline alloc_kind alloc_kind_from_sycl(sycl::usm::alloc alloc) {
    using error_msg = dal::detail::error_messages;
    switch (alloc) {
        case sycl::usm::alloc::host: return alloc_kind::usm_host;
        case sycl::usm::alloc::device: return alloc_kind::usm_device;
        case sycl::usm::alloc::shared: return alloc_kind::usm_shared;
        default: throw invalid_argument{ error_msg::unsupported_usm_alloc() };
    }
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
inline sycl::usm::alloc alloc_kind_to_sycl(alloc_kind kind) {
    using error_msg = dal::detail::error_messages;
    switch (kind) {
        case alloc_kind::usm_host: return sycl::usm::alloc::host;
        case alloc_kind::usm_device: return sycl::usm::alloc::device;
        case alloc_kind::usm_shared: return sycl::usm::alloc::shared;
        default: throw invalid_argument{ error_msg::unsupported_usm_alloc() };
    }
}
#endif

inline bool alloc_kind_requires_copy(alloc_kind src_alloc_kind, alloc_kind dst_alloc_kind) {
#ifdef ONEDAL_DATA_PARALLEL
    switch (dst_alloc_kind) {
        case alloc_kind::host: //
            return (src_alloc_kind == alloc_kind::usm_device);
        case alloc_kind::usm_host: //
            return (src_alloc_kind == alloc_kind::host) || //
                   (src_alloc_kind == alloc_kind::usm_device);
        case alloc_kind::usm_device: //
            return (src_alloc_kind == alloc_kind::host) || //
                   (src_alloc_kind == alloc_kind::usm_host);
        case alloc_kind::usm_shared: //
            return (src_alloc_kind != alloc_kind::usm_shared);
        default: //
            ONEDAL_ASSERT(!"Unsupported alloc_kind");
    }
#else
    ONEDAL_ASSERT(src_alloc_kind == alloc_kind::host);
    ONEDAL_ASSERT(dst_alloc_kind == alloc_kind::host);
#endif
    return false;
}

template <typename T>
inline alloc_kind get_alloc_kind(const array<T>& array) {
#ifdef ONEDAL_DATA_PARALLEL
    const auto opt_queue = array.get_queue();
    if (opt_queue.has_value()) {
        const auto queue = opt_queue.value();
        const auto array_alloc = sycl::get_pointer_type(array.get_data(), queue.get_context());
        return alloc_kind_from_sycl(array_alloc);
    }
#endif
    return alloc_kind::host;
}

template <typename Policy, typename Data>
ONEDAL_FORCEINLINE void reset_array(const Policy& policy,
                                    array<Data>& array,
                                    std::int64_t count,
                                    const alloc_kind& kind) {
    if constexpr (std::is_same_v<Policy, detail::default_host_policy>) {
        ONEDAL_ASSERT(kind == alloc_kind::host, "Incompatible policy and type of allocation");
    }
#ifdef ONEDAL_DATA_PARALLEL
    if (kind == alloc_kind::host) {
        array.reset(count);
    }
    else {
        if constexpr (std::is_same_v<Policy, detail::data_parallel_policy>) {
            const auto alloc = alloc_kind_to_sycl(kind);
            array.reset(policy.get_queue(), count, alloc);
        }
    }
#else
    array.reset(count);
#endif
}

template <typename DataSrc, typename DataDest>
ONEDAL_FORCEINLINE void refer_origin_data(const array<DataSrc>& src,
                                          std::int64_t src_start_index,
                                          std::int64_t dst_count,
                                          array<DataDest>& dst,
                                          bool preserve_mutability) {
    ONEDAL_ASSERT(src_start_index >= 0);
    ONEDAL_ASSERT(src.get_count() > src_start_index);
    ONEDAL_ASSERT((src.get_count() - src_start_index) * sizeof(DataSrc) >=
                  dst_count * sizeof(DataDest));

    if (src.has_mutable_data() && preserve_mutability) {
        auto start_pointer = reinterpret_cast<DataDest*>(src.get_mutable_data() + src_start_index);
        dst.reset(src, start_pointer, dst_count);
    }
    else {
        auto start_pointer = reinterpret_cast<const DataDest*>(src.get_data() + src_start_index);
        dst.reset(src, start_pointer, dst_count);
    }
}

inline table_metadata create_metadata(std::int64_t feature_count, data_type dtype) {
    auto default_ftype =
        detail::is_floating_point(dtype) ? feature_type::ratio : feature_type::ordinal;

    auto dtypes = array<data_type>::full(feature_count, dtype);
    auto ftypes = array<feature_type>::full(feature_count, default_ftype);
    return table_metadata{ dtypes, ftypes };
}

} // namespace oneapi::dal::backend
