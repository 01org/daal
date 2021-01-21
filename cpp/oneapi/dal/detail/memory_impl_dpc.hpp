/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifdef ONEDAL_DATA_PARALLEL
#include <cstring>

#include "oneapi/dal/detail/common_dpc.hpp"
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::detail {
namespace v1 {

ONEDAL_EXPORT void* malloc_impl_dpc(const data_parallel_policy&,
                                    std::int64_t size,
                                    const sycl::usm::alloc& alloc);
ONEDAL_EXPORT void free_impl_dpc(const data_parallel_policy&, void* pointer);
ONEDAL_EXPORT void fill_impl_dpc(const data_parallel_policy&,
                                 void* dest,
                                 std::int64_t size,
                                 const void* pattern,
                                 std::int64_t pattern_size);

template <typename T>
T* malloc(const data_parallel_policy& policy, std::int64_t count, const sycl::usm::alloc& alloc) {
    const std::int64_t bytes_count = sizeof(T) * count;
    ONEDAL_ASSERT(bytes_count > count);
    return static_cast<T*>(malloc_impl_host(policy, bytes_count, alloc));
}

template <typename T>
void free(const data_parallel_policy& policy, T* pointer) {
    using mutable_t = std::remove_const_t<T>;
    free_impl_dpc(policy, const_cast<mutable_t*>(pointer));
}

ONEDAL_EXPORT void memset(const data_parallel_policy& policy,
                          void* dest,
                          std::int32_t value,
                          std::int64_t size);
ONEDAL_EXPORT void memcpy(const data_parallel_policy& policy,
                          void* dest,
                          const void* src,
                          std::int64_t size);

template <typename T>
void fill(const data_parallel_policy& policy, T* dest, std::int64_t count, const T& value) {
    const std::int64_t bytes_count = sizeof(T) * count;
    ONEDAL_ASSERT(bytes_count > count);
    fill_impl_host(policy, dest, bytes_count, &value, sizeof(T));
}

template <typename T>
class data_parallel_allocator {
public:
    data_parallel_allocator(const data_parallel_policy& policy,
                            sycl::usm::alloc kind = sycl::usm::alloc::shared)
            : policy_(policy),
              kind_(kind) {}

    T* allocate(std::int64_t n) const {
        return malloc<T>(policy_, n, kind_);
    }

    void deallocate(T* p, std::int64_t n) const {
        free(policy_, p);
    }

    sycl::usm::alloc get_kind() const {
        return kind_;
    }

private:
    data_parallel_policy policy_;
    sycl::usm::alloc kind_;
};

} // namespace v1

using v1::malloc;
using v1::free;
using v1::memset;
using v1::memcpy;
using v1::fill;
using v1::data_parallel_allocator;

} // namespace oneapi::dal::detail

#endif // ONEDAL_DATA_PARALLEL
