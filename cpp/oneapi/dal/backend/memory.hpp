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

#include "oneapi/dal/backend/common.hpp"

namespace oneapi::dal::backend {

void memcpy(void* dest, const void* src, std::int64_t size);

template <typename T>
inline void copy(T* dest, const T* src, std::int64_t count) {
    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, sizeof(T), count);
    memcpy(dest, src, sizeof(T) * count);
}

#ifdef ONEDAL_DATA_PARALLEL
inline bool is_device_usm(const sycl::queue& queue, const void* pointer) {
    const auto pointer_type = sycl::get_pointer_type(pointer, queue.get_context());
    return pointer_type == sycl::usm::alloc::device;
}

inline bool is_shared_usm(const sycl::queue& queue, const void* pointer) {
    const auto pointer_type = sycl::get_pointer_type(pointer, queue.get_context());
    return pointer_type == sycl::usm::alloc::shared;
}

inline bool is_host_usm(const sycl::queue& queue, const void* pointer) {
    const auto pointer_type = sycl::get_pointer_type(pointer, queue.get_context());
    return pointer_type == sycl::usm::alloc::host;
}

inline bool is_device_friendly_usm(const sycl::queue& queue, const void* pointer) {
    const auto pointer_type = sycl::get_pointer_type(pointer, queue.get_context());
    return (pointer_type == sycl::usm::alloc::device) || //
           (pointer_type == sycl::usm::alloc::shared);
}

inline bool is_known_usm(const sycl::queue& queue, const void* pointer) {
    const auto pointer_type = sycl::get_pointer_type(pointer, queue.get_context());
    return pointer_type != sycl::usm::alloc::unknown;
}

inline void* malloc(const sycl::queue& queue, std::size_t size, const sycl::usm::alloc& alloc) {
    ONEDAL_ASSERT(size > 0);
    auto ptr = sycl::malloc(size, queue, alloc);
    if (!ptr) {
        if (alloc == sycl::usm::alloc::shared || alloc == sycl::usm::alloc::host) {
            throw dal::host_bad_alloc{};
        }
        else if (alloc == sycl::usm::alloc::device) {
            throw dal::device_bad_alloc{};
        }
        else {
            throw dal::invalid_argument{ detail::error_messages::unknown_usm_pointer_type() };
        }
    }
    return ptr;
}

inline void* malloc_device(const sycl::queue& queue, std::size_t size) {
    return malloc(queue, size, sycl::usm::alloc::device);
}

inline void* malloc_shared(const sycl::queue& queue, std::size_t size) {
    return malloc(queue, size, sycl::usm::alloc::shared);
}

inline void* malloc_host(const sycl::queue& queue, std::size_t size) {
    return malloc(queue, size, sycl::usm::alloc::host);
}

inline void free(const sycl::queue& queue, void* pointer) {
    ONEDAL_ASSERT(pointer == nullptr || is_known_usm(queue, pointer));
    sycl::free(pointer, queue);
}

template <typename T>
inline T* malloc(const sycl::queue& queue, std::int64_t count, const sycl::usm::alloc& alloc) {
    ONEDAL_ASSERT(count > 0);
    ONEDAL_ASSERT_MUL_OVERFLOW(std::size_t, sizeof(T), count);
    const std::size_t bytes_count = sizeof(T) * count;
    return static_cast<T*>(malloc(queue, bytes_count, alloc));
}

template <typename T>
inline T* malloc_device(const sycl::queue& queue, std::int64_t count) {
    return malloc<T>(queue, count, sycl::usm::alloc::device);
}

template <typename T>
inline T* malloc_shared(const sycl::queue& queue, std::int64_t count) {
    return malloc<T>(queue, count, sycl::usm::alloc::shared);
}

template <typename T>
inline T* malloc_host(const sycl::queue& queue, std::int64_t count) {
    return malloc<T>(queue, count, sycl::usm::alloc::host);
}

template <typename T>
class usm_deleter {
public:
    explicit usm_deleter(const sycl::queue& queue) : queue_(queue) {}

    void operator()(T* ptr) const {
        free(queue_, ptr);
    }

    sycl::queue& get_queue() {
        return queue_;
    }

    const sycl::queue& get_queue() const {
        return queue_;
    }

private:
    sycl::queue queue_;
};

template <typename T>
using unique_usm_ptr = std::unique_ptr<T, usm_deleter<T>>;

inline unique_usm_ptr<void> make_unique_usm(const sycl::queue& q,
                                            std::size_t size,
                                            sycl::usm::alloc alloc) {
    return unique_usm_ptr<void>{ malloc(q, size, alloc), usm_deleter<void>{ q } };
}

inline unique_usm_ptr<void> make_unique_usm_device(const sycl::queue& q, std::size_t size) {
    return unique_usm_ptr<void>{ malloc_device(q, size), usm_deleter<void>{ q } };
}

inline unique_usm_ptr<void> make_unique_usm_shared(const sycl::queue& q, std::size_t size) {
    return unique_usm_ptr<void>{ malloc_shared(q, size), usm_deleter<void>{ q } };
}

inline unique_usm_ptr<void> make_unique_usm_host(const sycl::queue& q, std::size_t size) {
    return unique_usm_ptr<void>{ malloc_host(q, size), usm_deleter<void>{ q } };
}

template <typename T>
inline unique_usm_ptr<T> make_unique_usm(const sycl::queue& q,
                                         std::int64_t count,
                                         sycl::usm::alloc alloc) {
    return unique_usm_ptr<T>{ malloc<T>(q, count, alloc), usm_deleter<T>{ q } };
}

template <typename T>
inline unique_usm_ptr<T> make_unique_usm_device(const sycl::queue& q, std::int64_t count) {
    return unique_usm_ptr<T>{ malloc_device<T>(q, count), usm_deleter<T>{ q } };
}

template <typename T>
inline unique_usm_ptr<T> make_unique_usm_shared(const sycl::queue& q, std::int64_t count) {
    return unique_usm_ptr<T>{ malloc_shared<T>(q, count), usm_deleter<T>{ q } };
}

template <typename T>
inline unique_usm_ptr<T> make_unique_usm_host(const sycl::queue& q, std::int64_t count) {
    return unique_usm_ptr<T>{ malloc_host<T>(q, count), usm_deleter<T>{ q } };
}

inline sycl::event memcpy(sycl::queue& queue,
                          void* dest,
                          const void* src,
                          std::size_t size,
                          const event_vector& deps = {}) {
    ONEDAL_ASSERT(size > 0);
    ONEDAL_ASSERT(is_known_usm(queue, dest));
    ONEDAL_ASSERT(is_known_usm(queue, src));
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.memcpy(dest, src, size);
    });
}

inline sycl::event memcpy_host2usm(sycl::queue& queue,
                                   void* dest_usm,
                                   const void* src_host,
                                   std::size_t size,
                                   const event_vector& deps = {}) {
    ONEDAL_ASSERT(is_known_usm(queue, dest_usm));

    // TODO: Remove additional copy to host usm memory once
    //       bug in `copy` with the host memory is fixed
    auto tmp_usm_host = make_unique_usm_host(queue, size);
    memcpy(tmp_usm_host.get(), src_host, size);
    memcpy(queue, dest_usm, tmp_usm_host.get(), size, deps).wait_and_throw();
    return {};
}

inline sycl::event memcpy_usm2host(sycl::queue& queue,
                                   void* dest_host,
                                   const void* src_usm,
                                   std::size_t size,
                                   const event_vector& deps = {}) {
    ONEDAL_ASSERT(is_known_usm(queue, src_usm));

    // TODO: Remove additional copy to host usm memory once
    //       bug in `copy` with the host memory is fixed
    auto tmp_usm_host = make_unique_usm_host(queue, size);
    memcpy(queue, tmp_usm_host.get(), src_usm, size, deps).wait_and_throw();
    memcpy(dest_host, tmp_usm_host.get(), size);
    return {};
}

template <typename T>
inline sycl::event copy(sycl::queue& queue,
                        T* dest,
                        const T* src,
                        std::int64_t count,
                        const event_vector& deps = {}) {
    ONEDAL_ASSERT(count > 0);
    const std::size_t n = detail::integral_cast<std::size_t>(count);
    ONEDAL_ASSERT_MUL_OVERFLOW(std::size_t, sizeof(T), n);
    return memcpy(queue, dest, src, sizeof(T) * n, deps);
}

template <typename T>
inline sycl::event copy_host2usm(sycl::queue& queue,
                                 T* dest_usm,
                                 const T* src_host,
                                 std::int64_t count,
                                 const event_vector& deps = {}) {
    ONEDAL_ASSERT(count > 0);
    const std::size_t n = detail::integral_cast<std::size_t>(count);
    ONEDAL_ASSERT_MUL_OVERFLOW(std::size_t, sizeof(T), n);
    return memcpy_host2usm(queue, dest_usm, src_host, sizeof(T) * n, deps);
}

template <typename T>
inline sycl::event copy_usm2host(sycl::queue& queue,
                                 T* dest_host,
                                 const T* src_usm,
                                 std::int64_t count,
                                 const event_vector& deps = {}) {
    ONEDAL_ASSERT(count > 0);
    const std::size_t n = detail::integral_cast<std::size_t>(count);
    ONEDAL_ASSERT_MUL_OVERFLOW(std::size_t, sizeof(T), n);
    return memcpy_usm2host(queue, dest_host, src_usm, sizeof(T) * n, deps);
}

template <typename T>
inline sycl::usm::alloc get_usm_type(const array<T>& ary) {
    if (ary.get_queue().has_value()) {
        auto q = ary.get_queue().value();
        return sycl::get_pointer_type(ary.get_data(), q.get_context());
    }
    else {
        return sycl::usm::alloc::unknown;
    }
}

template <typename T>
inline bool is_device_usm(const array<T>& ary) {
    return get_usm_type(ary) == sycl::usm::alloc::device;
}

template <typename T>
inline bool is_shared_usm(const array<T>& ary) {
    return get_usm_type(ary) == sycl::usm::alloc::shared;
}

template <typename T>
inline bool is_host_usm(const array<T>& ary) {
    return get_usm_type(ary) == sycl::usm::alloc::host;
}

template <typename T>
inline bool is_device_friendly_usm(const array<T>& ary) {
    const auto pointer_type = get_usm_type(ary);
    return (pointer_type == sycl::usm::alloc::device) || //
           (pointer_type == sycl::usm::alloc::shared);
}

template <typename T>
inline bool is_known_usm(const array<T>& ary) {
    return get_usm_type(ary) != sycl::usm::alloc::unknown;
}

#endif

} // namespace oneapi::dal::backend

namespace oneapi::dal::preview::detail {
struct byte_alloc_iface;
} // namespace oneapi::dal::preview::detail

namespace oneapi::dal::preview::backend {

template <typename T>
struct inner_alloc {
    using byte_t = char;
    using value_type = T;
    using pointer = T*;

    inner_alloc(detail::byte_alloc_iface* byte_allocator) : byte_allocator_(byte_allocator) {}

    inner_alloc(const detail::byte_alloc_iface* byte_allocator)
            : byte_allocator_(const_cast<detail::byte_alloc_iface*>(byte_allocator)) {}

    template <typename V>
    inner_alloc(inner_alloc<V>& other) : byte_allocator_(other.get_byte_allocator()) {}

    template <typename V>
    inner_alloc(const inner_alloc<V>& other) {
        byte_allocator_ = const_cast<detail::byte_alloc_iface*>(other.get_byte_allocator());
    }

    T* allocate(std::int64_t n) {
        return reinterpret_cast<T*>(byte_allocator_->allocate(n * sizeof(T)));
    }

    void deallocate(T* ptr, std::int64_t n) {
        byte_allocator_->deallocate(reinterpret_cast<byte_t*>(ptr), n * sizeof(T));
    }

    oneapi::dal::detail::shared<T> make_shared_memory(std::int64_t n) {
        return oneapi::dal::detail::shared<T>(this->allocate(n), [=](T* p) {
            this->deallocate(p, n);
        });
    }

    detail::byte_alloc_iface* get_byte_allocator() {
        return byte_allocator_;
    }

    const detail::byte_alloc_iface* get_byte_allocator() const {
        return byte_allocator_;
    }

private:
    inner_alloc() = default;

    detail::byte_alloc_iface* byte_allocator_;
};

} // namespace oneapi::dal::preview::backend
