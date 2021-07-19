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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/rng/partial_shuffle.hpp"

#include "oneapi/dal/algo/kmeans_init/common.hpp"

namespace oneapi::dal::kmeans_init::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float, typename Method>
struct kmeans_init_kernel {
    static sycl::event compute_initial_centroids(sycl::queue& queue,
                                                 const pr::ndview<Float, 2>& data,
                                                 pr::ndview<Float, 2>& centroids) {
        using msg = dal::detail::error_messages;
        if constexpr (std::is_same_v<Method, kmeans_init::method::plus_plus_dense>) {
            throw unimplemented(
                msg::kmeans_init_plus_plus_dense_method_is_not_implemented_for_gpu());
        }

        if constexpr (std::is_same_v<Method, kmeans_init::method::parallel_plus_dense>) {
            throw unimplemented(
                msg::kmeans_init_parallel_plus_dense_method_is_not_implemented_for_gpu());
        }
        return sycl::event();
    }
};

template <typename Float>
struct kmeans_init_kernel<Float, kmeans_init::method::dense> {
    static sycl::event compute_initial_centroids(sycl::queue& queue,
                                                 const pr::ndview<Float, 2>& data,
                                                 pr::ndview<Float, 2>& centroids) {
        ONEDAL_ASSERT(data.get_dimension(1) == centroids.get_dimension(1));
        ONEDAL_ASSERT(data.get_dimension(0) >= centroids.get_dimension(0));
        const std::int64_t cluster_count = centroids.get_dimension(0);
        const std::int64_t column_count = centroids.get_dimension(1);
        dal::detail::check_mul_overflow(cluster_count, column_count);
        const auto data_ptr = data.get_data();
        auto centroids_ptr = centroids.get_mutable_data();
        return bk::copy(queue, centroids_ptr, data_ptr, cluster_count * column_count);
    }
};

template <typename Float>
struct kmeans_init_kernel<Float, kmeans_init::method::random_dense> {
    static sycl::event compute_initial_centroids(sycl::queue& queue,
                                                 const pr::ndview<Float, 2>& data,
                                                 pr::ndview<Float, 2>& centroids) {
        ONEDAL_ASSERT(data.get_dimension(1) == centroids.get_dimension(1));
        ONEDAL_ASSERT(data.get_dimension(0) >= centroids.get_dimension(0));
        const std::uint64_t row_count =
            dal::detail::integral_cast<std::uint64_t>(data.get_dimension(0));
        const std::uint64_t cluster_count =
            dal::detail::integral_cast<std::uint64_t>(centroids.get_dimension(0));
        const std::int64_t column_count = centroids.get_dimension(1);
        const auto data_ptr = data.get_data();
        auto centroids_ptr = centroids.get_mutable_data();
        dal::detail::check_mul_overflow(
            cluster_count,
            dal::detail::integral_cast<std::uint64_t>(sizeof(std::int32_t)));

        auto indices = pr::ndarray<size_t, 1>::empty(queue, cluster_count);
        pr::partial_shuffle{}.generate(indices, row_count);
        auto indices_ptr = indices.get_data();

        const std::int64_t required_local_size = bk::device_max_wg_size(queue);
        const std::int64_t local_size = std::min(bk::down_pow2(column_count), required_local_size);

        auto gather_event = queue.submit([&](sycl::handler& cgh) {
            sycl::stream out(1024, 256, cgh);
            const auto range = bk::make_multiple_nd_range_2d(
                { local_size, dal::detail::integral_cast<std::int64_t>(cluster_count) },
                { local_size, 1 });
            cgh.parallel_for(range, [=](sycl::nd_item<2> id) {
                const auto cluster = id.get_global_id(1);
                const std::int64_t local_id = id.get_local_id(0);
                const std::int64_t local_size = id.get_local_range()[0];
                const std::uint64_t index = indices_ptr[cluster];
                for (std::int64_t k = local_id; k < column_count; k += local_size) {
                    centroids_ptr[cluster * column_count + k] = data_ptr[index * column_count + k];
                }
            });
        });
        bk::smart_event event(gather_event);
        event.attach(indices_ptr);
        return event;
    }
};

#endif

} // namespace oneapi::dal::kmeans_init::backend
