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

#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernels_impl.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans_init::backend {

using dal::backend::context_gpu;

namespace interop = dal::backend::interop;
namespace pr = dal::backend::primitives;

template <typename Task>
using result_t = compute_result<Task>;

template <typename Task>
using descriptor_t = detail::descriptor_base<Task>;

template <typename Task>
using input_t = compute_input<Task>;

template <typename Float, typename Method, typename Task>
static result_t<Task> call_daal_kernel(const context_gpu& ctx,
                                       const descriptor_t<Task>& params,
                                       const table& data) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const int64_t column_count = data.get_column_count();
    const int64_t row_count = data.get_row_count();
    const int64_t cluster_count = params.get_cluster_count();

    auto data_ptr =
        row_accessor<const Float>(data).pull(queue, { 0, -1 }, sycl::usm::alloc::device);
    auto arr_data = pr::ndarray<Float, 2>::wrap(data_ptr, { row_count, column_count });

    dal::detail::check_mul_overflow(cluster_count, column_count);
    auto arr_centroids = pr::ndarray<Float, 2>::empty(queue,
                                                      { cluster_count, column_count },
                                                      sycl::usm::alloc::device);

    kmeans_init_kernel<Float, Method>::compute_initial_centroids(queue, arr_data, arr_centroids)
        .wait_and_throw();
    return result_t<Task>().set_centroids(
        dal::homogen_table::wrap(arr_centroids.flatten(queue), cluster_count, column_count));
}

template <typename Float, typename Method, typename Task>
static result_t<Task> compute(const context_gpu& ctx,
                              const descriptor_t<Task>& desc,
                              const input_t<Task>& input) {
    return call_daal_kernel<Float, Method, Task>(ctx, desc, input.get_data());
}

template <typename Float, typename Method, typename Task>
result_t<Task> compute_kernel_gpu<Float, Method, Task>::operator()(
    const context_gpu& ctx,
    const descriptor_t<Task>& desc,
    const input_t<Task>& input) const {
    return compute<Float, Method, Task>(ctx, desc, input);
}

template struct compute_kernel_gpu<float, method::dense, task::init>;
template struct compute_kernel_gpu<double, method::dense, task::init>;
template struct compute_kernel_gpu<float, method::random_dense, task::init>;
template struct compute_kernel_gpu<double, method::random_dense, task::init>;
template struct compute_kernel_gpu<float, method::plus_plus_dense, task::init>;
template struct compute_kernel_gpu<double, method::plus_plus_dense, task::init>;
template struct compute_kernel_gpu<float, method::parallel_plus_dense, task::init>;
template struct compute_kernel_gpu<double, method::parallel_plus_dense, task::init>;

} // namespace oneapi::dal::kmeans_init::backend
