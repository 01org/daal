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

#include "oneapi/dal/algo/rbf_kernel/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/math.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

namespace oneapi::dal::rbf_kernel::backend {

using dal::backend::context_gpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace pr = dal::backend::primitives;

template <typename Float>
inline auto compute_exponents(sycl::queue& queue,
                              const pr::ndview<Float, 1>& sqr_x_nd,
                              const pr::ndview<Float, 1>& sqr_y_nd,
                              pr::ndview<Float, 2>& res_nd,
                              double sigma,
                              const dal::backend::event_vector& deps = {}) {
    const std::int64_t x_row_count = sqr_x_nd.get_dimension(0);
    const std::int64_t y_row_count = sqr_y_nd.get_dimension(0);
    ONEDAL_ASSERT(res_nd.get_count() == x_row_count * y_row_count);

    const Float coeff = static_cast<Float>(-0.5 / (sigma * sigma));

    const Float* sqr_x_ptr = sqr_x_nd.get_data();
    const Float* sqr_y_ptr = sqr_y_nd.get_data();
    Float* res_ptr = res_nd.get_mutable_data();

    const Float threshold = dal::backend::exp_low_threshold<Float>();

    const auto wg_size = dal::backend::propose_wg_size(queue);
    const auto range =
        dal::backend::make_multiple_nd_range_2d({ x_row_count, y_row_count }, { wg_size, 1 });

    auto compute_rbf_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        const std::size_t ld = y_row_count;

        cgh.parallel_for(range, [=](sycl::nd_item<2> item) {
            const std::size_t i = item.get_global_id(0);
            const std::size_t j = item.get_global_id(1);
            const Float sqr_x_i = sqr_x_ptr[i];
            const Float sqr_y_j = sqr_y_ptr[j];
            const Float res_rbf_ij = res_ptr[i * ld + j];
            const Float arg = sycl::fmax((sqr_x_i + sqr_y_j + res_rbf_ij) * coeff, threshold);

            res_ptr[i * ld + j] = sycl::exp(arg);
        });
    });

    return compute_rbf_event;
}

template <typename Float>
inline auto compute_rbf(sycl::queue& queue,
                        const pr::ndview<Float, 2>& x_nd,
                        const pr::ndview<Float, 2>& y_nd,
                        pr::ndview<Float, 2>& res_nd,
                        double sigma,
                        const dal::backend::event_vector& deps = {}) {
    const std::int64_t x_row_count = x_nd.get_dimension(0);
    const std::int64_t y_row_count = y_nd.get_dimension(0);

    auto sqr_x_nd = pr::ndarray<Float, 1>::empty(queue, { x_row_count }, sycl::usm::alloc::device);
    auto sqr_y_nd = pr::ndarray<Float, 1>::empty(queue, { y_row_count }, sycl::usm::alloc::device);

    auto reduce_x_event =
        pr::reduce_by_rows(queue, x_nd, sqr_x_nd, pr::sum<Float>{}, pr::square<Float>{}, deps);
    auto reduce_y_event =
        pr::reduce_by_rows(queue, y_nd, sqr_y_nd, pr::sum<Float>{}, pr::square<Float>{}, deps);

    constexpr Float alpha = -2.0;
    constexpr Float beta = 0.0;
    auto gemm_event = pr::gemm(queue, x_nd, y_nd.t(), res_nd, alpha, beta);

    auto compute_exponents_event =
        compute_exponents(queue,
                          sqr_x_nd,
                          sqr_y_nd,
                          res_nd,
                          sigma,
                          { reduce_x_event, reduce_y_event, gemm_event });

    auto smart_event =
        dal::backend::smart_event{ compute_exponents_event }.attach(sqr_x_nd).attach(sqr_y_nd);

    return smart_event;
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    const auto x = input.get_x();
    const auto y = input.get_y();

    auto& queue = ctx.get_queue();

    const std::int64_t x_row_count = x.get_row_count();
    const std::int64_t y_row_count = y.get_row_count();

    ONEDAL_ASSERT(x.get_column_count() == y.get_column_count());
    dal::detail::check_mul_overflow(x_row_count, y_row_count);

    const auto x_nd = pr::table2ndarray<Float>(queue, x, sycl::usm::alloc::device);
    const auto y_nd = pr::table2ndarray<Float>(queue, y, sycl::usm::alloc::device);

    auto res_nd =
        pr::ndarray<Float, 2>::empty(queue, { x_row_count, y_row_count }, sycl::usm::alloc::device);

    auto compute_rbf_event = compute_rbf(queue, x_nd, y_nd, res_nd, desc.get_sigma());

    const auto res_array = res_nd.flatten(queue, { compute_rbf_event });
    auto res_table = homogen_table::wrap(res_array, x_row_count, y_row_count);

    return result_t{}.set_values(res_table);
}

template <typename Float>
struct compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::rbf_kernel::backend
