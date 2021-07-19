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

#include <src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_predict_kernel_ucapi.h>

#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using dal::backend::context_gpu;

template <typename Task>
using descriptor_t = detail::descriptor_base<Task>;

namespace daal_classifier = daal::algorithms::classifier;
namespace daal_knn = daal::algorithms::bf_knn_classification;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_knn_brute_force_kernel_t =
    daal_knn::prediction::internal::KNNClassificationPredictKernelUCAPI<Float>;

template <typename Float, typename Task>
static infer_result<Task> call_daal_kernel(const context_gpu& ctx,
                                           const descriptor_t<Task>& desc,
                                           const table& data,
                                           const model<Task>& m) {
    auto distance_impl = detail::get_distance_impl(desc);
    if (!distance_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_distance_type() };
    }
    else if (distance_impl->get_daal_distance_type() != detail::v1::daal_distance_t::minkowski ||
             distance_impl->get_degree() != 2.0) {
        throw internal_error{ dal::detail::error_messages::distance_is_not_supported_for_gpu() };
    }

    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t neighbor_count = desc.get_neighbor_count();

    auto arr_responses = array<Float>{};
    auto arr_indices = array<std::int64_t>{};
    auto arr_distance = array<Float>{};

    const auto daal_data = interop::convert_to_daal_table(queue, data);

    auto daal_responses = daal::data_management::NumericTablePtr();
    auto daal_indices = daal::data_management::NumericTablePtr();
    auto daal_distance = daal::data_management::NumericTablePtr();

    const auto data_use_in_model = daal_knn::doNotUse;
    daal_knn::Parameter daal_parameter(
        dal::detail::integral_cast<std::size_t>(desc.get_class_count()),
        dal::detail::integral_cast<std::size_t>(desc.get_neighbor_count()),
        data_use_in_model);

    if (desc.get_result_options() & result_options::responses) {
        if constexpr (std::is_same_v<Task, task::classification>) {
            arr_responses = array<Float>::empty(queue, 1 * row_count, sycl::usm::alloc::device);
            daal_responses = interop::convert_to_daal_table(queue, arr_responses, row_count, 1);
        }
    }
    else {
        daal_parameter.resultsToEvaluate = daal_classifier::none;
    }

    if (desc.get_result_options() & result_options::indices) {
        dal::detail::check_mul_overflow(neighbor_count, row_count);
        daal_parameter.resultsToCompute |= daal_knn::computeIndicesOfNeighbors;
        arr_indices =
            array<std::int64_t>::empty(queue, neighbor_count * row_count, sycl::usm::alloc::device);
        daal_indices =
            interop::convert_to_daal_table(queue, arr_indices, row_count, neighbor_count);
    }

    if (desc.get_result_options() & result_options::distances) {
        dal::detail::check_mul_overflow(neighbor_count, row_count);
        daal_parameter.resultsToCompute |= daal_knn::computeDistances;
        arr_distance =
            array<Float>::empty(queue, neighbor_count * row_count, sycl::usm::alloc::device);
        daal_distance =
            interop::convert_to_daal_table(queue, arr_distance, row_count, neighbor_count);
    }

    const auto model_ptr = convert_onedal_to_daal_knn_model<Float, Task>(queue, m);

    interop::status_to_exception(
        daal_knn_brute_force_kernel_t<Float>().compute(daal_data.get(),
                                                       model_ptr.get(),
                                                       daal_responses.get(),
                                                       daal_indices.get(),
                                                       daal_distance.get(),
                                                       &daal_parameter));

    auto result = infer_result<Task>{}.set_result_options(desc.get_result_options());

    if (desc.get_result_options() & result_options::responses) {
        if constexpr (std::is_same_v<Task, task::classification>) {
            result = result.set_responses(
                dal::detail::homogen_table_builder{}.reset(arr_responses, row_count, 1).build());
        }
    }

    if (desc.get_result_options() & result_options::indices) {
        result = result.set_indices(dal::detail::homogen_table_builder{}
                                        .reset(arr_indices, row_count, neighbor_count)
                                        .build());
    }

    if (desc.get_result_options() & result_options::indices) {
        result = result.set_distances(dal::detail::homogen_table_builder{}
                                          .reset(arr_distance, row_count, neighbor_count)
                                          .build());
    }
    return result;
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_gpu& ctx,
                                const descriptor_t<Task>& desc,
                                const infer_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float, typename Task>
struct infer_kernel_gpu<Float, method::brute_force, Task> {
    infer_result<Task> operator()(const context_gpu& ctx,
                                  const descriptor_t<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, method::brute_force, task::classification>;
template struct infer_kernel_gpu<double, method::brute_force, task::classification>;
template struct infer_kernel_gpu<float, method::brute_force, task::search>;
template struct infer_kernel_gpu<double, method::brute_force, task::search>;

} // namespace oneapi::dal::knn::backend
