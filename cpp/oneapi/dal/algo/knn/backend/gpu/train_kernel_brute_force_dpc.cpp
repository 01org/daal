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

#include <src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_model_ucapi_impl.h>
#include <src/algorithms/k_nearest_neighbors/oneapi/bf_knn_classification_train_kernel_ucapi.h>

#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using dal::backend::context_gpu;

template <typename Task>
using descriptor_t = detail::descriptor_base<Task>;

namespace daal_knn = daal::algorithms::bf_knn_classification;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_knn_brute_force_kernel_t =
    daal_knn::training::internal::KNNClassificationTrainKernelUCAPI<Float>;

template <typename Float, typename Task>
static train_result<Task> call_daal_kernel(const context_gpu& ctx,
                                           const descriptor_t<Task>& desc,
                                           const table& data,
                                           const table& responses) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::int64_t column_count = data.get_column_count();
    const auto daal_data = interop::convert_to_daal_table(queue, data);
    auto daal_responses = daal::data_management::NumericTablePtr();

    const auto data_use_in_model = daal_knn::doNotUse;
    daal_knn::Parameter daal_parameter(
        dal::detail::integral_cast<std::size_t>(desc.get_class_count()),
        dal::detail::integral_cast<std::size_t>(desc.get_neighbor_count()),
        data_use_in_model);

    auto distance_impl = detail::get_distance_impl(desc);
    if (!distance_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_distance_type() };
    }
    else if (distance_impl->get_daal_distance_type() != detail::v1::daal_distance_t::minkowski ||
             distance_impl->get_degree() != 2.0) {
        throw internal_error{ dal::detail::error_messages::distance_is_not_supported_for_gpu() };
    }

    daal::algorithms::classifier::ModelPtr model_ptr(new daal_knn::Model(column_count));
    if (!model_ptr) {
        throw host_bad_alloc();
    }

    auto knn_model = static_cast<daal_knn::Model*>(model_ptr.get());
    // No need for data copy in case of brute-force. The data and responses
    // are not modified by the algorithm.
    const bool copy_data_responses = false;
    knn_model->impl()->setData<Float>(daal_data, copy_data_responses);
    if constexpr (!std::is_same_v<Task, task::search>) {
        daal_responses = interop::convert_to_daal_table(queue, responses);
        knn_model->impl()->setLabels<Float>(daal_responses, false);
    }

    interop::status_to_exception(
        daal_knn_brute_force_kernel_t<Float>().compute(daal_data.get(),
                                                       daal_responses.get(),
                                                       knn_model,
                                                       daal_parameter,
                                                       *daal_parameter.engine.get()));

    const auto model_impl = std::make_shared<brute_force_model_impl<Task>>(data, responses);
    return train_result<Task>().set_model(dal::detail::make_private<model<Task>>(model_impl));
}

template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const descriptor_t<Task>& desc,
                                const train_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_responses());
}

template <typename Float, typename Task>
struct train_kernel_gpu<Float, method::brute_force, Task> {
    train_result<Task> operator()(const context_gpu& ctx,
                                  const descriptor_t<Task>& desc,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, method::brute_force, task::classification>;
template struct train_kernel_gpu<double, method::brute_force, task::classification>;
template struct train_kernel_gpu<float, method::brute_force, task::search>;
template struct train_kernel_gpu<double, method::brute_force, task::search>;

} // namespace oneapi::dal::knn::backend
