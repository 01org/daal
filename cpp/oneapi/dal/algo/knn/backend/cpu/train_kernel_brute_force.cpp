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

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
#include "oneapi/dal/algo/knn/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include <daal/src/algorithms/k_nearest_neighbors/bf_knn_classification_train_kernel.h>
#include <algorithms/k_nearest_neighbors/bf_knn_classification_model.h>

namespace oneapi::dal::knn::backend {

using dal::backend::context_cpu;

namespace daal_knn = daal::algorithms::bf_knn_classification;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_knn_bf_kernel_t = daal_knn::training::internal::KNNClassificationTrainKernel<Float, Cpu>;

template <typename Float, typename Task>
static train_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const table& data,
                                           const table& responses) {
    using model_t = model<Task>;
    const std::int64_t column_count = data.get_column_count();

    const auto data_use_in_model = daal_knn::doUse;
    daal_knn::Parameter daal_parameter(
        dal::detail::integral_cast<std::size_t>(desc.get_class_count()),
        dal::detail::integral_cast<std::size_t>(desc.get_neighbor_count()),
        data_use_in_model);

    const auto model_ptr = daal_knn::ModelPtr(new daal_knn::Model(column_count));

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    model_ptr->impl()->setData<Float>(daal_data, false);

    auto daal_responses = daal::data_management::NumericTablePtr();
    if constexpr (!std::is_same_v<Task, task::search>) {
        daal_responses = interop::convert_to_daal_table<Float>(responses);
        model_ptr->impl()->setLabels<Float>(daal_responses, false);
    }

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_knn_bf_kernel_t>(ctx,
                                                               daal_data.get(),
                                                               daal_responses.get(),
                                                               model_ptr.get(),
                                                               daal_parameter,
                                                               *daal_parameter.engine));

    const auto model_impl = std::make_shared<brute_force_model_impl<Task>>(data, responses);
    return train_result<Task>().set_model(dal::detail::make_private<model_t>(model_impl));
}

template <typename Float, typename Task>
static train_result<Task> train(const context_cpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const train_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_responses());
}

template <typename Float, typename Task>
struct train_kernel_cpu<Float, method::brute_force, Task> {
    train_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::brute_force, task::classification>;
template struct train_kernel_cpu<double, method::brute_force, task::classification>;
template struct train_kernel_cpu<float, method::brute_force, task::search>;
template struct train_kernel_cpu<double, method::brute_force, task::search>;

} // namespace oneapi::dal::knn::backend
