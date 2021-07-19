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

#include <daal/src/algorithms/k_nearest_neighbors/kdtree_knn_classification_train_kernel.h>
#include <src/algorithms/k_nearest_neighbors/kdtree_knn_classification_model_impl.h>

#include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
#include "oneapi/dal/algo/knn/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using daal::services::Status;
using dal::backend::context_cpu;

namespace daal_knn = daal::algorithms::kdtree_knn_classification;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_knn_kd_tree_kernel_t = daal_knn::training::internal::
    KNNClassificationTrainBatchKernel<Float, daal_knn::training::defaultDense, Cpu>;

template <typename Float, typename Task>
static train_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const table& data,
                                           const table& responses) {
    using model_t = model<Task>;
    using daal_model_interop_t = model_interop;
    const std::int64_t column_count = data.get_column_count();

    const auto daal_data = interop::copy_to_daal_homogen_table<Float>(data);
    const std::int64_t dummy_seed = 777;
    const auto data_use_in_model = daal_knn::doUse;
    daal_knn::Parameter daal_parameter(
        dal::detail::integral_cast<std::size_t>(desc.get_class_count()),
        dal::detail::integral_cast<std::size_t>(desc.get_neighbor_count()),
        dal::detail::integral_cast<int>(dummy_seed),
        data_use_in_model);

    Status status;
    const auto model_ptr = daal_knn::Model::create(column_count, &status);
    interop::status_to_exception(status);

    auto knn_model = static_cast<daal_knn::Model*>(model_ptr.get());
    // Data or responses should not be copied, copy is already happened when
    // the tables are converted to NumericTables
    const bool copy_data_responses = data_use_in_model == daal_knn::doNotUse;
    knn_model->impl()->setData<Float>(daal_data, copy_data_responses);

    auto daal_responses = daal::data_management::NumericTablePtr();
    if constexpr (!std::is_same_v<Task, task::search>) {
        daal_responses = interop::copy_to_daal_homogen_table<Float>(responses);
        knn_model->impl()->setLabels<Float>(daal_responses, copy_data_responses);
    }

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_knn_kd_tree_kernel_t>(ctx,
                                                                    daal_data.get(),
                                                                    daal_responses.get(),
                                                                    knn_model,
                                                                    *daal_parameter.engine.get()));

    auto interop = new daal_model_interop_t(model_ptr);
    const auto model_impl = std::make_shared<kd_tree_model_impl<Task>>(interop);
    return train_result<Task>().set_model(dal::detail::make_private<model_t>(model_impl));
}

template <typename Float, typename Task>
static train_result<Task> train(const context_cpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const train_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_responses());
}

template <typename Float, typename Task>
struct train_kernel_cpu<Float, method::kd_tree, Task> {
    train_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::kd_tree, task::classification>;
template struct train_kernel_cpu<double, method::kd_tree, task::classification>;
template struct train_kernel_cpu<float, method::kd_tree, task::search>;
template struct train_kernel_cpu<double, method::kd_tree, task::search>;

} // namespace oneapi::dal::knn::backend
