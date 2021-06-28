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

#include <daal/src/algorithms/k_nearest_neighbors/bf_knn_classification_predict_kernel.h>
#include <algorithms/k_nearest_neighbors/bf_knn_classification_model.h>

#include "oneapi/dal/algo/knn/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"
#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using daal::services::Status;
using dal::backend::context_cpu;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace daal_knn = daal::algorithms::bf_knn_classification;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_knn_bf_kernel_t =
    daal_knn::prediction::internal::KNNClassificationPredictKernel<Float, Cpu>;

template <typename Float>
static infer_result<task::classification> call_daal_kernel(const context_cpu &ctx,
                                                           const descriptor_t &desc,
                                                           const table &data,
                                                           model<task::classification> m) {
    const std::int64_t row_count = data.get_row_count();

    auto arr_labels = array<Float>::empty(1 * row_count);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_labels = interop::convert_to_daal_homogen_table(arr_labels, row_count, 1);

    const auto data_use_in_model = daal_knn::doNotUse;
    daal_knn::Parameter original_daal_parameter(
        dal::detail::integral_cast<std::size_t>(desc.get_class_count()),
        dal::detail::integral_cast<std::size_t>(desc.get_neighbor_count()),
        data_use_in_model);

    daal_knn::prediction::internal::KernelParameter daal_parameter;
    daal_parameter.nClasses = original_daal_parameter.nClasses;
    daal_parameter.k = original_daal_parameter.k;
    daal_parameter.resultsToCompute = original_daal_parameter.resultsToCompute;
    daal_parameter.resultsToEvaluate = original_daal_parameter.resultsToEvaluate;

    auto distance_impl = detail::get_distance_impl(desc);
    if (!distance_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_distance_type() };
    }

    daal_parameter.pairwiseDistance = distance_impl->get_daal_distance_type();
    daal_parameter.minkowskiDegree = distance_impl->get_degree();

    const auto daal_voting_mode = convert_to_daal_bf_voting_mode(desc.get_voting_mode());
    daal_parameter.voteWeights = daal_voting_mode;

    if(!dynamic_cast<brute_force_model_impl_cls*>(&dal::detail::get_impl(m))) {
        throw internal_error{ dal::detail::error_messages::unsupported_knn_model() };
    }

    const auto deserialized_model = dynamic_cast<brute_force_model_impl_cls*>(&dal::detail::get_impl(m));

    const auto daal_train_data = interop::convert_to_daal_table<Float>(deserialized_model->data_);
    const auto daal_train_labels = interop::convert_to_daal_table<Float>(deserialized_model->labels_);
    const std::int64_t column_count = daal_train_data.get_column_count();

    Status status;
    const auto model_ptr = daal_knn::ModelPtr(new daal_knn::Model(column_count));
    interop::status_to_exception(status);

    // Data or labels should not be copied, copy is already happened when
    // the tables are converted to NumericTables
    model_ptr->impl()->setData<Float>(daal_train_data, false);
    model_ptr->impl()->setLabels<Float>(daal_train_labels, false);

    interop::status_to_exception(interop::call_daal_kernel<Float, daal_knn_bf_kernel_t>(
        ctx,
        daal_data.get(),
        model_ptr.get(),
        daal_labels.get(),
        nullptr,
        nullptr,
        &daal_parameter));
    return infer_result<task::classification>().set_labels(
        dal::detail::homogen_table_builder{}.reset(arr_labels, row_count, 1).build());
}

template <typename Float>
static infer_result<task::classification> infer(const context_cpu &ctx,
                                                const descriptor_t &desc,
                                                const infer_input<task::classification> &input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float>
struct infer_kernel_cpu<Float, method::brute_force, task::classification> {
    infer_result<task::classification> operator()(
        const context_cpu &ctx,
        const descriptor_t &desc,
        const infer_input<task::classification> &input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, method::brute_force, task::classification>;
template struct infer_kernel_cpu<double, method::brute_force, task::classification>;

} // namespace oneapi::dal::knn::backend
