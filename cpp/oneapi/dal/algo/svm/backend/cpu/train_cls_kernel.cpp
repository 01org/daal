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

#include <daal/src/algorithms/svm/svm_train_boser_kernel.h>
#include <daal/src/algorithms/svm/svm_train_thunder_kernel.h>
#include <daal/src/algorithms/svm/svm_train.h>
#include <daal/src/algorithms/multiclassclassifier/multiclassclassifier_train_kernel.h>
#include <daal/src/algorithms/multiclassclassifier/multiclassclassifier_svm_model.h>

#include "algorithms/svm/svm_train.h"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/svm/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/algo/svm/backend/utils.hpp"
#include "oneapi/dal/algo/svm/backend/model_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::svm::backend {

using dal::backend::context_cpu;

namespace daal_svm = daal::algorithms::svm;
namespace daal_classifier = daal::algorithms::classifier;
namespace daal_multiclass = daal::algorithms::multi_class_classifier;
namespace daal_multiclass_internal = daal_multiclass::internal;

namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu, typename Method>
using daal_svm_kernel_t =
    daal_svm::training::internal::SVMTrainImpl<to_daal_method<Method>::value, Float, Cpu>;

template <typename Float, daal::CpuType Cpu>
using daal_multiclass_kernel_t = daal_multiclass::training::internal::
    MultiClassClassifierTrainKernel<daal_multiclass::training::oneAgainstOne, Float, Cpu>;

template <typename Task>
static auto create_daal_parameter(const detail::descriptor_base<Task>& desc, const bool is_dense) {
    const std::uint64_t cache_megabyte = static_cast<std::uint64_t>(desc.get_cache_size());
    constexpr std::uint64_t megabyte = 1024 * 1024;
    dal::detail::check_mul_overflow(cache_megabyte, megabyte);
    const std::uint64_t cache_byte = cache_megabyte * megabyte;

    auto kernel_impl = detail::get_kernel_function_impl(desc);
    if (!kernel_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_kernel_function_type() };
    }
    const auto daal_kernel = kernel_impl->get_daal_kernel_function(is_dense);
    daal_svm::training::internal::KernelParameter daal_svm_parameter;

    daal_svm_parameter.kernel = daal_kernel;
    daal_svm_parameter.accuracyThreshold = desc.get_accuracy_threshold();
    daal_svm_parameter.tau = desc.get_tau();
    daal_svm_parameter.maxIterations =
        dal::detail::integral_cast<std::size_t>(desc.get_max_iteration_count());
    daal_svm_parameter.doShrinking = desc.get_shrinking();
    daal_svm_parameter.cacheSize = cache_byte;

    if constexpr (std::is_same_v<Task, task::nu_classification>) {
        daal_svm_parameter.nu = desc.get_nu();
        daal_svm_parameter.svmType = daal_svm::training::internal::SvmType::nu_classification;
    }
    else {
        daal_svm_parameter.C = desc.get_c();
        daal_svm_parameter.svmType = daal_svm::training::internal::SvmType::classification;
    }

    return daal_svm_parameter;
}

template <typename Float, typename Method, typename Task, typename ModelImpl>
static train_result<Task> call_multiclass_daal_kernel(const context_cpu& ctx,
                                                      const detail::descriptor_base<Task>& desc,
                                                      const table& data,
                                                      const table& responses,
                                                      const table& weights,
                                                      const std::uint64_t class_count) {
    const std::int64_t column_count = data.get_column_count();

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_weights = interop::convert_to_daal_table<Float>(weights);

    const bool is_dense{ data.get_kind() != dal::detail::csr_table::kind() };
    daal_svm::training::internal::KernelParameter daal_svm_parameter =
        create_daal_parameter<Task>(desc, is_dense);

    const auto daal_responses = interop::convert_to_daal_table<Float>(responses);

    daal_multiclass::training::internal::KernelParameter daal_multiclass_parameter;
    daal_multiclass_parameter.nClasses = class_count;

    daal_multiclass::Parameter daal_multiclass_parameter_public(class_count);

    auto daal_model =
        daal_multiclass::Model::create(column_count, &daal_multiclass_parameter_public);

    const auto daal_layout = daal_data->getDataLayout();
    auto daal_svm_model =
        daal_multiclass_internal::SvmModel::create<Float>(class_count, column_count, daal_layout);
    using svm_batch_t = typename daal_svm::training::internal::Batch<Float, to_daal_method<Method>::value>;
    auto svm_batch = daal::services::SharedPtr<svm_batch_t>(new svm_batch_t());
    svm_batch->parameter = daal_svm_parameter;
    daal_multiclass_parameter.training =
        daal::services::staticPointerCast<daal_classifier::training::Batch>(svm_batch);

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_multiclass_kernel_t>(ctx,
                                                                   daal_data.get(),
                                                                   daal_responses.get(),
                                                                   daal_weights.get(),
                                                                   daal_model.get(),
                                                                   daal_svm_model.get(),
                                                                   daal_multiclass_parameter));
    const std::int64_t n_sv = daal_svm_model->getSupportIndices()->getNumberOfRows();
    if (n_sv == 0) {
        return train_result<Task>{};
    }
    auto table_support_indices =
        interop::convert_from_daal_homogen_table<Float>(daal_svm_model->getSupportIndices());
    const auto trained_model = std::make_shared<ModelImpl>(new model_interop_cls{ daal_model });
    trained_model->class_count = class_count;

    auto trained_model_svm = convert_from_daal_multiclass_model<Task, Float>(daal_svm_model);

    trained_model->support_vectors = trained_model_svm.get_support_vectors();
    trained_model->biases = trained_model_svm.get_biases();
    trained_model->coeffs = trained_model_svm.get_coeffs();

    auto m = dal::detail::make_private<model<Task>>(trained_model);

    return train_result<Task>().set_model(m).set_support_indices(table_support_indices);
}

template <typename Float, typename Method, typename Task>
static train_result<Task> call_binary_daal_kernel(const context_cpu& ctx,
                                                  const detail::descriptor_base<Task>& desc,
                                                  const table& data,
                                                  const table& responses,
                                                  const table& weights) {
    const std::int64_t column_count = data.get_column_count();

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_weights = interop::convert_to_daal_table<Float>(weights);

    const bool is_dense{ data.get_kind() != dal::detail::csr_table::kind() };
    daal_svm::training::internal::KernelParameter daal_svm_parameter =
        create_daal_parameter<Task>(desc, is_dense);

    const binary_response_t<Float> old_unique_responses = get_unique_responses<Float>(responses);
    const auto new_responses =
        convert_binary_responses(responses, { Float(-1.0), Float(1.0) }, old_unique_responses);
    const auto daal_responses = interop::convert_to_daal_table<Float>(new_responses);

    const auto daal_layout = daal_data->getDataLayout();
    auto daal_model = daal_svm::Model::create<Float>(column_count, daal_layout);
    interop::status_to_exception(dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return daal_svm_kernel_t<
                   Float,
                   oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                   Method>()
            .compute(daal_data,
                     daal_weights,
                     *daal_responses,
                     daal_model.get(),
                     daal_svm_parameter);
    }));

    const std::int64_t n_sv = daal_model->getSupportIndices()->getNumberOfRows();
    if (n_sv == 0) {
        return train_result<Task>{};
    }

    auto table_support_indices =
        interop::convert_from_daal_homogen_table<Float>(daal_model->getSupportIndices());

    auto trained_model = convert_from_daal_model<Task, Float>(*daal_model)
                             .set_first_class_response(old_unique_responses.first)
                             .set_second_class_response(old_unique_responses.second);

    return train_result<Task>().set_model(trained_model).set_support_indices(table_support_indices);
}

template <typename Float, typename Method, typename Task, typename ModelImpl>
static train_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const table& data,
                                           const table& responses,
                                           const table& weights) {
    const std::uint64_t class_count = desc.get_class_count();

    if (class_count > 2) {
        return call_multiclass_daal_kernel<Float, Method, Task, ModelImpl>(ctx,
                                                                           desc,
                                                                           data,
                                                                           responses,
                                                                           weights,
                                                                           class_count);
    }
    else {
        return call_binary_daal_kernel<Float, Method, Task>(ctx, desc, data, responses, weights);
    }
}

template <typename Float, typename Method, typename Task, typename ModelImpl>
static train_result<Task> train(const context_cpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const train_input<Task>& input) {
    return call_daal_kernel<Float, Method, Task, ModelImpl>(ctx,
                                                            desc,
                                                            input.get_data(),
                                                            input.get_responses(),
                                                            input.get_weights());
}

template <typename Float, typename Method>
struct train_kernel_cpu<Float, Method, task::classification> {
    train_result<task::classification> operator()(
        const context_cpu& ctx,
        const detail::descriptor_base<task::classification>& desc,
        const train_input<task::classification>& input) const {
        return train<Float, Method, task::classification, model_impl_cls>(ctx, desc, input);
    }
};

template <typename Float, typename Method>
struct train_kernel_cpu<Float, Method, task::nu_classification> {
    train_result<task::nu_classification> operator()(
        const context_cpu& ctx,
        const detail::descriptor_base<task::nu_classification>& desc,
        const train_input<task::nu_classification>& input) const {
        return train<Float, Method, task::nu_classification, model_impl_nu_cls>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, method::thunder, task::classification>;
template struct train_kernel_cpu<float, method::smo, task::classification>;
template struct train_kernel_cpu<float, method::thunder, task::nu_classification>;
template struct train_kernel_cpu<double, method::thunder, task::classification>;
template struct train_kernel_cpu<double, method::smo, task::classification>;
template struct train_kernel_cpu<double, method::thunder, task::nu_classification>;

} // namespace oneapi::dal::svm::backend
