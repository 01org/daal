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

#include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"

#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using dal::backend::context_gpu;

template <typename Float, typename Task>
struct infer_kernel_gpu<Float, method::kd_tree, Task> {
    infer_result<Task> operator()(const context_gpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        throw unimplemented(
            dal::detail::error_messages::knn_kd_tree_method_is_not_implemented_for_gpu());
        return infer_result<Task>();
    }
};

template struct infer_kernel_gpu<float, method::kd_tree, task::classification>;
template struct infer_kernel_gpu<double, method::kd_tree, task::classification>;
template struct infer_kernel_gpu<float, method::kd_tree, task::search>;
template struct infer_kernel_gpu<double, method::kd_tree, task::search>;

} // namespace oneapi::dal::knn::backend
