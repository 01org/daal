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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/algo/decision_forest/train_types.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_misc_structs.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_impurity_data.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float,
          typename Bin = std::uint32_t,
          typename Index = std::int32_t,
          typename Task = task::by_default>
class train_best_split_impl {
    using result_t = train_result<Task>;
    using impl_const_t = impl_const<Index, Task>;
    using descriptor_t = detail::descriptor_base<Task>;
    using context_t = train_context<Float, Index, Task>;
    using imp_data_t = impurity_data<Float, Index, Task>;
    using msg = dal::detail::error_messages;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;

public:
    train_best_split_impl(sycl::queue& q) : queue_(q) {}
    ~train_best_split_impl() = default;

    sycl::event compute_best_split_by_histogram(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<hist_type_t, 1>& node_hist_list,
        const dal::backend::primitives::ndarray<Index, 1>& selected_ftr_list,
        const dal::backend::primitives::ndarray<Index, 1>& bin_offset_list,
        const imp_data_t& imp_data_list,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index node_ind_ofs,
        dal::backend::primitives::ndarray<Index, 1>& node_list,
        imp_data_t& left_child_imp_data_list,
        dal::backend::primitives::ndarray<Float, 1>& node_imp_dec_list,
        bool update_imp_dec_required,
        Index node_count,
        const dal::backend::event_vector& deps = {});

    sycl::event compute_best_split_single_pass(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& tree_order,
        const dal::backend::primitives::ndarray<Index, 1>& selected_ftr_list,
        const dal::backend::primitives::ndarray<Index, 1>& bin_offset_list,
        const imp_data_t& imp_data_list,
        const dal::backend::primitives::ndarray<Index, 1>& node_ind_list,
        Index node_ind_ofs,
        dal::backend::primitives::ndarray<Index, 1>& node_list,
        imp_data_t& left_child_imp_data_list,
        dal::backend::primitives::ndarray<Float, 1>& node_imp_dec_list,
        bool update_imp_dec_required,
        Index node_count,
        const be::event_vector& deps = {});

private:
    sycl::queue queue_;
};

#endif

} // namespace oneapi::dal::decision_forest::backend
