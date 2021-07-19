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

#include "oneapi/dal/algo/decision_forest/backend/gpu/helper_rng_engine.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_misc_structs.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_impurity_data.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_service_kernels.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_feature_type.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_model_manager.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_best_split_impl.hpp"

#ifdef DISTRIBUTED_SUPPORT_ENABLED
#include "oneapi/dal/policy/mpi.hpp"
#endif

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float,
          typename Bin = std::uint32_t,
          typename Index = std::int32_t,
          typename Task = task::by_default>
class train_kernel_hist_impl {
    using result_t = train_result<Task>;
    using train_service_kernels_t = train_service_kernels<Float, Bin, Index, Task>;
    using bs_kernels_t = train_best_split_impl<Float, Bin, Index, Task>;
    using impl_const_t = impl_const<Index, Task>;
    using descriptor_t = detail::descriptor_base<Task>;
    using model_manager_t = train_model_manager<Float, Index, Task>;
    using context_t = train_context<Float, Index, Task>;
    using imp_data_t = impurity_data<Float, Index, Task>;
    using engine_list_t = dal::array<engine_impl>;
    using msg = dal::detail::error_messages;
#ifdef DISTRIBUTED_SUPPORT_ENABLED
    using comm_t = dal::preview::detail::mpi_communicator;
#endif

public:
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;

    train_kernel_hist_impl(sycl::queue& q) : queue_(q), train_service_kernels_(q), bs_kernels_(q) {}
    ~train_kernel_hist_impl() = default;

    result_t operator()(const descriptor_t& desc, const table& data, const table& labels);

private:
    std::uint64_t get_part_hist_required_mem_size(Index selected_ftr_count,
                                                  Index max_bin_count_among_ftrs,
                                                  Index class_count) const;

    sycl::event allreduce_ndarray_inplace(dal::backend::primitives::ndarray<Index, 1>& src_dst,
                                          const dal::backend::event_vector& deps = {});

    sycl::event allreduce_ndarray_inplace(dal::backend::primitives::ndarray<Float, 1>& src_dst,
                                          const dal::backend::event_vector& deps = {});

    sycl::event gen_initial_tree_order(
        context_t& ctx,
        engine_list_t& engine_list,
        dal::backend::primitives::ndarray<Index, 1>& node_list,
        dal::backend::primitives::ndarray<Index, 1>& tree_order_level,
        dal::backend::primitives::ndarray<Index, 1>& selected_row_global_host,
        dal::backend::primitives::ndarray<Index, 1>& selected_row_host,
        Index engine_offset,
        Index node_count);

    void validate_input(const descriptor_t& desc, const table& data, const table& labels) const;

    Index get_row_total_count(bool distr_mode, Index row_count);
    Index get_global_row_offset(bool distr_mode, Index row_count);

    void init_params(context_t& ctx,
                     const descriptor_t& desc,
                     const table& data,
                     const table& labels);
    void allocate_buffers(const context_t& ctx_);

    std::tuple<dal::backend::primitives::ndarray<Index, 1>, sycl::event> gen_feature_list(
        Index node_count,
        const dal::backend::primitives::ndarray<Index, 1>& node_vs_tree_map,
        engine_list_t& engine_list,
        const context_t& ctx_);

    sycl::event compute_initial_imp_for_node_list(
        const context_t& ctx,
        imp_data_t& imp_data_list,
        dal::backend::primitives::ndarray<Index, 1>& node_list,
        Index node_count,
        const dal::backend::event_vector& deps = {});

    sycl::event compute_initial_histogram_local(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        imp_data_t& imp_data_list,
        Index node_count,
        const dal::backend::event_vector& deps);

    sycl::event compute_initial_sum_local(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& tree_order,
        const dal::backend::primitives::ndarray<Index, 1>& node_list,
        dal::backend::primitives::ndarray<Float, 1>& sum_list,
        Index node_count,
        const dal::backend::event_vector& deps);

    sycl::event compute_initial_sum2cent_local(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& tree_order,
        const dal::backend::primitives::ndarray<Index, 1>& node_list,
        const dal::backend::primitives::ndarray<Float, 1>& sum_list,
        dal::backend::primitives::ndarray<Float, 1>& sum2cent_list,
        Index node_count,
        const dal::backend::event_vector& deps);

    sycl::event fin_initial_imp(const context_t& ctx,
                                const dal::backend::primitives::ndarray<Index, 1>& node_list,
                                const dal::backend::primitives::ndarray<Float, 1>& sum_list,
                                const dal::backend::primitives::ndarray<Float, 1>& sum2cent_list,
                                imp_data_t& imp_data_list,
                                Index node_count,
                                const dal::backend::event_vector& deps);

    sycl::event compute_initial_histogram(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        imp_data_t& imp_data_list,
        Index node_count,
        const dal::backend::event_vector& deps);

    sycl::event do_node_split(const dal::backend::primitives::ndarray<Index, 1>& node_list,
                              const dal::backend::primitives::ndarray<Index, 1>& node_vs_tree_map,
                              const imp_data_t& imp_data_list,
                              const imp_data_t& left_child_imp_data_list,
                              dal::backend::primitives::ndarray<Index, 1>& node_list_new,
                              dal::backend::primitives::ndarray<Index, 1>& node_vs_tree_map_new,
                              imp_data_t& imp_data_list_new,
                              Index node_count,
                              Index node_count_new,
                              const context_t& ctx,
                              const dal::backend::event_vector& deps);

    sycl::event compute_best_split(
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const imp_data_t& imp_data_list,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        imp_data_t& left_child_imp_data_list,
        dal::backend::primitives::ndarray<Float, 1>& nodeImpDecreaseList,
        bool updateImpDecreaseRequired,
        Index nNodes,
        const context_t& ctx,
        const dal::backend::event_vector& deps = {});

    std::tuple<dal::backend::primitives::ndarray<hist_type_t, 1>, sycl::event> compute_histogram(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const dal::backend::primitives::ndarray<Index, 1>& node_list,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        Index nPartialHistograms,
        Index node_count,
        const dal::backend::event_vector& deps = {});

    std::tuple<dal::backend::primitives::ndarray<hist_type_t, 1>, sycl::event>
    compute_histogram_distr(const context_t& ctx,
                            const dal::backend::primitives::ndarray<Bin, 2>& data,
                            const dal::backend::primitives::ndview<Float, 1>& response,
                            const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
                            const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
                            const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
                            const dal::backend::primitives::ndarray<Index, 1>& node_list,
                            const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
                            Index nodeIndicesOffset,
                            Index nPartialHistograms,
                            Index node_count,
                            const dal::backend::event_vector& deps = {});

    sycl::event compute_partial_histograms(
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const dal::backend::primitives::ndarray<Index, 1>& nodeList,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        dal::backend::primitives::ndarray<hist_type_t, 1>& partialHistograms,
        Index nPartialHistograms,
        Index node_count,
        const context_t& ctx,
        const dal::backend::event_vector& deps = {});

    sycl::event reduce_partial_histograms(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<hist_type_t, 1>& partialHistograms,
        dal::backend::primitives::ndarray<hist_type_t, 1>& histograms,
        Index nPartialHistograms,
        Index node_count,
        const dal::backend::event_vector& deps = {});

    sycl::event compute_partial_count_and_sum(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const dal::backend::primitives::ndarray<Index, 1>& node_list,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        dal::backend::primitives::ndarray<Float, 1>& partialHistograms,
        Index part_hist_count,
        Index node_count,
        const dal::backend::event_vector& deps = {},
        const task::regression task_val = {});

    sycl::event compute_partial_sum2cent(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndview<Float, 1>& sum_list,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const dal::backend::primitives::ndarray<Index, 1>& node_list,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        dal::backend::primitives::ndarray<Float, 1>& partialHistograms,
        Index part_hist_count,
        Index node_count,
        const dal::backend::event_vector& deps = {},
        const task::regression task_val = {});

    sycl::event sum_reduce_partial_histograms(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<Float, 1>& partialHistograms,
        dal::backend::primitives::ndarray<Float, 1>& histograms,
        Index part_hist_count,
        Index node_count,
        Index hist_prop_count,
        const dal::backend::event_vector& deps = {});

    sycl::event fin_histogram_distr(
        const context_t& ctx,
        const dal::backend::primitives::ndarray<Float, 1>& sum_list,
        const dal::backend::primitives::ndarray<Float, 1>& sum2cent_list,
        dal::backend::primitives::ndarray<Float, 1>& histogram_list,
        Index node_count,
        const dal::backend::event_vector& deps = {});

    sycl::event compute_best_split_by_histogram(
        const dal::backend::primitives::ndarray<hist_type_t, 1>& nodesHistograms,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const imp_data_t& imp_data_list,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        imp_data_t& left_child_imp_data_list,
        dal::backend::primitives::ndarray<Float, 1>& nodeImpDecreaseList,
        bool updateImpDecreaseRequired,
        Index node_count,
        const context_t& ctx,
        const dal::backend::event_vector& deps = {});

    sycl::event compute_best_split_single_pass(
        const dal::backend::primitives::ndarray<Bin, 2>& data,
        const dal::backend::primitives::ndview<Float, 1>& response,
        const dal::backend::primitives::ndarray<Index, 1>& treeOrder,
        const dal::backend::primitives::ndarray<Index, 1>& selectedFeatures,
        const dal::backend::primitives::ndarray<Index, 1>& binOffsets,
        const imp_data_t& imp_data_list,
        const dal::backend::primitives::ndarray<Index, 1>& nodeIndices,
        Index nodeIndicesOffset,
        dal::backend::primitives::ndarray<Index, 1>& nodeList,
        imp_data_t& left_child_imp_data_list,
        dal::backend::primitives::ndarray<Float, 1>& nodeImpDecreaseList,
        bool updateImpDecreaseRequired,
        Index node_count,
        const context_t& ctx,
        const dal::backend::event_vector& deps = {});

    Float compute_oob_error(const model_manager_t& model_manager,
                            const dal::backend::primitives::ndarray<Float, 1>& data_host,
                            const dal::backend::primitives::ndarray<Float, 1>& response_host,
                            const dal::backend::primitives::ndarray<Index, 1>& oob_row_list,
                            dal::backend::primitives::ndarray<hist_type_t, 1>& oob_per_obs_list,
                            Index tree_idx,
                            Index indicesOffset,
                            Index n,
                            const context_t& ctx,
                            const dal::backend::event_vector& deps = {});
    Float compute_oob_error_perm(
        const model_manager_t& model_manager,
        const dal::backend::primitives::ndarray<Float, 1>& data_host,
        const dal::backend::primitives::ndarray<Float, 1>& response_host,
        const dal::backend::primitives::ndarray<Index, 1>& oob_row_list,
        const dal::backend::primitives::ndarray<Index, 1>& permutation_host,
        Index tree_idx,
        Index indicesOffset,
        Index n,
        Index column_idx,
        const context_t& ctx,
        const dal::backend::event_vector& deps = {});

    sycl::event compute_results(const model_manager_t& model_manager,
                                const dal::backend::primitives::ndarray<Float, 1>& data_host,
                                const dal::backend::primitives::ndarray<Float, 1>& response_host,
                                const dal::backend::primitives::ndarray<Index, 1>& oob_row_list,
                                const dal::backend::primitives::ndarray<Index, 1>& oobRowsNumList,
                                dal::backend::primitives::ndarray<hist_type_t, 1>& oob_per_obs_list,
                                dal::backend::primitives::ndarray<Float, 1>& var_imp,
                                dal::backend::primitives::ndarray<Float, 1>& var_imp_variance,
                                const dal::array<engine_impl>& engine_arr,
                                Index tree_idx,
                                Index tree_in_block,
                                Index built_tree_count,
                                const context_t& ctx,
                                const dal::backend::event_vector& deps = {});

    sycl::event finalize_oob_error(
        const dal::backend::primitives::ndarray<Float, 1>& response_host,
        dal::backend::primitives::ndarray<hist_type_t, 1>& oob_per_obs_list,
        dal::backend::primitives::ndarray<Float, 1>& res_oob_err,
        dal::backend::primitives::ndarray<Float, 1>& res_oob_err_obs,
        const context_t& ctx,
        const dal::backend::event_vector& deps = {});

    sycl::event finalize_var_imp(dal::backend::primitives::ndarray<Float, 1>& var_imp,
                                 dal::backend::primitives::ndarray<Float, 1>& var_imp_variance,
                                 const context_t& ctx,
                                 const dal::backend::event_vector& deps = {});

private:
    sycl::queue queue_;
#ifdef DISTRIBUTED_SUPPORT_ENABLED
    comm_t comm_;
#endif

    train_service_kernels_t train_service_kernels_;
    bs_kernels_t bs_kernels_;

    // algo buffers which are allocated one time at the beggining
    dal::backend::primitives::ndarray<Bin, 2> full_data_nd_;
    dal::backend::primitives::ndarray<Index, 1> ftr_bin_offsets_nd_;
    std::vector<dal::backend::primitives::ndarray<Float, 1>> bin_borders_host_;
    dal::backend::primitives::ndarray<Float, 1> response_nd_;
    dal::backend::primitives::ndarray<Float, 1> response_host_;
    dal::backend::primitives::ndarray<Float, 1> data_host_;

    dal::backend::primitives::ndarray<Index, 1> selected_row_host_;
    dal::backend::primitives::ndarray<Index, 1> selected_row_global_host_;
    dal::backend::primitives::ndarray<Index, 1> tree_order_lev_;
    dal::backend::primitives::ndarray<Index, 1> tree_order_lev_buf_;

    dal::backend::primitives::ndarray<Float, 1> node_imp_decr_list_;

    dal::backend::primitives::ndarray<hist_type_t, 1> oob_per_obs_list_;
    dal::backend::primitives::ndarray<Float, 1> var_imp_variance_host_;

    dal::backend::primitives::ndarray<Float, 1> res_var_imp_;
};

#endif

} // namespace oneapi::dal::decision_forest::backend
