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

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

//#define DISTRIBUTED_SUPPORT_ENABLED

namespace oneapi::dal::decision_forest::backend {

// task_types
template <typename Float, typename Index, typename Task = task::by_default>
struct task_types;

template <typename Float, typename Index>
struct task_types<Float, Index, task::classification> {
    using hist_type_t = Index; // histogram data type
};

template <typename Float, typename Index>
struct task_types<Float, Index, task::regression> {
    using hist_type_t = Float; // histogram data type
};

template <typename Index, typename Task>
struct impl_const;

template <typename Index>
struct impl_const<Index, task::classification> {
    constexpr static Index bad_val_ = -1;
    constexpr static Index leaf_mark_ = bad_val_;
    constexpr static Index node_prop_count_ = 8; // rows offset, rows count, ftr id, ftr val(bin),
        // left part rows count, response
    // node props mapping
    constexpr static Index ind_ofs = 0; // property index for local row offset
    constexpr static Index ind_lrc = 1; // property index for local row count
    constexpr static Index ind_fid = 2; // property index for local row count
    constexpr static Index ind_bin = 3; // property index for local row count
    constexpr static Index ind_lch_grc = 4; // property index for left child global row count
    constexpr static Index ind_win = 5; // property index for winner class
    constexpr static Index ind_grc = 6; // property index for global row count
    constexpr static Index ind_lch_lrc = 7; // property index for left child local row count

    constexpr static Index node_imp_prop_count_ = 1; // only node impurity is stored
    constexpr static Index oob_aux_prop_count_ = 0; // 0 due to class count is used instead
        // added for consistency with regression
    constexpr static Index max_private_class_hist_buff_size = 16;
    constexpr static Index private_hist_buff_size = max_private_class_hist_buff_size;

    constexpr static Index hist_prop_count_ = 0; // isn't used for classififcation
    constexpr static Index hist_prop_sum_count_ = 0; // isn't used for classififcation
    constexpr static Index hist_prop_sum2cent_count_ = 0; // isn't used for classififcation
    // maximal number of classses which can be stored private memeory
    // should be replaced with specialization constants
};

template <typename Index>
struct impl_const<Index, task::regression> {
    constexpr static Index bad_val_ = -1;
    constexpr static Index leaf_mark_ = bad_val_;
    constexpr static Index node_prop_count_ = 8; // rows offset, rows count, ftr id, ftr val(bin),
        // left part rows count, response
    // node props mapping
    constexpr static Index ind_ofs = 0; // property index for local row offset
    constexpr static Index ind_lrc = 1; // property index for local row count
    constexpr static Index ind_fid = 2; // property index for local row count
    constexpr static Index ind_bin = 3; // property index for local row count
    constexpr static Index ind_lch_grc = 4; // property index for left child global row count
    //constexpr static Index ind_win  = 5; // is not used for regression, added for consistency
    constexpr static Index ind_grc = 6; // property index for global row count
    constexpr static Index ind_lch_lrc = 7; // property index for left child local row count

    constexpr static Index node_imp_prop_count_ = 2; // mean and sum2cent, mean also is a response
    constexpr static Index hist_prop_count_ =
        node_imp_prop_count_ + 1; // obs count + imp_prop_count
    constexpr static Index private_hist_buff_size = hist_prop_count_;
    constexpr static Index oob_aux_prop_count_ = 2; // cumulative value and count
    constexpr static Index hist_prop_sum_count_ = 2; // obs count, sum
    constexpr static Index hist_prop_sum2cent_count_ = 1;
};

template <typename Float, typename Index = std::int32_t, typename Task = task::by_default>
struct train_context {
    bool distr_mode_ = false;

    Index class_count_ = 0;
    Index row_count_ = 0;
    Index row_total_count_ = 0;
    Index column_count_ = 0;
    Index total_bins_ = 0;
    Index tree_count_ = 0;

    Index global_row_offset_ = 0;

    Index selected_ftr_count_ = 0;
    Index selected_row_count_ = 0;
    Index selected_row_total_count_ = 0;
    Index min_observations_in_leaf_node_ = 0;
    Index max_tree_depth_ = 0;

    Float impurity_threshold_;
    Float float_min_;
    Index index_max_;

    bool mda_required_ = false;
    bool mda_scaled_required_ = false;
    bool mdi_required_ = false;
    bool oob_required_ = false;
    bool oob_err_required_ = false;
    bool oob_err_obs_required_ = false;
    bool bootstrap_ = false;

    Index total_bin_count_ = 0;
    Index max_bin_count_among_ftrs_ = 0;

    Index tree_in_block_ = 0;
    Index preferable_local_size_for_part_hist_kernel_ = 0;
    Index max_part_hist_cumulative_size_ = 0;
    Index oob_prop_count_ = 0;

    static constexpr inline double global_mem_fraction_for_tree_block_ = 0.6;
    // part of free global mem which can be used for processing block of tree
    static constexpr inline double global_mem_fraction_for_part_hist_ = 0.2;
    // part of free global mem which can be used for partial histograms

    static constexpr inline std::uint64_t max_mem_alloc_size_for_algo_ = 1073741824;
    // 1 Gb it showed better efficiency than using just platform info.maxMemAllocSize
    static constexpr inline Index min_row_block_count_to_use_max_part_hist_count_ = 16384;
    static constexpr inline Index min_row_block_count_for_one_hist_ = 128;
    static constexpr inline Index max_part_hist_count_ = 256;
    static constexpr inline Index reduce_local_size_part_hist_ = 64;

    static constexpr inline Index min_preferable_local_size_for_part_hist_kernel_ = 32;

    // update _nNode naming
    static constexpr inline Index node_group_count_ = 3;
    // all nodes are split on groups (big, medium, small)
    static constexpr inline Index node_group_prop_count_ = 2;
    // each nodes Group contains props: numOfNodes, maxNumOfBlocks

    static constexpr inline Index preferable_group_size_ = 256;
    static constexpr inline Index preferable_sbg_size_ = 16;
    static constexpr inline Index max_local_block_count_ = 1024;
};
} // namespace oneapi::dal::decision_forest::backend
