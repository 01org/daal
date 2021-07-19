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

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#ifdef ONEDAL_DATA_PARALLEL

#include <CL/sycl/ONEAPI/experimental/builtins.hpp>

#endif

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_kernel_hist_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace be = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;
using address = sycl::access::address_space;

using sycl::ONEAPI::broadcast;
using sycl::ONEAPI::reduce;
using sycl::ONEAPI::plus;
using sycl::ONEAPI::minimum;
using sycl::ONEAPI::maximum;
using sycl::ONEAPI::exclusive_scan;

template <typename T>
using enable_if_float_t = std::enable_if_t<detail::is_valid_float_v<T>>;

template <typename Data>
using local_accessor_rw_t =
    sycl::accessor<Data, 1, sycl::access::mode::read_write, sycl::access::target::local>;

template <typename F>
struct float_accuracy;

template <>
struct float_accuracy<float> {
    static constexpr float val = float(1e-5);
};

template <>
struct float_accuracy<double> {
    static constexpr double val = double(1e-10);
};

template <typename T>
inline T atomic_global_add(T* ptr, T operand) {
    return sycl::atomic_fetch_add<T, address::global_space>(
        { sycl::multi_ptr<T, address::global_space>{ ptr } },
        operand);
}

template <typename Float, typename Bin, typename Index, typename Task>
std::uint64_t train_kernel_hist_impl<Float, Bin, Index, Task>::get_part_hist_required_mem_size(
    Index selected_ftr_count,
    Index max_bin_count_among_ftrs,
    Index hist_prop_count) const {
    // mul overflow for nSelectedFeatures * ctx_.max_bin_count_among_ftrs_ and for nHistBins * _nHistProps were checked before kernel call in compute
    return selected_ftr_count * max_bin_count_among_ftrs * hist_prop_count;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::allreduce_ndarray_inplace(
    pr::ndarray<Index, 1>& src_dst,
    const be::event_vector& deps) {
#ifdef DISTRIBUTED_SUPPORT_ENABLED

    auto src_dst_host = src_dst.to_host(queue_, deps);
    auto tgt_host = pr::ndarray<Index, 1>::empty(src_dst.get_shape());

    comm_
        .allreduce_add_int(src_dst_host.get_data(),
                           tgt_host.get_mutable_data(),
                           src_dst_host.get_count())
        .wait();

    auto last_event = src_dst.assign(queue_, tgt_host);

    last_event.wait_and_throw();

    return last_event;
#else
    return sycl::event{};
#endif
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::allreduce_ndarray_inplace(
    pr::ndarray<Float, 1>& src_dst,
    const be::event_vector& deps) {
#ifdef DISTRIBUTED_SUPPORT_ENABLED

    auto src_dst_host = src_dst.to_host(queue_, deps);
    auto tgt_host = pr::ndarray<Float, 1>::empty(src_dst.get_shape());

    comm_
        .allreduce_add_float(src_dst_host.get_data(),
                             tgt_host.get_mutable_data(),
                             src_dst_host.get_count())
        .wait();

    auto last_event = src_dst.assign(queue_, tgt_host);

    last_event.wait_and_throw();

    return last_event;
#else
    return sycl::event{};
#endif
}

template <typename Float, typename Bin, typename Index, typename Task>
void train_kernel_hist_impl<Float, Bin, Index, Task>::validate_input(const descriptor_t& desc,
                                                                     const table& data,
                                                                     const table& labels) const {
    if (data.get_row_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_rows());
    }
    if (data.get_column_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_columns());
    }
    if (desc.get_tree_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_number_of_trees());
    }
    if (desc.get_min_observations_in_leaf_node() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_number_of_min_observations_in_leaf_node());
    }
    if (desc.get_features_per_node() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_number_of_feature_per_node());
    }
    if (desc.get_max_bins() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_number_of_max_bins());
    }
    if (desc.get_min_bin_size() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_value_for_min_bin_size());
    }
    if constexpr (std::is_same_v<Task, task::classification>) {
        if (desc.get_class_count() > de::limits<Index>::max()) {
            throw domain_error(msg::invalid_number_of_classes());
        }
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
Index train_kernel_hist_impl<Float, Bin, Index, Task>::get_row_total_count(bool distr_mode,
                                                                           Index row_count) {
    Index row_total_count = row_count;

#ifdef DISTRIBUTED_SUPPORT_ENABLED
    if (distr_mode) {
        comm_.allreduce_add_int(&row_count, &row_total_count, 1).wait();
    }
#endif

    return row_total_count;
}

template <typename Float, typename Bin, typename Index, typename Task>
Index train_kernel_hist_impl<Float, Bin, Index, Task>::get_global_row_offset(bool distr_mode,
                                                                             Index row_count) {
    Index global_row_offset = 0;

#ifdef DISTRIBUTED_SUPPORT_ENABLED
    if (distr_mode) {
        auto row_count_list_host = pr::ndarray<Index, 1>::empty({ comm_.get_rank_count() });
        Index* row_count_list_host_ptr = row_count_list_host.get_mutable_data();
        comm_
            .allgather(reinterpret_cast<byte_t*>(&row_count),
                       sizeof(Index),
                       reinterpret_cast<byte_t*>(row_count_list_host_ptr),
                       sizeof(Index))
            .wait();

        for (std::int64_t i = 0; i < comm_.get_rank(); i++) {
            global_row_offset += row_count_list_host_ptr[i];
        }
    }
#endif

    return global_row_offset;
}

template <typename Float, typename Bin, typename Index, typename Task>
void train_kernel_hist_impl<Float, Bin, Index, Task>::init_params(context_t& ctx,
                                                                  const descriptor_t& desc,
                                                                  const table& data,
                                                                  const table& responses) {
#ifdef DISTRIBUTED_SUPPORT_ENABLED
    ctx.distr_mode_ = (comm_.get_rank_count() > 1);
#endif

    if constexpr (std::is_same_v<Task, task::classification>) {
        ctx.class_count_ = de::integral_cast<Index>(desc.get_class_count());
    }
    ctx.row_count_ = de::integral_cast<Index>(data.get_row_count());
    ctx.row_total_count_ = get_row_total_count(ctx.distr_mode_, ctx.row_count_);

    ctx.column_count_ = de::integral_cast<Index>(data.get_column_count());

    // in case of distributed mode selected_row_count is defined during initial gen of tree order
    ctx.selected_row_count_ = ctx.distr_mode_
                                  ? impl_const_t::bad_val_
                                  : desc.get_observations_per_tree_fraction() * ctx.row_count_;
    ctx.selected_row_total_count_ =
        desc.get_observations_per_tree_fraction() * ctx.row_total_count_;

    ctx.global_row_offset_ = get_global_row_offset(ctx.distr_mode_, ctx.row_count_);

    ctx.tree_count_ = de::integral_cast<Index>(desc.get_tree_count());

    ctx.bootstrap_ = desc.get_bootstrap();
    ctx.max_tree_depth_ = desc.get_max_tree_depth();

    if constexpr (std::is_same_v<Task, task::classification>) {
        ctx.selected_ftr_count_ = desc.get_features_per_node() ? desc.get_features_per_node()
                                                               : std::sqrt(ctx.column_count_);
    }
    else {
        ctx.selected_ftr_count_ = desc.get_features_per_node()
                                      ? desc.get_features_per_node()
                                      : ctx.column_count_ / 3 ? ctx.column_count_ / 3 : 1;
    }
    ctx.min_observations_in_leaf_node_ = desc.get_min_observations_in_leaf_node();
    ctx.impurity_threshold_ = desc.get_impurity_threshold();

    ctx.min_observations_in_leaf_node_ = desc.get_min_observations_in_leaf_node();
    ctx.impurity_threshold_ = desc.get_impurity_threshold();

    if (0 >= ctx.selected_row_total_count_) {
        throw domain_error(msg::invalid_value_for_observations_per_tree_fraction());
    }

    ctx.preferable_local_size_for_part_hist_kernel_ = ctx.preferable_group_size_;
    while (ctx.preferable_local_size_for_part_hist_kernel_ >
           std::max(ctx.selected_ftr_count_, ctx.min_preferable_local_size_for_part_hist_kernel_)) {
        ctx.preferable_local_size_for_part_hist_kernel_ >>= 1;
    }

    auto vimp = desc.get_variable_importance_mode();
    ctx.mda_required_ =
        (variable_importance_mode::mda_raw == vimp || variable_importance_mode::mda_scaled == vimp);
    ctx.mdi_required_ = variable_importance_mode::mdi == vimp;
    ctx.mda_scaled_required_ = (variable_importance_mode::mda_scaled == vimp);

    auto emm = desc.get_error_metric_mode();
    ctx.oob_required_ =
        (check_mask_flag(emm, error_metric_mode::out_of_bag_error) ||
         check_mask_flag(emm, error_metric_mode::out_of_bag_error_per_observation) ||
         ctx.mda_required_);
    ctx.oob_err_required_ = check_mask_flag(emm, error_metric_mode::out_of_bag_error);
    ctx.oob_err_obs_required_ =
        check_mask_flag(emm, error_metric_mode::out_of_bag_error_per_observation);

    // init ftr -> bins map and related params
#ifdef DISTRIBUTED_SUPPORT_ENABLED
    indexed_features<Float, Bin, Index> ind_ftrs(queue_,
                                                 comm_,
                                                 desc.get_min_bin_size(),
                                                 desc.get_max_bins());
#else
    indexed_features<Float, Bin, Index> ind_ftrs(queue_,
                                                 desc.get_min_bin_size(),
                                                 desc.get_max_bins());
#endif
    { ind_ftrs(data).wait_and_throw(); }

    ctx.total_bin_count_ = ind_ftrs.get_total_bin_count();
    full_data_nd_ = ind_ftrs.get_full_data();
    ftr_bin_offsets_nd_ = ind_ftrs.get_bin_offsets();

    bin_borders_host_.resize(ctx.column_count_);
    for (Index clmn_idx = 0; clmn_idx < ctx.column_count_; clmn_idx++) {
        bin_borders_host_[clmn_idx] = ind_ftrs.get_bin_borders(clmn_idx).to_host(queue_);
    }

    data_host_ = pr::table2ndarray_1d<Float>(queue_, data, alloc::device).to_host(queue_);

    response_nd_ = pr::table2ndarray_1d<Float>(queue_, responses, alloc::device);

    response_host_ = response_nd_.to_host(queue_);

    // calculating the maximal number of bins for feature among all features
    ctx.max_bin_count_among_ftrs_ = 0;
    for (Index clmn_idx = 0; clmn_idx < ctx.column_count_; clmn_idx++) {
        auto ftr_bins = ind_ftrs.get_bin_count(clmn_idx);
        ctx.max_bin_count_among_ftrs_ = std::max(ctx.max_bin_count_among_ftrs_, ftr_bins);
    }

    ctx.float_min_ = de::limits<Float>::min();
    ctx.index_max_ = de::limits<Index>::max();

    // define number of trees which can be built in parallel
    const std::uint64_t device_global_mem_size =
        queue_.get_device().get_info<sycl::info::device::global_mem_size>();
    const std::uint64_t device_max_mem_alloc_size =
        queue_.get_device().get_info<sycl::info::device::max_mem_alloc_size>();

    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<task::classification, Task>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const_t::hist_prop_count_;
    }
    const auto part_hist_size = get_part_hist_required_mem_size(ctx.selected_ftr_count_,
                                                                ctx.max_bin_count_among_ftrs_,
                                                                hist_prop_count);
    const auto max_mem_alloc_size =
        std::min(device_max_mem_alloc_size, std::uint64_t(ctx.max_mem_alloc_size_for_algo_));

    std::uint64_t used_mem_size =
        sizeof(Float) * ctx.row_count_ * (ctx.column_count_ + 1); // input table size + response
    used_mem_size +=
        ind_ftrs.get_required_mem_size(ctx.row_count_, ctx.column_count_, desc.get_max_bins());
    used_mem_size += ctx.oob_required_ ? sizeof(Float) * ctx.row_count_ * ctx.class_count_ : 0;
    used_mem_size += part_hist_size; // space at least for one part hist

    std::uint64_t available_global_mem_size =
        device_global_mem_size > used_mem_size ? device_global_mem_size - used_mem_size : 0;

    std::uint64_t available_mem_size_for_tree_block =
        std::min(max_mem_alloc_size,
                 static_cast<std::uint64_t>(available_global_mem_size *
                                            ctx.global_mem_fraction_for_tree_block_));

    std::uint64_t required_mem_size_for_one_tree =
        ctx.oob_required_ ? train_service_kernels_.get_oob_rows_required_mem_size(
                                ctx.row_count_,
                                1 /* for 1 tree */,
                                desc.get_observations_per_tree_fraction())
                          : 0;

    // TODO : figure out the universal formula for multi and single run
    required_mem_size_for_one_tree += sizeof(Index) * ctx.selected_row_total_count_ * 2;

    ctx.tree_in_block_ = de::integral_cast<Index>(available_mem_size_for_tree_block /
                                                  required_mem_size_for_one_tree);

    if (ctx.tree_in_block_ <= 0) {
        // not enough memory even for one tree
        throw domain_error(msg::not_enough_memory_to_build_one_tree());
    }

    ctx.tree_in_block_ = std::min(ctx.tree_count_, ctx.tree_in_block_);

    available_global_mem_size =
        available_global_mem_size > (ctx.tree_in_block_ * required_mem_size_for_one_tree)
            ? available_global_mem_size - (ctx.tree_in_block_ * required_mem_size_for_one_tree)
            : 0;
    // size for one part hist was already reserved, add some more if there is available mem
    ctx.max_part_hist_cumulative_size_ = std::min(
        max_mem_alloc_size,
        static_cast<std::uint64_t>(part_hist_size + available_global_mem_size *
                                                        ctx.global_mem_fraction_for_part_hist_));

    ctx.oob_prop_count_ = impl_const_t::oob_aux_prop_count_;
    if constexpr (std::is_same_v<Task, task::classification>) {
        ctx.oob_prop_count_ = ctx.class_count_;
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
void train_kernel_hist_impl<Float, Bin, Index, Task>::allocate_buffers(const context_t& ctx) {
    de::check_mul_overflow(ctx.selected_row_total_count_, ctx.tree_in_block_);
    selected_row_global_host_ =
        pr::ndarray<Index, 1>::empty({ ctx.selected_row_total_count_ * ctx.tree_in_block_ });
    selected_row_host_ =
        pr::ndarray<Index, 1>::empty({ ctx.selected_row_total_count_ * ctx.tree_in_block_ });

    // main tree order and auxilliary one are used for partitioning
    tree_order_lev_ =
        pr::ndarray<Index, 1>::empty(queue_,
                                     { ctx.selected_row_total_count_ * ctx.tree_in_block_ },
                                     alloc::device);
    tree_order_lev_buf_ =
        pr::ndarray<Index, 1>::empty(queue_,
                                     { ctx.selected_row_total_count_ * ctx.tree_in_block_ },
                                     alloc::device);
    if (ctx.oob_required_) {
        // oob_per_obs_list contains class_count number of counters for all out of bag observations for all trees
        de::check_mul_overflow(ctx.row_count_, ctx.class_count_);
        auto [oob_per_obs_list, event] =
            pr::ndarray<hist_type_t, 1>::zeros(queue_,
                                               { ctx.row_count_ * ctx.oob_prop_count_ },
                                               alloc::device);
        oob_per_obs_list_ = oob_per_obs_list;
        event.wait_and_throw();
    }

    // blocks for MDA scaled error calculation
    if (ctx.mda_scaled_required_) {
        var_imp_variance_host_ = pr::ndarray<Float, 1>::zeros({ ctx.column_count_ });
    }

    if (ctx.mdi_required_ || ctx.mda_required_) {
        res_var_imp_ = pr::ndarray<Float, 1>::empty(queue_, { ctx.column_count_ }, alloc::device);
        res_var_imp_.fill(queue_, 0); // addd deps??
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::gen_initial_tree_order(
    context_t& ctx,
    engine_list_t& engine_list,
    pr::ndarray<Index, 1>& node_list,
    pr::ndarray<Index, 1>& tree_order_level,
    pr::ndarray<Index, 1>& selected_row_global_host,
    pr::ndarray<Index, 1>& selected_row_host,
    Index engine_offset,
    Index node_count) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(tree_order_level.get_count() ==
                  ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(selected_row_global_host.get_count() ==
                  ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(selected_row_host.get_count() ==
                  ctx.tree_in_block_ * ctx.selected_row_total_count_);

    engine_impl* engines = engine_list.get_mutable_data();
    Index* selected_row_global_ptr = selected_row_global_host.get_mutable_data();
    Index* selected_row_ptr = selected_row_host.get_mutable_data();
    Index* node_list_ptr = node_list.get_mutable_data();

    for (Index node_idx = 0; node_idx < node_count; node_idx++) {
        rng<Index> rn_gen;
        Index* gen_row_idx_global_ptr =
            selected_row_global_ptr + ctx.selected_row_total_count_ * node_idx;
        rn_gen.uniform(ctx.selected_row_total_count_,
                       gen_row_idx_global_ptr,
                       engines[engine_offset + node_idx].get_state(),
                       0,
                       ctx.row_total_count_);

        if (ctx.distr_mode_) {
            Index* node_ptr = node_list_ptr + node_idx * impl_const_t::node_prop_count_;
            Index* src = gen_row_idx_global_ptr;

            Index* dst = selected_row_ptr + ctx.selected_row_total_count_ * node_idx;

            Index row_idx = 0;
            for (Index i = 0; i < ctx.selected_row_total_count_; i++) {
                dst[i] = 0;
                if (src[i] >= ctx.global_row_offset_ &&
                    src[i] < ctx.global_row_offset_ + ctx.row_count_) {
                    dst[row_idx++] = src[i] - ctx.global_row_offset_;
                }
            }
            node_ptr[impl_const_t::ind_lrc] = row_idx;
        }
    }

    sycl::event event = ctx.distr_mode_ ? tree_order_level.assign(queue_, selected_row_host)
                                        : tree_order_level.assign(queue_, selected_row_global_host);
    event.wait_and_throw();

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
std::tuple<pr::ndarray<Index, 1>, sycl::event>
train_kernel_hist_impl<Float, Bin, Index, Task>::gen_feature_list(
    Index node_count,
    const pr::ndarray<Index, 1>& node_vs_tree_map_list,
    engine_list_t& engine_list,
    const context_t& ctx) {
    de::check_mul_overflow((node_count + 1), ctx.selected_ftr_count_);
    // first part is used for features indices, +1 block - part for generator
    auto selected_features_host =
        pr::ndarray<Index, 1>::empty({ (node_count + 1) * ctx.selected_ftr_count_ });
    auto selected_features_com =
        pr::ndarray<Index, 1>::empty(queue_,
                                     { node_count * ctx.selected_ftr_count_ },
                                     alloc::device);

    auto selected_features_host_ptr = selected_features_host.get_mutable_data();

    auto node_vs_tree_map_list_host = node_vs_tree_map_list.to_host(queue_);

    engine_impl* engines = engine_list.get_mutable_data();

    if (ctx.selected_ftr_count_ != ctx.column_count_) {
        rng<Index> rn_gen;
        auto treeMap_ptr = node_vs_tree_map_list_host.get_mutable_data();

        for (Index node = 0; node < node_count; node++) {
            rn_gen.uniform_without_replacement(
                ctx.selected_ftr_count_,
                selected_features_host_ptr + node * ctx.selected_ftr_count_,
                selected_features_host_ptr + (node + 1) * ctx.selected_ftr_count_,
                engines[treeMap_ptr[node]].get_state(),
                0,
                ctx.column_count_);
        }
    }
    else {
        for (Index node = 0; node < node_count; node++) {
            for (Index i = 0; i < ctx.selected_ftr_count_; i++) {
                selected_features_host_ptr[node * ctx.selected_ftr_count_ + i] = i;
            }
        }
    }

    auto event = selected_features_com.assign(queue_,
                                              selected_features_host_ptr,
                                              selected_features_com.get_count());

    return std::tuple{ selected_features_com, event };
}

// Kernel context
template <typename Float, typename Index, typename Task>
struct kernel_context {
    kernel_context(const train_context<Float, Index, Task>& ctx)
            : float_min_(ctx.float_min_),
              index_max_(ctx.index_max_),
              min_observations_in_leaf_node_(ctx.min_observations_in_leaf_node_),
              class_count_(ctx.class_count_),
              impurity_threshold_(ctx.impurity_threshold_) {}

    Float float_min_;
    Index index_max_;
    Index min_observations_in_leaf_node_;
    Index class_count_;
    Float impurity_threshold_;
};

template <typename Float, typename Index>
inline void add_val_to_hist(
    typename task_types<Float, Index, task::classification>::hist_type_t* hist_ptr,
    Float val) {
    Index class_id = static_cast<Index>(val);
    hist_ptr[class_id] += 1;
}

template <typename Float, typename Index>
inline void add_val_to_hist(
    typename task_types<Float, Index, task::regression>::hist_type_t* hist_ptr,
    Float val) {
    hist_ptr[0] += Float(1);
    Float inv_n = Float(1) / hist_ptr[0];
    Float delta = val - hist_ptr[1]; // y[i] - mean
    hist_ptr[1] += delta * inv_n; // updated mean
    hist_ptr[2] += delta * (val - hist_ptr[1]); // updated sum2Cent
}

// merge src and dst stat
template <typename Float, typename Index, typename T = enable_if_float_t<Float>>
inline void merge_stat(Float* dst, const Float* src, Index) {
    if (Float(0) == src[0])
        return;

    Float sum_n1n2 = dst[0] + src[0];
    Float mul_n1n2 = dst[0] * src[0];
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float mean_scl = (Float)1 / sum_n1n2;
    Float delta = src[1] - dst[1];

    dst[2] = dst[2] + src[2] + delta * delta * delta_scl;
    dst[1] = (dst[1] * dst[0] + src[1] * src[0]) * mean_scl;
    dst[0] = sum_n1n2;
}

template <typename Index>
inline void merge_stat(Index* dst, const Index* src, Index count) {
    for (Index i = 0; i < count; i++) {
        dst[i] += src[i];
    }
}

template <typename Index>
inline void merge_stat(Index* dst, Index* accum, const Index* src, Index count) {
    for (Index i = 0; i < count; i++) {
        *accum += src[i];
        dst[i] += src[i];
    }
}

template <typename Float, typename Index, typename T = enable_if_float_t<Float>>
inline void sub_stat(Float* dst, const Float* src, const Float* mrg, Index elem_count) {
    dst[0] = mrg[0] - src[0];

    dst[1] = dst[0] >= Float(1) ? mrg[1] + (src[0] * (mrg[1] - src[1])) / dst[0] : Float(0);

    Float sum_n1n2 = mrg[0];
    Float mul_n1n2 = src[0] * dst[0];
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float delta = src[1] - dst[1];

    dst[2] = dst[0] >= Float(1) ? (mrg[2] - src[2] - delta * delta * delta_scl) : Float(0);
}

// classification compute_hist_for_node
template <typename Float, typename Index>
inline void compute_hist_for_node(
    sycl::nd_item<2> item,
    Index ind_start,
    Index ind_end,
    const Float* response_ptr,
    Index* node_ptr,
    const Index* node_tree_order_ptr,
    typename task_types<Float, Index, task::classification>::hist_type_t*
        local_buf_ptr, /* for unification */
    const imp_data_list_ptr_mutable<Float, Index, task::classification>& imp_list_ptr,
    const kernel_context<Float, Index, task::classification>& ctx,
    Index node_id) {
    using task_t = task::classification;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, task_t>::hist_type_t;

    constexpr Index buff_size = impl_const_t::max_private_class_hist_buff_size;
    hist_type_t private_histogram[buff_size] = { 0 };
    Index* node_histogram_ptr = imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;
    Float* node_imp_ptr = imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
    const Index row_count = node_ptr[impl_const_t::ind_lrc];

    for (Index i = ind_start; i < ind_end; i++) {
        Index id = node_tree_order_ptr[i];
        add_val_to_hist<Float, Index>(private_histogram, response_ptr[id]);
    }

    for (Index cls_idx = 0; cls_idx < ctx.class_count_; cls_idx++) {
        atomic_global_add(node_histogram_ptr + cls_idx, private_histogram[cls_idx]);
    }

    item.barrier(sycl::access::fence_space::local_space);

    Float imp = Float(1);
    Float div = Float(1) / (Float(row_count) * row_count);
    Index max_cls_count = 0;
    Index win_cls = 0;
    Index cls_count = 0;

    for (Index cls_idx = 0; cls_idx < ctx.class_count_; cls_idx++) {
        cls_count = node_histogram_ptr[cls_idx];
        imp -= Float(cls_count) * (cls_count)*div;

        if (cls_count > max_cls_count) {
            max_cls_count = cls_count;
            win_cls = cls_idx;
        }
    }

    node_ptr[5] = win_cls;
    node_imp_ptr[0] = sycl::max(imp, Float(0));
}

// regression compute_hist_for_node
template <typename Float, typename Index>
inline void compute_hist_for_node(
    sycl::nd_item<2> item,
    Index ind_start,
    Index ind_end,
    const Float* response_ptr,
    Index* node_ptr,
    const Index* node_tree_order_ptr,
    typename task_types<Float, Index, task::regression>::hist_type_t* local_buf_ptr,
    const imp_data_list_ptr_mutable<Float, Index, task::regression>& imp_list_ptr,
    const kernel_context<Float, Index, task::regression>& ctx,
    Index node_id) {
    using task_t = task::regression;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, task_t>::hist_type_t;

    const Index local_id = item.get_local_id()[0];
    const Index local_size = item.get_local_range()[0];

    Float* node_imp_ptr = imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
    constexpr Index hist_prop_count = impl_const_t::hist_prop_count_;
    hist_type_t private_histogram[hist_prop_count] = { 0 };

    for (Index i = ind_start; i < ind_end; i++) {
        Index id = node_tree_order_ptr[i];
        add_val_to_hist<Float, Index>(private_histogram, response_ptr[id]);
    }

    hist_type_t* local_h_ptr = local_buf_ptr + local_id * hist_prop_count;

    local_h_ptr[0] = private_histogram[0];
    local_h_ptr[1] = private_histogram[1];
    local_h_ptr[2] = private_histogram[2];

    for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
        item.barrier(sycl::access::fence_space::local_space);
        if (local_id < offset) {
            hist_type_t* h_ptr = local_buf_ptr + (local_id + offset) * hist_prop_count;
            merge_stat(local_h_ptr, h_ptr, hist_prop_count);
        }
    }

    if (local_id == 0) {
        node_imp_ptr[0] = local_h_ptr[1]; // store mean
        node_imp_ptr[1] = local_h_ptr[2]; // store sum2cent
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_imp_for_node_list(
    const context_t& ctx,
    imp_data_t& imp_data_list,
    pr::ndarray<Index, 1>& node_list,
    Index node_count,
    const be::event_vector& deps) {
    sycl::event last_event;

    if constexpr (std::is_same_v<task::classification, Task>) {
        auto class_hist_list_host = imp_data_list.class_hist_list_.to_host(queue_, deps);
        auto imp_list_host = imp_data_list.imp_list_.to_host(queue_);
        auto node_list_host = node_list.to_host(queue_);

        const Index* class_hist_list_host_ptr = class_hist_list_host.get_data();
        Float* imp_list_host_ptr = imp_list_host.get_mutable_data();
        Index* node_list_host_ptr = node_list_host.get_mutable_data();

        for (Index node_idx = 0; node_idx < node_count; node_idx++) {
            const Index* node_histogram_ptr =
                class_hist_list_host_ptr + node_idx * ctx.class_count_;
            Float* node_imp_ptr = imp_list_host_ptr + node_idx * impl_const_t::node_imp_prop_count_;
            Index* node_ptr = node_list_host_ptr + node_idx * impl_const_t::node_prop_count_;

            Index row_count = node_ptr[impl_const_t::ind_grc];

            Float imp = Float(1);
            Float div = Float(1) / (Float(row_count) * row_count);
            Index max_cls_count = 0;
            Index win_cls = 0;
            Index cls_count = 0;

            for (Index cls_idx = 0; cls_idx < ctx.class_count_; cls_idx++) {
                cls_count = node_histogram_ptr[cls_idx];
                imp -= Float(cls_count) * (cls_count)*div;

                if (cls_count > max_cls_count) {
                    max_cls_count = cls_count;
                    win_cls = cls_idx;
                }
            }

            node_ptr[impl_const_t::ind_win] = win_cls;
            node_imp_ptr[0] = cl::sycl::max(imp, Float(0));
        }
        imp_data_list.imp_list_.assign(queue_, imp_list_host_ptr, imp_list_host.get_count())
            .wait_and_throw();
        node_list.assign(queue_, node_list_host_ptr, node_list_host.get_count()).wait_and_throw();
    }

    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_histogram_local(
    const context_t& ctx,
    const pr::ndarray<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    pr::ndarray<Index, 1>& node_list,
    imp_data_t& imp_data_list,
    Index node_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() ==
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() == node_count * ctx.class_count_);
    }

    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = tree_order.get_data();
    Index* node_list_ptr = node_list.get_mutable_data();

    // num of split attributes for node
    const Index node_prop_count = impl_const_t::node_prop_count_;
    // num of impurity attributes for node
    const Index node_imp_prop_count = impl_const_t::node_imp_prop_count_;

    sycl::event fill_event = {};
    if constexpr (std::is_same_v<Task, task::classification>) {
        fill_event = imp_data_list.class_hist_list_.fill(queue_, 0, deps);
    }

    imp_data_list_ptr_mutable<Float, Index, Task> imp_list_ptr(imp_data_list);
    const kernel_context<Float, Index, Task> krn_ctx(ctx);

    auto local_size = ctx.preferable_group_size_;
    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.depends_on(fill_event);
        // local_buf is used for regression only, but need to be present for classification also
        local_accessor_rw_t<hist_type_t> local_buf(local_size * (node_imp_prop_count + 1), cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            const Index node_id = item.get_global_id()[1];
            const Index local_id = item.get_local_id()[0];
            const Index local_size = item.get_local_range()[0];

            Index* node_ptr = node_list_ptr + node_id * node_prop_count;

            const Index row_offset = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];

            const Index elem_count = row_count / local_size + bool(row_count % local_size);

            const Index ind_start = local_id * elem_count;
            const Index ind_end =
                sycl::min(static_cast<Index>((local_id + 1) * elem_count), row_count);

            const Index* node_tree_order_ptr = &tree_order_ptr[row_offset];

            hist_type_t* local_buf_ptr = nullptr;
            if constexpr (std::is_same_v<Task, task::regression>) {
                local_buf_ptr = local_buf.get_pointer().get();
            }
            compute_hist_for_node<Float, Index>(item,
                                                ind_start,
                                                ind_end,
                                                response_ptr,
                                                node_ptr,
                                                node_tree_order_ptr,
                                                local_buf_ptr,
                                                imp_list_ptr,
                                                krn_ctx,
                                                node_id);
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_sum_local(
    const context_t& ctx,
    const pr::ndarray<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    const pr::ndarray<Index, 1>& node_list,
    pr::ndarray<Float, 1>& sum_list,
    Index node_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(sum_list.get_count() == node_count);

    auto fill_event = sum_list.fill(queue_, 0, deps);
    fill_event.wait_and_throw();

    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = tree_order.get_data();
    const Index* node_list_ptr = node_list.get_data();
    Float* sum_list_ptr = sum_list.get_mutable_data();

    // num of attributes for node
    const Index node_prop_count = impl_const_t::node_prop_count_;

    auto local_size = ctx.preferable_group_size_;
    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<Float> local_buf(local_size, cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            const Index node_id = item.get_global_id()[1];
            const Index local_id = item.get_local_id()[0];
            const Index local_size = item.get_local_range()[0];

            const Index* node_ptr = node_list_ptr + node_id * node_prop_count;

            const Index row_offset = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];

            const Index* node_tree_order_ptr = &tree_order_ptr[row_offset];

            Float* local_buf_ptr = local_buf.get_pointer().get();

            Float sum = Float(0);
            for (Index i = local_id; i < row_count; i += local_size) {
                sum += response_ptr[node_tree_order_ptr[i]];
            }

            local_buf_ptr[local_id] = sum;

            for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if (local_id < offset) {
                    local_buf_ptr[local_id] += local_buf_ptr[local_id + offset];
                }
            }

            if (local_id == 0) {
                sum_list_ptr[node_id] = local_buf_ptr[local_id];
            }
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_sum2cent_local(
    const context_t& ctx,
    const pr::ndarray<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Float, 1>& sum_list,
    pr::ndarray<Float, 1>& sum2cent_list,
    Index node_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(sum_list.get_count() == node_count);

    auto fill_event = sum2cent_list.fill(queue_, 0, deps);
    fill_event.wait_and_throw();

    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = tree_order.get_data();
    const Index* node_list_ptr = node_list.get_data();
    const Float* sum_list_ptr = sum_list.get_data();
    Float* sum2cent_list_ptr = sum2cent_list.get_mutable_data();

    // num of attributes for node
    const Index node_prop_count = impl_const_t::node_prop_count_;

    auto local_size = ctx.preferable_group_size_;
    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<Float> local_buf(local_size, cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            const Index node_id = item.get_global_id()[1];
            const Index local_id = item.get_local_id()[0];
            const Index local_size = item.get_local_range()[0];

            const Index* node_ptr = node_list_ptr + node_id * node_prop_count;

            const Index row_offset = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];
            const Index global_row_count = node_ptr[impl_const_t::ind_grc];

            const Index* node_tree_order_ptr = &tree_order_ptr[row_offset];

            const Float mean = sum_list_ptr[node_id] / global_row_count;

            Float* local_buf_ptr = local_buf.get_pointer().get();

            Float sum2cent = Float(0);
            for (Index i = local_id; i < row_count; i += local_size) {
                sum2cent += (response_ptr[node_tree_order_ptr[i]] - mean) *
                            (response_ptr[node_tree_order_ptr[i]] - mean);
            }

            local_buf_ptr[local_id] = sum2cent;

            for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if (local_id < offset) {
                    local_buf_ptr[local_id] += local_buf_ptr[local_id + offset];
                }
            }

            if (local_id == 0) {
                sum2cent_list_ptr[node_id] = local_buf_ptr[local_id];
            }
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::fin_initial_imp(
    const context_t& ctx,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Float, 1>& sum_list,
    const pr::ndarray<Float, 1>& sum2cent_list,
    imp_data_t& imp_data_list,
    Index node_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(sum_list.get_count() == node_count);
    ONEDAL_ASSERT(sum2cent_list.get_count() == node_count);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() ==
                  node_count * impl_const_t::node_imp_prop_count_);

    const Index* node_list_ptr = node_list.get_data();
    const Float* sum_list_ptr = sum_list.get_data();
    const Float* sum2cent_list_ptr = sum2cent_list.get_data();
    Float* imp_list_ptr = imp_data_list.imp_list_.get_mutable_data();

    const sycl::range<1> range{ de::integral_cast<size_t>(node_count) };

    auto last_event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> node_idx) {
            // set mean
            // node grc can't be 0 due to this is initial computation on whole ds block
            imp_list_ptr[node_idx * impl_const_t::node_imp_prop_count_ + 0] =
                sum_list_ptr[node_idx] /
                node_list_ptr[node_idx * impl_const_t::node_prop_count_ + impl_const_t::ind_grc];
            // set sum2cent
            imp_list_ptr[node_idx * impl_const_t::node_imp_prop_count_ + 1] =
                sum2cent_list_ptr[node_idx];
        });
    });

    last_event.wait_and_throw();

    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_histogram(
    const context_t& ctx,
    const pr::ndarray<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    pr::ndarray<Index, 1>& node_list,
    imp_data_t& imp_data_list,
    Index node_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() ==
                  node_count * impl_const_t::node_imp_prop_count_);

    sycl::event last_event;

    if (ctx.distr_mode_) {
        if constexpr (std::is_same_v<Task, task::classification>) {
            ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() ==
                          node_count * ctx.class_count_);

            last_event = compute_initial_histogram_local(ctx,
                                                         response,
                                                         tree_order,
                                                         node_list,
                                                         imp_data_list,
                                                         node_count,
                                                         deps);
            last_event = allreduce_ndarray_inplace(imp_data_list.class_hist_list_, { last_event });
            last_event = compute_initial_imp_for_node_list(ctx,
                                                           imp_data_list,
                                                           node_list,
                                                           node_count,
                                                           { last_event });
        }
        else {
            auto sum_list = pr::ndarray<Float, 1>::empty(queue_, { node_count });
            auto sum2cent_list = pr::ndarray<Float, 1>::empty(queue_, { node_count });
            last_event = compute_initial_sum_local(ctx,
                                                   response,
                                                   tree_order,
                                                   node_list,
                                                   sum_list,
                                                   node_count,
                                                   deps);
            last_event = allreduce_ndarray_inplace(sum_list, { last_event });
            last_event = compute_initial_sum2cent_local(ctx,
                                                        response,
                                                        tree_order,
                                                        node_list,
                                                        sum_list,
                                                        sum2cent_list,
                                                        node_count,
                                                        { last_event });
            last_event = allreduce_ndarray_inplace(sum2cent_list, { last_event });
            last_event = fin_initial_imp(ctx,
                                         node_list,
                                         sum_list,
                                         sum2cent_list,
                                         imp_data_list,
                                         node_count,
                                         { last_event });
        }
    }
    else {
        last_event = compute_initial_histogram_local(ctx,
                                                     response,
                                                     tree_order,
                                                     node_list,
                                                     imp_data_list,
                                                     node_count,
                                                     deps);
    }

    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_best_split(
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const imp_data_t& imp_data_list,
    pr::ndarray<Index, 1>& node_list,
    imp_data_t& left_child_imp_data_list,
    pr::ndarray<Float, 1>& nodeImpDecreaseList,
    bool updateImpDecreaseRequired,
    Index node_count,
    const context_t& ctx,
    const be::event_vector& deps) {
    // no overflow check is required because of ctx.node_group_count_ and ctx.node_group_prop_count_ are small constants

    auto nodesGroups =
        pr::ndarray<Index, 1>::empty(queue_,
                                     { ctx.node_group_count_ * ctx.node_group_prop_count_ },
                                     alloc::device);
    auto nodeIndices = pr::ndarray<Index, 1>::empty(queue_, { node_count }, alloc::device);

    sycl::event last_event;
    last_event =
        train_service_kernels_.split_node_list_on_groups_by_size(ctx,
                                                                 node_list,
                                                                 nodesGroups,
                                                                 nodeIndices,
                                                                 node_count,
                                                                 ctx.node_group_count_,
                                                                 ctx.node_group_prop_count_,
                                                                 deps);
    last_event.wait_and_throw();

    auto nodesGroupsHost_nd = nodesGroups.to_host(queue_, { last_event });
    const Index* nodesGroupsHost = nodesGroupsHost_nd.get_data();

    Index nGroupNodes = 0;
    Index processedNodes = 0;

    for (Index i = 0; i < ctx.node_group_count_; i++, processedNodes += nGroupNodes) {
        nGroupNodes = nodesGroupsHost[i * ctx.node_group_prop_count_ + 0];
        if (0 == nGroupNodes)
            continue;

        Index maxGroupBlocksNum = nodesGroupsHost[i * ctx.node_group_prop_count_ + 1];

        Index groupIndicesOffset = processedNodes;

        if (maxGroupBlocksNum > 1 || ctx.distr_mode_) {
            Index hist_prop_count = 0;
            if constexpr (std::is_same_v<task::classification, Task>) {
                hist_prop_count = ctx.class_count_;
            }
            else {
                hist_prop_count = impl_const_t::hist_prop_count_;
            }

            const Index partHistSize =
                get_part_hist_required_mem_size(ctx.selected_ftr_count_,
                                                ctx.max_bin_count_among_ftrs_,
                                                hist_prop_count);

            Index nPartialHistograms = maxGroupBlocksNum <= ctx.min_row_block_count_for_one_hist_
                                           ? 1
                                           : ctx.max_part_hist_count_;

            if (nPartialHistograms > 1 &&
                maxGroupBlocksNum < ctx.min_row_block_count_to_use_max_part_hist_count_) {
                while (nPartialHistograms > 1 &&
                       (nPartialHistograms * ctx.min_row_block_count_for_one_hist_ >
                            maxGroupBlocksNum ||
                        nPartialHistograms * partHistSize > ctx.max_part_hist_cumulative_size_)) {
                    nPartialHistograms >>= 1;
                }
            }

            de::check_mul_overflow(nGroupNodes, partHistSize);
            de::check_mul_overflow(nGroupNodes * partHistSize, nPartialHistograms);

            // TODO check sizeof(Float) -> sizeof(hist_type_t)
            const Index maxPHBlockElems = ctx.max_part_hist_cumulative_size_ / sizeof(Float);

            const Index nPHBlockElems = nGroupNodes * nPartialHistograms * partHistSize;
            const Index nPHBlocks =
                nPHBlockElems / maxPHBlockElems
                    ? (nPHBlockElems / maxPHBlockElems + bool(nPHBlockElems % maxPHBlockElems))
                    : 1;

            Index nBlockNodes = nGroupNodes / nPHBlocks + bool(nGroupNodes % nPHBlocks);

            for (Index blockIndicesOffset = groupIndicesOffset;
                 blockIndicesOffset < groupIndicesOffset + nGroupNodes;
                 blockIndicesOffset += nBlockNodes) {
                nBlockNodes =
                    std::min(nBlockNodes, groupIndicesOffset + nGroupNodes - blockIndicesOffset);

                auto [nodesHistograms, event] = ctx.distr_mode_
                                                    ? compute_histogram_distr(ctx,
                                                                              data,
                                                                              response,
                                                                              treeOrder,
                                                                              selectedFeatures,
                                                                              binOffsets,
                                                                              node_list,
                                                                              nodeIndices,
                                                                              blockIndicesOffset,
                                                                              nPartialHistograms,
                                                                              nBlockNodes,
                                                                              { last_event })

                                                    : compute_histogram(ctx,
                                                                        data,
                                                                        response,
                                                                        treeOrder,
                                                                        selectedFeatures,
                                                                        binOffsets,
                                                                        node_list,
                                                                        nodeIndices,
                                                                        blockIndicesOffset,
                                                                        nPartialHistograms,
                                                                        nBlockNodes,
                                                                        { last_event });
                last_event = event;

                {
                    last_event =
                        bs_kernels_.compute_best_split_by_histogram(ctx,
                                                                    nodesHistograms,
                                                                    selectedFeatures,
                                                                    binOffsets,
                                                                    imp_data_list,
                                                                    nodeIndices,
                                                                    blockIndicesOffset,
                                                                    node_list,
                                                                    left_child_imp_data_list,
                                                                    nodeImpDecreaseList,
                                                                    updateImpDecreaseRequired,
                                                                    nBlockNodes,
                                                                    { last_event });
                    last_event.wait_and_throw();
                }
            }
        }
        else {
            last_event = bs_kernels_.compute_best_split_single_pass(ctx,
                                                                    data,
                                                                    response,
                                                                    treeOrder,
                                                                    selectedFeatures,
                                                                    binOffsets,
                                                                    imp_data_list,
                                                                    nodeIndices,
                                                                    groupIndicesOffset,
                                                                    node_list,
                                                                    left_child_imp_data_list,
                                                                    nodeImpDecreaseList,
                                                                    updateImpDecreaseRequired,
                                                                    nGroupNodes,
                                                                    { last_event });
            last_event.wait_and_throw();
        }
    }

    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task>
std::tuple<pr::ndarray<typename task_types<Float, Index, Task>::hist_type_t, 1>, sycl::event>
train_kernel_hist_impl<Float, Bin, Index, Task>::compute_histogram(
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    Index nPartialHistograms,
    Index node_count,
    const be::event_vector& deps) {
    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<task::classification, Task>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const_t::hist_prop_count_;
    }

    const Index partHistSize = get_part_hist_required_mem_size(ctx.selected_ftr_count_,
                                                               ctx.max_bin_count_among_ftrs_,
                                                               hist_prop_count);
    auto nodesHistograms =
        pr::ndarray<hist_type_t, 1>::empty(queue_, { node_count * partHistSize }, alloc::device);

    sycl::event last_event;

    if (1 == nPartialHistograms) {
        last_event = compute_partial_histograms(data,
                                                response,
                                                treeOrder,
                                                selectedFeatures,
                                                binOffsets,
                                                node_list,
                                                nodeIndices,
                                                nodeIndicesOffset,
                                                nodesHistograms,
                                                nPartialHistograms,
                                                node_count,
                                                ctx,
                                                { deps });

        last_event.wait_and_throw();
    }
    else {
        auto partialHistograms =
            pr::ndarray<hist_type_t, 1>::empty(queue_,
                                               { node_count * nPartialHistograms * partHistSize },
                                               alloc::device);
        last_event = compute_partial_histograms(data,
                                                response,
                                                treeOrder,
                                                selectedFeatures,
                                                binOffsets,
                                                node_list,
                                                nodeIndices,
                                                nodeIndicesOffset,
                                                partialHistograms,
                                                nPartialHistograms,
                                                node_count,
                                                ctx,
                                                { deps });

        last_event = reduce_partial_histograms(ctx,
                                               partialHistograms,
                                               nodesHistograms,
                                               nPartialHistograms,
                                               node_count,
                                               { last_event });

        last_event.wait_and_throw();
    }

    return std::make_tuple(nodesHistograms, last_event);
}

template <typename Float, typename Bin, typename Index, typename Task>
std::tuple<pr::ndarray<typename task_types<Float, Index, Task>::hist_type_t, 1>, sycl::event>
train_kernel_hist_impl<Float, Bin, Index, Task>::compute_histogram_distr(
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    Index nPartialHistograms,
    Index node_count,
    const be::event_vector& deps) {
    pr::ndarray<hist_type_t, 1> nodesHistograms;
    sycl::event last_event;

    if constexpr (std::is_same_v<task::classification, Task>) {
        std::tie(nodesHistograms, last_event) = compute_histogram(ctx,
                                                                  data,
                                                                  response,
                                                                  treeOrder,
                                                                  selectedFeatures,
                                                                  binOffsets,
                                                                  node_list,
                                                                  nodeIndices,
                                                                  nodeIndicesOffset,
                                                                  nPartialHistograms,
                                                                  node_count,
                                                                  deps);
        last_event = allreduce_ndarray_inplace(nodesHistograms, { last_event });
    }
    else {
        //Index hist_prop_count = impl_const_t::hist_prop_count_;

        const Index partHistSize = get_part_hist_required_mem_size(ctx.selected_ftr_count_,
                                                                   ctx.max_bin_count_among_ftrs_,
                                                                   impl_const_t::hist_prop_count_);

        nodesHistograms = pr::ndarray<hist_type_t, 1>::empty(queue_,
                                                             { node_count * partHistSize },
                                                             alloc::device);

        sycl::event last_event;
        if (1 == nPartialHistograms) {
            const Index part_sum_hist_size =
                get_part_hist_required_mem_size(ctx.selected_ftr_count_,
                                                ctx.max_bin_count_among_ftrs_,
                                                impl_const_t::hist_prop_sum_count_);

            const Index part_sum2cent_hist_size =
                get_part_hist_required_mem_size(ctx.selected_ftr_count_,
                                                ctx.max_bin_count_among_ftrs_,
                                                impl_const_t::hist_prop_sum2cent_count_);

            auto sum_list = pr::ndarray<Float, 1>::empty(queue_,
                                                         { node_count * part_sum_hist_size },
                                                         alloc::device);
            auto sum2cent_list =
                pr::ndarray<Float, 1>::empty(queue_,
                                             { node_count * part_sum2cent_hist_size },
                                             alloc::device);
            last_event = compute_partial_count_and_sum(ctx,
                                                       data,
                                                       response,
                                                       treeOrder,
                                                       selectedFeatures,
                                                       binOffsets,
                                                       node_list,
                                                       nodeIndices,
                                                       nodeIndicesOffset,
                                                       sum_list,
                                                       nPartialHistograms,
                                                       node_count,
                                                       { last_event });

            last_event.wait_and_throw();
            last_event = allreduce_ndarray_inplace(sum_list, { last_event });

            last_event = compute_partial_sum2cent(ctx,
                                                  data,
                                                  response,
                                                  sum_list,
                                                  treeOrder,
                                                  selectedFeatures,
                                                  binOffsets,
                                                  node_list,
                                                  nodeIndices,
                                                  nodeIndicesOffset,
                                                  sum2cent_list,
                                                  nPartialHistograms,
                                                  node_count,
                                                  { last_event });

            last_event.wait_and_throw();
            last_event = allreduce_ndarray_inplace(sum2cent_list, { last_event });

            last_event = fin_histogram_distr(ctx,
                                             sum_list,
                                             sum2cent_list,
                                             nodesHistograms,
                                             node_count,
                                             { last_event });
            last_event.wait_and_throw();
        }
        else {
            const Index part_sum_hist_size =
                get_part_hist_required_mem_size(ctx.selected_ftr_count_,
                                                ctx.max_bin_count_among_ftrs_,
                                                impl_const_t::hist_prop_sum_count_);

            const Index part_sum2cent_hist_size =
                get_part_hist_required_mem_size(ctx.selected_ftr_count_,
                                                ctx.max_bin_count_among_ftrs_,
                                                impl_const_t::hist_prop_sum2cent_count_);

            auto sum_list = pr::ndarray<Float, 1>::empty(queue_,
                                                         { node_count * part_sum_hist_size },
                                                         alloc::device);
            auto sum2cent_list =
                pr::ndarray<Float, 1>::empty(queue_,
                                             { node_count * part_sum2cent_hist_size },
                                             alloc::device);

            auto part_sum_list = pr::ndarray<Float, 1>::empty(
                queue_,
                { node_count * nPartialHistograms * part_sum_hist_size },
                alloc::device);
            auto part_sum2cent_list = pr::ndarray<Float, 1>::empty(
                queue_,
                { node_count * nPartialHistograms * part_sum2cent_hist_size },
                alloc::device);

            last_event = compute_partial_count_and_sum(ctx,
                                                       data,
                                                       response,
                                                       treeOrder,
                                                       selectedFeatures,
                                                       binOffsets,
                                                       node_list,
                                                       nodeIndices,
                                                       nodeIndicesOffset,
                                                       part_sum_list,
                                                       nPartialHistograms,
                                                       node_count,
                                                       { last_event });

            last_event.wait_and_throw();

            last_event = sum_reduce_partial_histograms(ctx,
                                                       part_sum_list,
                                                       sum_list,
                                                       nPartialHistograms,
                                                       node_count,
                                                       impl_const_t::hist_prop_sum_count_,
                                                       { last_event });

            last_event.wait_and_throw();

            last_event = allreduce_ndarray_inplace(sum_list, { last_event });

            last_event = compute_partial_sum2cent(ctx,
                                                  data,
                                                  response,
                                                  sum_list,
                                                  treeOrder,
                                                  selectedFeatures,
                                                  binOffsets,
                                                  node_list,
                                                  nodeIndices,
                                                  nodeIndicesOffset,
                                                  part_sum2cent_list,
                                                  nPartialHistograms,
                                                  node_count,
                                                  { last_event });

            last_event.wait_and_throw();

            last_event = sum_reduce_partial_histograms(ctx,
                                                       part_sum2cent_list,
                                                       sum2cent_list,
                                                       nPartialHistograms,
                                                       node_count,
                                                       impl_const_t::hist_prop_sum2cent_count_,
                                                       { last_event });

            last_event.wait_and_throw();

            last_event = allreduce_ndarray_inplace(sum2cent_list, { last_event });

            last_event = fin_histogram_distr(ctx,
                                             sum_list,
                                             sum2cent_list,
                                             nodesHistograms,
                                             node_count,
                                             { last_event });
        }
    }

    return std::make_tuple(nodesHistograms, last_event);
}

template <typename Index>
inline void get_block_borders(Index total_elem_count,
                              Index block_count,
                              Index block_id,
                              Index& ind_start,
                              Index& ind_end) {
    const Index elem_count = total_elem_count / block_count + bool(total_elem_count % block_count);

    ind_start = block_id * elem_count;
    ind_end = sycl::min(static_cast<Index>(block_id + 1) * elem_count, total_elem_count);
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_partial_histograms(
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    pr::ndarray<hist_type_t, 1>& partialHistograms,
    Index nPartialHistograms,
    Index node_count,
    const context_t& ctx,
    const be::event_vector& deps) {
    //using cst = impl_const<Index, Task>;

    auto fill_event = partialHistograms.fill(queue_, 0, deps);
    fill_event.wait_and_throw();

    const Bin* data_ptr = data.get_data();
    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = treeOrder.get_data();
    const Index* selected_ftrs_ptr = selectedFeatures.get_data();
    const Index* node_list_ptr = node_list.get_data();
    const Index* node_indices_ptr = nodeIndices.get_data();

    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const<Index, task::regression>::hist_prop_count_;
    }
    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;
    const Index selected_ftr_count = ctx.selected_ftr_count_;
    const Index column_count = ctx.column_count_;

    hist_type_t* partial_histogram_ptr = partialHistograms.get_mutable_data();

    auto local_size = ctx.preferable_local_size_for_part_hist_kernel_;
    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ nPartialHistograms * local_size, node_count },
                                      { local_size, 1 });

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(fill_event);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            const Index nodeIdx = item.get_global_id()[1];
            const Index nodeId = node_indices_ptr[nodeIndicesOffset + nodeIdx];
            const Index ftrGrpIdx = item.get_local_id()[0];
            const Index ftrGrpSize = item.get_local_range()[0];
            const Index nPartHist = item.get_group_range(0);
            const Index histIdx = item.get_group().get_id(0);

            const Index rowsOffset =
                node_list_ptr[nodeId * node_prop_count + impl_const_t::ind_ofs];
            //const Index row_count = node_list_ptr[nodeId * node_prop_count + 1];
            const Index row_count = node_list_ptr[nodeId * node_prop_count +
                                                  impl_const_t::ind_lrc]; // get local_row_count

            Index ind_start;
            Index ind_end;
            get_block_borders(row_count, nPartHist, histIdx, ind_start, ind_end);

            for (Index i = ind_start; i < ind_end; i++) {
                Index id = tree_order_ptr[rowsOffset + i];
                for (Index featIdx = ftrGrpIdx; featIdx < selected_ftr_count;
                     featIdx += ftrGrpSize) {
                    const Index featId = selected_ftrs_ptr[nodeId * selected_ftr_count + featIdx];

                    hist_type_t* histogram_ptr =
                        partial_histogram_ptr +
                        ((nodeIdx * nPartHist + histIdx) * selected_ftr_count + featIdx) *
                            max_bin_count_among_ftrs * hist_prop_count;

                    Index bin = data_ptr[id * column_count + featId];
                    add_val_to_hist<Float, Index>(histogram_ptr + bin * hist_prop_count,
                                                  response_ptr[id]);
                }
            }
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_partial_count_and_sum(
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    pr::ndarray<Float, 1>& partialHistograms,
    Index part_hist_count,
    Index node_count,
    const be::event_vector& deps,
    const task::regression task_val) {
    const Index hist_prop_count = impl_const_t::hist_prop_sum_count_;
    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;
    const Index selected_ftr_count = ctx.selected_ftr_count_;
    const Index column_count = ctx.column_count_;

    ONEDAL_ASSERT(partialHistograms.get_count() == hist_prop_count * ctx.max_bin_count_among_ftrs_ *
                                                       selected_ftr_count * part_hist_count *
                                                       node_count);

    auto fill_event = partialHistograms.fill(queue_, 0, deps);
    fill_event.wait_and_throw();

    const Bin* data_ptr = data.get_data();
    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = treeOrder.get_data();
    const Index* selected_ftrs_ptr = selectedFeatures.get_data();
    const Index* node_list_ptr = node_list.get_data();
    const Index* node_indices_ptr = nodeIndices.get_data();

    Float* partial_histogram_ptr = partialHistograms.get_mutable_data();

    auto local_size = ctx.preferable_local_size_for_part_hist_kernel_;
    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ part_hist_count * local_size, node_count },
                                      { local_size, 1 });

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(fill_event);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            const Index nodeIdx = item.get_global_id()[1];
            const Index nodeId = node_indices_ptr[nodeIndicesOffset + nodeIdx];
            const Index ftrGrpIdx = item.get_local_id()[0];
            const Index ftrGrpSize = item.get_local_range()[0];
            const Index nPartHist = item.get_group_range(0);
            const Index histIdx = item.get_group().get_id(0);

            const Index rowsOffset =
                node_list_ptr[nodeId * node_prop_count + impl_const_t::ind_ofs];
            const Index row_count = node_list_ptr[nodeId * node_prop_count +
                                                  impl_const_t::ind_lrc]; // get local_row_count

            Index ind_start;
            Index ind_end;
            get_block_borders(row_count, nPartHist, histIdx, ind_start, ind_end);

            for (Index i = ind_start; i < ind_end; i++) {
                Index id = tree_order_ptr[rowsOffset + i];
                for (Index featIdx = ftrGrpIdx; featIdx < selected_ftr_count;
                     featIdx += ftrGrpSize) {
                    const Index featId = selected_ftrs_ptr[nodeId * selected_ftr_count + featIdx];

                    Float* histogram_ptr =
                        partial_histogram_ptr +
                        ((nodeIdx * nPartHist + histIdx) * selected_ftr_count + featIdx) *
                            max_bin_count_among_ftrs * hist_prop_count;

                    Index bin = data_ptr[id * column_count + featId];
                    histogram_ptr[bin * hist_prop_count + 0] += Float(1);
                    histogram_ptr[bin * hist_prop_count + 1] += response_ptr[id];
                }
            }
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_partial_sum2cent(
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndview<Float, 1>& sum_list,
    const pr::ndarray<Index, 1>& treeOrder,
    const pr::ndarray<Index, 1>& selectedFeatures,
    const pr::ndarray<Index, 1>& binOffsets,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& nodeIndices,
    Index nodeIndicesOffset,
    pr::ndarray<Float, 1>& partialHistograms,
    Index part_hist_count,
    Index node_count,
    const be::event_vector& deps,
    const task::regression task_val) {
    const Index hist_prop_sum_count = impl_const_t::hist_prop_sum_count_;
    const Index hist_prop_count = impl_const_t::hist_prop_sum2cent_count_;
    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;
    const Index selected_ftr_count = ctx.selected_ftr_count_;
    const Index column_count = ctx.column_count_;

    ONEDAL_ASSERT(sum_list.get_count() == hist_prop_sum_count * ctx.max_bin_count_among_ftrs_ *
                                              selected_ftr_count * part_hist_count * node_count);
    ONEDAL_ASSERT(partialHistograms.get_count() == hist_prop_count * ctx.max_bin_count_among_ftrs_ *
                                                       selected_ftr_count * part_hist_count *
                                                       node_count);

    auto fill_event = partialHistograms.fill(queue_, 0, deps);
    fill_event.wait_and_throw();

    const Bin* data_ptr = data.get_data();
    const Float* response_ptr = response.get_data();
    const Float* sum_list_ptr = sum_list.get_data();
    const Index* tree_order_ptr = treeOrder.get_data();
    const Index* selected_ftrs_ptr = selectedFeatures.get_data();
    const Index* node_list_ptr = node_list.get_data();
    const Index* node_indices_ptr = nodeIndices.get_data();

    Float* partial_histogram_ptr = partialHistograms.get_mutable_data();

    auto local_size = ctx.preferable_local_size_for_part_hist_kernel_;
    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ part_hist_count * local_size, node_count },
                                      { local_size, 1 });

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(fill_event);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            const Index nodeIdx = item.get_global_id()[1];
            const Index nodeId = node_indices_ptr[nodeIndicesOffset + nodeIdx];
            const Index ftrGrpIdx = item.get_local_id()[0];
            const Index ftrGrpSize = item.get_local_range()[0];
            const Index nPartHist = item.get_group_range(0);
            const Index histIdx = item.get_group().get_id(0);

            const Index rowsOffset =
                node_list_ptr[nodeId * node_prop_count + impl_const_t::ind_ofs];
            const Index row_count = node_list_ptr[nodeId * node_prop_count +
                                                  impl_const_t::ind_lrc]; // get local_row_count

            Index ind_start;
            Index ind_end;
            get_block_borders(row_count, nPartHist, histIdx, ind_start, ind_end);

            for (Index i = ind_start; i < ind_end; i++) {
                Index id = tree_order_ptr[rowsOffset + i];
                for (Index featIdx = ftrGrpIdx; featIdx < selected_ftr_count;
                     featIdx += ftrGrpSize) {
                    const Index featId = selected_ftrs_ptr[nodeId * selected_ftr_count + featIdx];

                    Float* histogram_ptr =
                        partial_histogram_ptr +
                        ((nodeIdx * nPartHist + histIdx) * selected_ftr_count + featIdx) *
                            max_bin_count_among_ftrs * hist_prop_count;

                    const Float* sum_ptr = sum_list_ptr + (nodeIdx * selected_ftr_count + featIdx) *
                                                              max_bin_count_among_ftrs *
                                                              hist_prop_sum_count;

                    Index bin = data_ptr[id * column_count + featId];
                    Float count = sum_ptr[bin * hist_prop_sum_count + 0];
                    Float mean = (count >= Float(1))
                                     ? sum_ptr[bin * hist_prop_sum_count + 1] / count
                                     : Float(0);
                    histogram_ptr[bin * hist_prop_count + 0] +=
                        (response_ptr[id] - mean) * (response_ptr[id] - mean);
                }
            }
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::fin_histogram_distr(
    const context_t& ctx,
    const pr::ndarray<Float, 1>& sum_list,
    const pr::ndarray<Float, 1>& sum2cent_list,
    pr::ndarray<Float, 1>& hist_list,
    Index node_count,
    const be::event_vector& deps) {
    const Index hist_prop_sum_count = impl_const_t::hist_prop_sum_count_;
    const Index hist_prop_sum2cent_count = impl_const_t::hist_prop_sum2cent_count_;
    const Index hist_prop_count = impl_const_t::hist_prop_count_;

    ONEDAL_ASSERT(3 == hist_prop_count); // count, mean, sum2cent
    ONEDAL_ASSERT(sum_list.get_count() == hist_prop_sum_count * ctx.max_bin_count_among_ftrs_ *
                                              ctx.selected_ftr_count_ * node_count);
    ONEDAL_ASSERT(sum2cent_list.get_count() == hist_prop_sum2cent_count *
                                                   ctx.max_bin_count_among_ftrs_ *
                                                   ctx.selected_ftr_count_ * node_count);
    ONEDAL_ASSERT(hist_list.get_count() == hist_prop_count * ctx.max_bin_count_among_ftrs_ *
                                               ctx.selected_ftr_count_ * node_count);

    const Float* sum_list_ptr = sum_list.get_data();
    const Float* sum2cent_list_ptr = sum2cent_list.get_data();

    Float* hist_ptr = hist_list.get_mutable_data();

    //mul overflow is checked during hist_list accumulation
    const sycl::range<1> range{ de::integral_cast<size_t>(ctx.max_bin_count_among_ftrs_ *
                                                          ctx.selected_ftr_count_ * node_count) };

    auto last_event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            Float count = sum_list_ptr[idx * hist_prop_sum_count + 0];
            hist_ptr[idx * hist_prop_count + 0] = count;
            hist_ptr[idx * hist_prop_count + 1] =
                (count >= Float(1)) ? sum_list_ptr[idx * hist_prop_sum_count + 1] / count
                                    : Float(0);
            hist_ptr[idx * hist_prop_count + 2] =
                sum2cent_list_ptr[idx * hist_prop_sum2cent_count + 0];
        });
    });

    last_event.wait_and_throw();

    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::reduce_partial_histograms(
    const context_t& ctx,
    const pr::ndarray<hist_type_t, 1>& partialHistograms,
    pr::ndarray<hist_type_t, 1>& histograms,
    Index nPartialHistograms,
    Index node_count,
    const be::event_vector& deps) {
    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const<Index, task::regression>::hist_prop_count_;
    }
    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;
    const Index selected_ftr_count = ctx.selected_ftr_count_;

    const hist_type_t* partial_histogram_ptr = partialHistograms.get_data();
    hist_type_t* histogram_ptr = histograms.get_mutable_data();

    // overflow for nMaxBinsAmongFtrs * nSelectedFeatures should be checked in compute
    const sycl::nd_range<3> nd_range =
        be::make_multiple_nd_range_3d({ max_bin_count_among_ftrs * selected_ftr_count,
                                        ctx.reduce_local_size_part_hist_,
                                        node_count },
                                      { 1, ctx.reduce_local_size_part_hist_, 1 });

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> buf(ctx.reduce_local_size_part_hist_ * hist_prop_count,
                                             cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) {
            const Index nodeIdx = item.get_global_id()[2];
            const Index binId = item.get_global_id()[0];
            const Index local_id = item.get_local_id()[1];
            const Index local_size = item.get_local_range()[1];

            hist_type_t* buf_ptr = buf.get_pointer().get();

            for (Index prop = 0; prop < hist_prop_count; prop++) {
                buf_ptr[local_id * hist_prop_count + prop] = 0;
            }

            const hist_type_t* nodePartialHistograms =
                partial_histogram_ptr + nodeIdx * nPartialHistograms * selected_ftr_count *
                                            max_bin_count_among_ftrs * hist_prop_count;
            hist_type_t* nodeHistogram = histogram_ptr + nodeIdx * selected_ftr_count *
                                                             max_bin_count_among_ftrs *
                                                             hist_prop_count;

            for (Index i = local_id; i < nPartialHistograms; i += local_size) {
                Index offset = i * selected_ftr_count * max_bin_count_among_ftrs * hist_prop_count +
                               binId * hist_prop_count;
                merge_stat(buf_ptr + local_id * hist_prop_count,
                           nodePartialHistograms + offset,
                           hist_prop_count);
            }

            for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if (local_id < offset) {
                    merge_stat(buf_ptr + local_id * hist_prop_count,
                               buf_ptr + (local_id + offset) * hist_prop_count,
                               hist_prop_count);
                }
            }

            if (local_id == 0) {
                merge_stat(nodeHistogram + binId * hist_prop_count,
                           buf_ptr + local_id * hist_prop_count,
                           hist_prop_count);
            }
        });
    });
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::sum_reduce_partial_histograms(
    const context_t& ctx,
    const pr::ndarray<Float, 1>& partialHistograms,
    pr::ndarray<Float, 1>& histograms,
    Index part_hist_count,
    Index node_count,
    Index hist_prop_count,
    const be::event_vector& deps) {
    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;
    const Index selected_ftr_count = ctx.selected_ftr_count_;

    ONEDAL_ASSERT(partialHistograms.get_count() == hist_prop_count * max_bin_count_among_ftrs *
                                                       selected_ftr_count * part_hist_count *
                                                       node_count);
    ONEDAL_ASSERT(histograms.get_count() ==
                  max_bin_count_among_ftrs * selected_ftr_count * part_hist_count * node_count);

    const Float* partial_histogram_ptr = partialHistograms.get_data();
    Float* histogram_ptr = histograms.get_mutable_data();

    // overflow for nMaxBinsAmongFtrs * nSelectedFeatures should be checked in compute
    const sycl::nd_range<3> nd_range =
        be::make_multiple_nd_range_3d({ max_bin_count_among_ftrs * selected_ftr_count,
                                        ctx.reduce_local_size_part_hist_,
                                        node_count },
                                      { 1, ctx.reduce_local_size_part_hist_, 1 });

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<Float> buf(ctx.reduce_local_size_part_hist_ * hist_prop_count, cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) {
            const Index nodeIdx = item.get_global_id()[2];
            const Index binId = item.get_global_id()[0];
            const Index local_id = item.get_local_id()[1];
            const Index local_size = item.get_local_range()[1];

            Float* buf_ptr = buf.get_pointer().get();

            for (Index prop = 0; prop < hist_prop_count; prop++) {
                buf_ptr[local_id * hist_prop_count + prop] = 0;
            }

            const Float* nodePartialHistograms =
                partial_histogram_ptr + nodeIdx * part_hist_count * selected_ftr_count *
                                            max_bin_count_among_ftrs * hist_prop_count;
            Float* nodeHistogram = histogram_ptr + nodeIdx * selected_ftr_count *
                                                       max_bin_count_among_ftrs * hist_prop_count;

            for (Index i = local_id; i < part_hist_count; i += local_size) {
                Index offset = i * selected_ftr_count * max_bin_count_among_ftrs * hist_prop_count +
                               binId * hist_prop_count;
                for (Index prop = 0; prop < hist_prop_count; prop++) {
                    buf_ptr[local_id * hist_prop_count + prop] +=
                        nodePartialHistograms[offset + prop];
                }
            }

            for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if (local_id < offset) {
                    for (Index prop = 0; prop < hist_prop_count; prop++) {
                        buf_ptr[local_id * hist_prop_count + prop] +=
                            buf_ptr[(local_id + offset) * hist_prop_count + prop];
                    }
                }
            }

            if (local_id == 0) {
                for (Index prop = 0; prop < hist_prop_count; prop++) {
                    nodeHistogram[binId * hist_prop_count + prop] =
                        buf_ptr[local_id * hist_prop_count + prop];
                }
            }
        });
    });
    return event;
}

template <typename Float, typename Index, typename Task>
static void do_node_imp_split(const imp_data_list_ptr<Float, Index, Task>& imp_list_ptr,
                              const imp_data_list_ptr<Float, Index, Task>& left_imp_list_ptr,
                              const imp_data_list_ptr_mutable<Float, Index, Task>& imp_list_ptr_new,
                              const Index* node_par,
                              Index* node_lch,
                              Index* node_rch,
                              const kernel_context<Float, Index, Task>& ctx,
                              Index node_id,
                              Index new_left_node_pos) {
    using impl_const_t = impl_const<Index, Task>;

    if constexpr (std::is_same_v<Task, task::classification>) {
        // assign class hist and compute winner for new nodes
        const Index* class_hist_p = imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;
        const Index* left_child_class_hist =
            left_imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;
        Index* class_hist_left =
            imp_list_ptr_new.class_hist_list_ptr_ + new_left_node_pos * ctx.class_count_;
        Index* class_hist_right =
            imp_list_ptr_new.class_hist_list_ptr_ + (new_left_node_pos + 1) * ctx.class_count_;

        Index max_cls_count_left = 0;
        Index max_cls_count_right = 0;
        Index win_cls_left = 0;
        Index win_cls_right = 0;

        Index rows_right = node_rch[impl_const_t::ind_grc];
        Float imp_right = Float(1);
        Float div_right = (0 < rows_right) ? Float(1) / (Float(rows_right) * rows_right) : Float(0);

        for (Index class_id = 0; class_id < ctx.class_count_; class_id++) {
            class_hist_left[class_id] = left_child_class_hist[class_id];
            Index class_count_right = class_hist_p[class_id] - left_child_class_hist[class_id];
            class_hist_right[class_id] = class_count_right;

            imp_right -= Float(class_count_right) * (class_count_right)*div_right;

            if (class_hist_left[class_id] > max_cls_count_left) {
                max_cls_count_left = class_hist_left[class_id];
                win_cls_left = class_id;
            }

            if (class_hist_right[class_id] > max_cls_count_right) {
                max_cls_count_right = class_hist_right[class_id];
                win_cls_right = class_id;
            }
        }

        node_lch[impl_const_t::ind_win] = win_cls_left;
        node_rch[impl_const_t::ind_win] = win_cls_right;

        // assign impurity for new nodes
        const Float* left_child_imp =
            left_imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float* node_lch_imp =
            imp_list_ptr_new.imp_list_ptr_ + new_left_node_pos * impl_const_t::node_imp_prop_count_;
        Float* node_rch_imp = imp_list_ptr_new.imp_list_ptr_ +
                              (new_left_node_pos + 1) * impl_const_t::node_imp_prop_count_;
        node_lch_imp[0] = left_child_imp[0];
        node_rch_imp[0] = sycl::max(imp_right, Float(0));
    }
    else {
        constexpr Index buff_size = impl_const_t::node_imp_prop_count_ + 1;
        const Float* left_child_imp =
            left_imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        const Float* impP =
            imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float* node_lch_imp =
            imp_list_ptr_new.imp_list_ptr_ + new_left_node_pos * impl_const_t::node_imp_prop_count_;
        Float* node_rch_imp = imp_list_ptr_new.imp_list_ptr_ +
                              (new_left_node_pos + 1) * impl_const_t::node_imp_prop_count_;

        Float node_hist[buff_size] = { static_cast<Float>(node_par[impl_const_t::ind_grc]),
                                       impP[0],
                                       impP[1] };
        Float left_hist[buff_size] = { static_cast<Float>(node_lch[impl_const_t::ind_grc]),
                                       left_child_imp[0],
                                       left_child_imp[1] };
        Float right_hist[buff_size] = { 0 };

        sub_stat(&right_hist[0], &left_hist[0], &node_hist[0], buff_size);

        node_lch_imp[0] = left_child_imp[0];
        node_lch_imp[1] = left_child_imp[1];

        node_rch_imp[0] = right_hist[1];
        node_rch_imp[1] = right_hist[2];
    }
}
//////////////////////////// DO NODE SPLIT
template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::do_node_split(
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& node_vs_tree_map_list,
    const imp_data_t& imp_data_list,
    const imp_data_t& left_child_imp_data_list,
    pr::ndarray<Index, 1>& node_list_new,
    pr::ndarray<Index, 1>& node_vs_tree_map_list_new,
    imp_data_t& imp_data_list_new,
    Index node_count,
    Index node_count_new,
    const context_t& ctx,
    const be::event_vector& deps) {
    // input asserts is going to be added

    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index bad_val = impl_const_t::bad_val_;

    const Index* node_list_ptr = node_list.get_data();
    const Index* node_vs_tree_map_list_ptr = node_vs_tree_map_list.get_data();

    const bool distr_mode = ctx.distr_mode_;

    Index* node_list_new_ptr = node_list_new.get_mutable_data();
    Index* node_vs_tree_map_list_new_ptr = node_vs_tree_map_list_new.get_mutable_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);
    imp_data_list_ptr<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    imp_data_list_ptr_mutable<Float, Index, Task> imp_list_ptr_new(imp_data_list_new);

    const kernel_context<Float, Index, Task> krn_ctx(ctx);

    auto local_size = be::device_max_sg_size(queue_);
    const sycl::nd_range<1> nd_range = be::make_multiple_nd_range_1d(local_size, local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }

            const Index local_id = sbg.get_local_id();
            const Index local_size = sbg.get_local_range()[0];

            Index created_node_count = 0;
            for (Index node_id = local_id; node_id < node_count; node_id += local_size) {
                Index splitNode =
                    Index(node_list_ptr[node_id * node_prop_count + impl_const_t::ind_fid] !=
                          bad_val); // featId != -1
                Index new_left_node_pos =
                    created_node_count + exclusive_scan(sbg, splitNode, plus<Index>()) * 2;
                if (splitNode) {
                    // split parent node on left and right nodes
                    const Index* node_prn = node_list_ptr + node_id * node_prop_count;
                    Index* node_lch = node_list_new_ptr + new_left_node_pos * node_prop_count;
                    Index* node_rch = node_list_new_ptr + (new_left_node_pos + 1) * node_prop_count;

                    node_lch[impl_const_t::ind_ofs] =
                        node_prn[impl_const_t::ind_ofs]; // rows offset
                    node_lch[impl_const_t::ind_lrc] =
                        distr_mode ? node_prn[impl_const_t::ind_lch_lrc]
                                   : node_prn[impl_const_t::ind_lch_grc]; // local row_count
                    node_lch[impl_const_t::ind_grc] =
                        node_prn[impl_const_t::ind_lch_grc]; // global nRows
                    node_lch[impl_const_t::ind_fid] = impl_const_t::leaf_mark_; // featureId
                    node_lch[impl_const_t::ind_bin] = impl_const_t::leaf_mark_; // featureVal
                    node_lch[impl_const_t::ind_lch_lrc] = 0;
                    node_lch[impl_const_t::ind_lch_grc] = 0;

                    node_rch[impl_const_t::ind_ofs] =
                        node_prn[impl_const_t::ind_ofs] + node_lch[impl_const_t::ind_lrc];
                    node_rch[impl_const_t::ind_lrc] =
                        node_prn[impl_const_t::ind_lrc] - node_lch[impl_const_t::ind_lrc];
                    node_rch[impl_const_t::ind_grc] =
                        node_prn[impl_const_t::ind_grc] - node_lch[impl_const_t::ind_grc];
                    node_rch[impl_const_t::ind_fid] = impl_const_t::leaf_mark_;
                    node_rch[impl_const_t::ind_bin] = impl_const_t::leaf_mark_;
                    node_rch[impl_const_t::ind_lch_lrc] = 0;
                    node_rch[impl_const_t::ind_lch_grc] = 0;

                    node_vs_tree_map_list_new_ptr[new_left_node_pos] =
                        node_vs_tree_map_list_ptr[node_id];
                    node_vs_tree_map_list_new_ptr[new_left_node_pos + 1] =
                        node_vs_tree_map_list_ptr[node_id];

                    do_node_imp_split<Float, Index, Task>(imp_list_ptr,
                                                          left_imp_list_ptr,
                                                          imp_list_ptr_new,
                                                          node_prn,
                                                          node_lch,
                                                          node_rch,
                                                          krn_ctx,
                                                          node_id,
                                                          new_left_node_pos);
                }
                created_node_count += reduce(sbg, splitNode, plus<Index>()) * 2;
            }
        });
    });
    event.wait_and_throw();

    return event;
}

/////////////////////////////////////////////////////////
/// compute results
template <typename Float, typename Bin, typename Index, typename Task>
Float train_kernel_hist_impl<Float, Bin, Index, Task>::compute_oob_error(
    const model_manager_t& model_manager,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
    Index tree_idx,
    Index indicesOffset,
    Index n,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(data_host.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response_host.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(oob_per_obs_list.get_count() == ctx.row_count_ * ctx.oob_prop_count_);
    // input asserts is going to be added

    auto oob_row_list_host = oob_row_list.to_host(queue_, deps);
    auto oob_per_obs_list_host = oob_per_obs_list.to_host(queue_, deps);

    const Float* data_host_ptr = data_host.get_data();
    const Float* response_host_ptr = response_host.get_data();
    const Index* oob_row_list_host_ptr = oob_row_list_host.get_data();
    hist_type_t* oob_per_obs_list_host_ptr = oob_per_obs_list_host.get_mutable_data();

    //compute prediction error on each OOB row and get its mean online formulae (Welford)

    Float mean = 0;
    for (Index i = 0; i < n; i++) {
        Index row_ind = oob_row_list_host_ptr[indicesOffset + i];
        ONEDAL_ASSERT(row_ind < ctx.row_count_);

        auto prediction =
            model_manager.get_tree_response(tree_idx, &data_host_ptr[row_ind * ctx.column_count_]);

        if constexpr (std::is_same_v<Task, task::classification>) {
            ONEDAL_ASSERT(ctx.class_count_ == ctx.oob_prop_count_);
            Index class_idx = static_cast<Index>(prediction);
            oob_per_obs_list_host_ptr[row_ind * ctx.oob_prop_count_ + class_idx]++;
            mean += Float(class_idx != Index(response_host_ptr[row_ind]));
        }
        else {
            ONEDAL_ASSERT(2 == ctx.oob_prop_count_);
            oob_per_obs_list_host_ptr[row_ind * ctx.oob_prop_count_ + 0] += prediction;
            oob_per_obs_list_host_ptr[row_ind * ctx.oob_prop_count_ + 1] += Float(1);
            mean += (prediction - response_host_ptr[row_ind]) *
                    (prediction - response_host_ptr[row_ind]);
        }
    }

    oob_per_obs_list = oob_per_obs_list_host.to_device(queue_);

    return mean / Float(n);
}

template <typename Float, typename Bin, typename Index, typename Task>
Float train_kernel_hist_impl<Float, Bin, Index, Task>::compute_oob_error_perm(
    const model_manager_t& model_manager,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    const pr::ndarray<Index, 1>& permutation_host,
    Index tree_idx,
    Index indicesOffset,
    Index n,
    Index column_idx,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(data_host.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response_host.get_count() == ctx.column_count_);
    ONEDAL_ASSERT(permutation_host.get_count() == n);
    ONEDAL_ASSERT(column_idx < ctx.column_count_);
    // input asserts is going to be added

    auto oob_row_list_host = oob_row_list.to_host(queue_, deps);

    const Float* data_host_ptr = data_host.get_data();
    const Float* response_host_ptr = response_host.get_data();
    const Index* oob_row_list_host_ptr = oob_row_list_host.get_data();
    const Index* permutation_ptr = permutation_host.get_data();

    auto buf = pr::ndarray<Float, 1>::empty({ ctx.column_count_ });
    Float* buf_ptr = buf.get_mutable_data();

    Float mean = Float(0);
    for (Index i = 0; i < n; i++) {
        int row_ind = oob_row_list_host_ptr[indicesOffset + i];
        int row_ind_perm = permutation_ptr[i];
        ONEDAL_ASSERT(row_ind < ctx.row_count_);
        ONEDAL_ASSERT(row_ind_perm < ctx.row_count_);

        memcpy(de::default_host_policy{},
               buf_ptr,
               &data_host_ptr[row_ind * ctx.column_count_],
               ctx.column_count_ * sizeof(Float));
        buf_ptr[column_idx] = data_host_ptr[row_ind_perm * ctx.column_count_ + column_idx];
        auto prediction = model_manager.get_tree_response(tree_idx, buf_ptr);
        if constexpr (std::is_same_v<Task, task::classification>) {
            Index class_idx = static_cast<Index>(prediction);
            mean += Float(class_idx != Index(response_host_ptr[row_ind]));
        }
        else {
            mean += (prediction - response_host_ptr[row_ind]) *
                    (prediction - response_host_ptr[row_ind]);
        }
    }

    return mean / Float(n);
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_results(
    const model_manager_t& model_manager,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    const pr::ndarray<Index, 1>& oob_rows_num_list,
    pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
    pr::ndarray<Float, 1>& var_imp,
    pr::ndarray<Float, 1>& var_imp_variance,
    const dal::array<engine_impl>& engine_arr,
    Index tree_idx_in_block,
    Index tree_in_block_count,
    Index built_tree_count,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(oob_rows_num_list.get_count() == tree_in_block_count + 1);
    ONEDAL_ASSERT(
        (ctx.mdi_required_ || ctx.mda_required_) ? var_imp.get_count() == ctx.column_count_ : true);
    ONEDAL_ASSERT(ctx.mda_scaled_required_ ? var_imp_variance.get_count() == ctx.column_count_
                                           : true);

    auto oob_rows_count_list_host = oob_rows_num_list.to_host(queue_, deps);
    const Index* oob_rows_count_list_host_ptr = oob_rows_count_list_host.get_data();
    Index oob_indices_offset = oob_rows_count_list_host_ptr[tree_idx_in_block];
    Index oob_row_count = oob_rows_count_list_host_ptr[tree_idx_in_block + 1] -
                          oob_rows_count_list_host_ptr[tree_idx_in_block];

    if ((ctx.oob_required_ || ctx.mda_required_) && oob_row_count) {
        const Float oob_err = compute_oob_error(model_manager,
                                                data_host,
                                                response_host,
                                                oob_row_list,
                                                oob_per_obs_list,
                                                tree_idx_in_block,
                                                oob_indices_offset,
                                                oob_row_count,
                                                ctx);

        if (ctx.mda_required_) {
            auto permutation = pr::ndarray<Index, 1>::empty({ oob_row_count });
            Index* permutation_ptr = permutation.get_mutable_data();
            for (Index i = 0; i < oob_row_count; ++i) {
                permutation_ptr[i] = i;
            }

            auto var_imp_host = var_imp.to_host(queue_);
            Float* var_imp_host_ptr = var_imp_host.get_mutable_data();
            Float* var_imp_var_host_ptr = nullptr;
            if (ctx.mda_scaled_required_) {
                auto var_imp_var_host = var_imp_variance.to_host(queue_);
                var_imp_var_host_ptr = var_imp_var_host.get_mutable_data();
            }

            const Float div1 = Float(1) / Float(built_tree_count + tree_idx_in_block + 1);

            rng<Index> rn_gen;

            for (Index column_idx = 0; column_idx < ctx.column_count_; column_idx++) {
                rn_gen
                    .shuffle(oob_row_count,
                             permutation_ptr,
                             engine_arr[built_tree_count + tree_idx_in_block].get_state())
                    .wait_and_throw();
                const Float oob_err_perm = compute_oob_error_perm(model_manager,
                                                                  data_host,
                                                                  response_host,
                                                                  oob_row_list,
                                                                  permutation,
                                                                  tree_idx_in_block,
                                                                  oob_indices_offset,
                                                                  oob_row_count,
                                                                  column_idx,
                                                                  ctx);

                const Float diff = (oob_err_perm - oob_err);
                const Float delta = diff - var_imp_host_ptr[column_idx];
                var_imp_host_ptr[column_idx] += div1 * delta;
                if (var_imp_var_host_ptr) {
                    var_imp_var_host_ptr[column_idx] +=
                        delta * (diff - var_imp_host_ptr[column_idx]);
                }
            }

            var_imp = var_imp_host.to_device(queue_);
            if (ctx.mda_scaled_required_) {
                var_imp_variance = var_imp_variance_host_.to_device(queue_);
            }
        }
    }

    return sycl::event{};
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::finalize_oob_error(
    const pr::ndarray<Float, 1>& response_host,
    pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
    pr::ndarray<Float, 1>& res_oob_err,
    pr::ndarray<Float, 1>& res_oob_err_obs,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(oob_per_obs_list.get_count() == ctx.row_count_ * ctx.oob_prop_count_);

    auto oob_per_obs_list_host = oob_per_obs_list.to_host(queue_, deps);

    const Float* response_host_ptr = response_host.get_data();
    const hist_type_t* oob_per_obs_list_host_ptr = oob_per_obs_list_host.get_mutable_data();

    auto res_oob_err_host = pr::ndarray<Float, 1>::empty({ 1 });
    auto res_oob_err_obs_host = pr::ndarray<Float, 1>::empty({ ctx.row_count_ });
    Float* res_oob_err_host_ptr = res_oob_err_host.get_mutable_data();
    Float* res_oob_err_obs_host_ptr = res_oob_err_obs_host.get_mutable_data();

    Index predicted_count = 0;
    Float oob_err = 0;

    for (Index i = 0; i < ctx.row_count_; i++) {
        bool predicted = false;
        hist_type_t prediction = hist_type_t(0);

        if constexpr (std::is_same_v<Task, task::classification>) {
            ONEDAL_ASSERT(ctx.class_count_ == ctx.oob_prop_count_);
            Index max_val = 0;
            for (Index class_idx = 0; class_idx < ctx.class_count_; class_idx++) {
                Index val = oob_per_obs_list_host_ptr[i * ctx.oob_prop_count_ + class_idx];
                if (val > max_val) {
                    max_val = val;
                    prediction = class_idx;
                }
            }

            predicted = bool(0 < max_val);
        }
        else {
            ONEDAL_ASSERT(2 == ctx.oob_prop_count_);
            Float value = oob_per_obs_list_host_ptr[i * ctx.oob_prop_count_ + 0];
            Float count = oob_per_obs_list_host_ptr[i * ctx.oob_prop_count_ + 1];

            predicted = bool(count > Float(0));
            prediction = value / count;
        }

        if (predicted) {
            Float prediction_res = Float(0);
            if constexpr (std::is_same_v<Task, task::classification>) {
                prediction_res = Float(prediction != static_cast<Index>(response_host_ptr[i]));
            }
            else {
                prediction_res =
                    (prediction - response_host_ptr[i]) * (prediction - response_host_ptr[i]);
            }

            if (ctx.oob_err_obs_required_)
                res_oob_err_obs_host_ptr[i] = prediction_res;
            oob_err += prediction_res;
            predicted_count++;
        }
        else if (ctx.oob_err_obs_required_)
            //was not in OOB set of any tree and hence not predicted
            res_oob_err_obs_host_ptr[i] = Float(-1);
    }

    if (ctx.oob_err_required_) {
        *res_oob_err_host_ptr = (0 < predicted_count) ? oob_err / Float(predicted_count) : 0;
        res_oob_err = res_oob_err_host.to_device(queue_);
    }

    if (ctx.oob_err_obs_required_) {
        res_oob_err_obs = res_oob_err_obs_host.to_device(queue_);
    }

    return sycl::event{};
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::finalize_var_imp(
    pr::ndarray<Float, 1>& var_imp,
    pr::ndarray<Float, 1>& var_imp_variance,
    const context_t& ctx,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(var_imp.get_count() == ctx.column_count_);

    auto var_imp_host = var_imp.to_host(queue_);
    Float* var_imp_host_ptr = var_imp_host.get_mutable_data();

    if (ctx.mda_scaled_required_) {
        if (ctx.tree_count_ > 1) {
            ONEDAL_ASSERT(var_imp_variance.get_count() == ctx.column_count_);
            auto var_imp_var_host = var_imp_variance.to_host(queue_);
            Float* var_imp_var_host_ptr = var_imp_var_host.get_mutable_data();

            const Float div = Float(1) / Float(ctx.tree_count_);
            for (Index i = 0; i < ctx.column_count_; i++) {
                var_imp_var_host_ptr[i] *= div;
                if (var_imp_var_host_ptr[i] > Float(0)) {
                    var_imp_host_ptr[i] /= std::sqrt(var_imp_var_host_ptr[i] * div);
                }
            }
            var_imp = var_imp_host.to_device(queue_);
        }
        else {
            var_imp.fill(queue_, 0); // addd deps??
        }
    }
    else if (ctx.mdi_required_) {
        const Float div = Float(1) / ctx.tree_count_;
        for (Index i = 0; i < ctx.column_count_; i++)
            var_imp_host_ptr[i] *= div;
        var_imp = var_imp_host.to_device(queue_);
    }

    return sycl::event{};
}

/////////////////////////////////////////////////////////
/// Main compute
template <typename Float, typename Bin, typename Index, typename Task>
train_result<Task> train_kernel_hist_impl<Float, Bin, Index, Task>::operator()(
    const descriptor_t& desc,
    const table& data,
    const table& responses) {
    using imp_data_mng_t = impurity_data_manager<Float, Index, Task>;
    using tree_level_record_t = tree_level_record<Float, Index, Task>;

    validate_input(desc, data, responses);

    context_t ctx;
    init_params(ctx, desc, data, responses);
    allocate_buffers(ctx);

    result_t res;
    model_manager_t model_manager(ctx.tree_count_, ctx.column_count_, ctx);

    /*init engines*/
    auto skip_num =
        de::check_mul_overflow<size_t>(ctx.row_total_count_, (ctx.selected_ftr_count_ + 1));
    skip_num = de::check_mul_overflow<size_t>(ctx.tree_count_, skip_num);

    de::check_mul_overflow<size_t>((ctx.tree_count_ - 1), skip_num);

    engine_collection collection(ctx.tree_count_);
    dal::array<engine_impl> engine_arr = collection([&](size_t i, size_t& skip) {
        skip = i * skip_num;
    });

    pr::ndarray<Float, 1> node_imp_decrease_list;

    sycl::event last_event;

    for (Index iter = 0; iter < ctx.tree_count_; iter += ctx.tree_in_block_) {
        Index iter_tree_count = std::min(ctx.tree_count_ - iter, ctx.tree_in_block_);

        Index node_count = iter_tree_count; // num of potential nodes to split on current tree level
        auto oob_rows_num_list =
            pr::ndarray<Index, 1>::empty(queue_, { iter_tree_count + 1 }, alloc::device);
        pr::ndarray<Index, 1> oob_rows_list;

        std::vector<tree_level_record_t> level_records;
        // lists of nodes int props(rowsOffset, rows, ftrId, ftrVal ... )
        std::vector<pr::ndarray<Index, 1>> level_node_lists;

        imp_data_mng_t imp_data_holder(queue_, ctx);
        // initilizing imp_list and class_hist_list (for classification)
        imp_data_holder.init_new_level(node_count);

        de::check_mul_overflow(node_count, impl_const_t::node_prop_count_);
        de::check_mul_overflow(node_count, impl_const_t::node_imp_prop_count_);
        auto node_vs_tree_map_list_host = pr::ndarray<Index, 1>::empty({ node_count });
        auto level_node_list_init_host =
            pr::ndarray<Index, 1>::empty({ node_count * impl_const_t::node_prop_count_ });

        auto treeMap = node_vs_tree_map_list_host.get_mutable_data();
        auto node_list_ptr = level_node_list_init_host.get_mutable_data();

        for (Index node = 0; node < node_count; node++) {
            Index* node_ptr = node_list_ptr + node * impl_const_t::node_prop_count_;
            treeMap[node] = iter + node;
            node_ptr[impl_const_t::ind_ofs] =
                ctx.selected_row_total_count_ * node; // local row offset
            node_ptr[impl_const_t::ind_lrc] =
                ctx.distr_mode_
                    ? 0
                    : ctx.selected_row_count_; // for distr_mode it will be updated during tree_order_gen
            node_ptr[impl_const_t::ind_grc] =
                ctx.selected_row_total_count_; // global selected rows - it is already filtered for current block
            node_ptr[impl_const_t::ind_lch_lrc] =
                0; // for distr_mode it will be updated during tree_order_gen
        }

        if (ctx.bootstrap_) {
            last_event = gen_initial_tree_order(ctx,
                                                engine_arr,
                                                level_node_list_init_host,
                                                tree_order_lev_,
                                                selected_row_global_host_,
                                                selected_row_host_,
                                                iter,
                                                node_count);
        }
        else {
            last_event = train_service_kernels_.initialize_tree_order(tree_order_lev_,
                                                                      node_count,
                                                                      ctx.selected_row_count_);
        }

        auto node_vs_tree_map_list = node_vs_tree_map_list_host.to_device(queue_);
        level_node_lists.push_back(level_node_list_init_host.to_device(queue_));

        last_event = compute_initial_histogram(ctx,
                                               response_nd_,
                                               tree_order_lev_,
                                               level_node_lists[0],
                                               imp_data_holder.get_mutable_data(0),
                                               node_count,
                                               { last_event });
        last_event.wait_and_throw();

        if (ctx.oob_required_) {
            sycl::event event = train_service_kernels_.get_oob_row_list(
                tree_order_lev_,
                oob_rows_num_list,
                oob_rows_list,
                ctx.selected_row_count_,
                iter_tree_count); // oob_rows_num_list and oob_rows_list are the output
            event.wait_and_throw();
        }

        for (Index level = 0; node_count > 0; level++) {
            auto node_list = level_node_lists[level];

            imp_data_t left_child_imp_data(queue_, ctx, node_count);

            auto [selected_features_com, event] =
                gen_feature_list(node_count, node_vs_tree_map_list, engine_arr, ctx);
            event.wait_and_throw();

            if (ctx.mdi_required_) {
                node_imp_decrease_list =
                    pr::ndarray<Float, 1>::empty(queue_, { node_count }, alloc::device);
            }

            last_event = compute_best_split(full_data_nd_,
                                            response_nd_,
                                            tree_order_lev_,
                                            selected_features_com,
                                            ftr_bin_offsets_nd_,
                                            imp_data_holder.get_data(level),
                                            node_list,
                                            left_child_imp_data,
                                            node_imp_decrease_list,
                                            ctx.mdi_required_,
                                            node_count,
                                            ctx,
                                            { last_event });
            last_event.wait_and_throw();

            tree_level_record_t level_record(queue_,
                                             node_list,
                                             imp_data_holder.get_data(level),
                                             node_count,
                                             ctx);
            level_records.push_back(level_record);

            if (ctx.mdi_required_) {
                //mdi is calculated only on split nodes and is not calculated on last level
                last_event =
                    train_service_kernels_.update_mdi_var_importance(node_list,
                                                                     node_imp_decrease_list,
                                                                     res_var_imp_,
                                                                     ctx.column_count_,
                                                                     node_count,
                                                                     { last_event });
            }

            Index node_count_new;
            last_event = train_service_kernels_.get_split_node_count(node_list,
                                                                     node_count,
                                                                     node_count_new,
                                                                     { last_event });
            last_event.wait_and_throw();

            if (node_count_new) {
                //there are split nodes -> next level is required
                node_count_new *= 2;

                de::check_mul_overflow(node_count_new, impl_const_t::node_prop_count_);
                auto node_list_new = pr::ndarray<Index, 1>::empty(
                    queue_,
                    { node_count_new * impl_const_t::node_prop_count_ },
                    alloc::device);

                imp_data_holder.init_new_level(node_count_new);

                auto node_vs_tree_map_list_new =
                    pr::ndarray<Index, 1>::empty(queue_, { node_count_new }, alloc::device);

                if (ctx.distr_mode_) {
                    last_event =
                        train_service_kernels_.calculate_left_child_row_count_on_local_data(
                            ctx,
                            full_data_nd_,
                            node_list,
                            tree_order_lev_,
                            ctx.column_count_,
                            node_count,
                            { last_event });
                    last_event.wait_and_throw();
                }

                last_event = do_node_split(node_list,
                                           node_vs_tree_map_list,
                                           imp_data_holder.get_data(level),
                                           left_child_imp_data,
                                           node_list_new,
                                           node_vs_tree_map_list_new,
                                           imp_data_holder.get_mutable_data(level + 1),
                                           node_count,
                                           node_count_new,
                                           ctx,
                                           { last_event });
                last_event.wait_and_throw();

                if (ctx.max_tree_depth_ > 0 && ctx.max_tree_depth_ == level) {
                    tree_level_record_t level_record(queue_,
                                                     node_list_new,
                                                     imp_data_holder.get_data(level + 1),
                                                     node_count_new,
                                                     ctx);
                    level_records.push_back(level_record);
                    node_count_new = 0;
                }
                else {
                    level_node_lists.push_back(node_list_new);

                    node_vs_tree_map_list = node_vs_tree_map_list_new;

                    last_event = train_service_kernels_.do_level_partition_by_groups(
                        ctx,
                        full_data_nd_,
                        node_list,
                        tree_order_lev_,
                        tree_order_lev_buf_,
                        ctx.row_count_,
                        ctx.selected_row_total_count_,
                        ctx.column_count_,
                        node_count,
                        ctx.tree_in_block_,
                        { last_event });
                }
            }

            node_count = node_count_new;
        }

        model_manager.add_tree_block(level_records, bin_borders_host_, iter_tree_count);

        for (Index tree_idx = 0; tree_idx < iter_tree_count; tree_idx++) {
            compute_results(model_manager,
                            data_host_,
                            response_host_,
                            oob_rows_list,
                            oob_rows_num_list,
                            oob_per_obs_list_,
                            res_var_imp_,
                            var_imp_variance_host_,
                            engine_arr,
                            tree_idx,
                            iter_tree_count,
                            iter,
                            ctx,
                            { last_event })
                .wait_and_throw();
        }
    }

    // Finalize results
    if (ctx.oob_err_required_ || ctx.oob_err_obs_required_) {
        pr::ndarray<Float, 1> res_oob_err;
        pr::ndarray<Float, 1> res_oob_err_obs;

        finalize_oob_error(response_host_, oob_per_obs_list_, res_oob_err, res_oob_err_obs, ctx)
            .wait_and_throw();

        if (ctx.oob_err_required_) {
            auto res_oob_err_host = res_oob_err.to_host(queue_);
            res.set_oob_err(homogen_table::wrap(res_oob_err_host.flatten(), 1, 1));
        }

        if (ctx.oob_err_obs_required_) {
            auto res_oob_err_obs_host = res_oob_err_obs.to_host(queue_);
            res.set_oob_err_per_observation(
                homogen_table::wrap(res_oob_err_obs_host.flatten(), ctx.row_count_, 1));
        }
    }

    if (ctx.mdi_required_ || ctx.mda_required_) {
        finalize_var_imp(res_var_imp_, var_imp_variance_host_, ctx).wait_and_throw();
        auto res_var_imp_host = res_var_imp_.to_host(queue_);
        res.set_var_importance(
            homogen_table::wrap(res_var_imp_host.flatten(), 1, ctx.column_count_));
    }

    return res.set_model(model_manager.get_model());
}

#define INSTANTIATE(F, B, I, T) template class train_kernel_hist_impl<F, B, I, T>;

//INSTANTIATE(ONEDAL_FLOAT, std::uint32_t, std::int32_t, task::classification);
//INSTANTIATE(ONEDAL_FLOAT, std::uint32_t, std::int32_t, task::regression);

} // namespace oneapi::dal::decision_forest::backend
