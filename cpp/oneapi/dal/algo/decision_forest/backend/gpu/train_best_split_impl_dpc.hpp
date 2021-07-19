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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_best_split_impl.hpp"

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

//////////////////////////////////////////// Best split kernels
template <typename Float>
inline bool float_eq(Float a, Float b) {
    return sycl::fabs(a - b) <= float_accuracy<Float>::val;
}

template <typename Float>
inline bool float_gt(Float a, Float b) {
    return (a - b) > float_accuracy<Float>::val;
}

template <typename Index>
inline void mark_bin_processed(std::uint64_t* bin_map, Index bin_idx) {
    std::uint64_t mask = 1ul << (bin_idx % 64);
    bin_map[bin_idx / 64] = bin_map[bin_idx / 64] & mask;
}

template <typename Index>
inline bool is_bin_processed(const std::uint64_t* bin_map, Index bin_idx) {
    std::uint64_t mask = 1ul << (bin_idx % 64);
    return bin_map[bin_idx / 64] & mask;
}

template <typename Float, typename Index, typename Task>
struct split_smp {
    using task_t = Task;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;

    // classififcation version
    inline void add_val(Index& left_count_,
                        hist_type_t* left_class_hist_,
                        Index ftr_bin_,
                        Index obs_bin,
                        Float obs_response) {
        Index class_id = static_cast<Index>(obs_response);

        left_count_ += Index(obs_bin <= ftr_bin_);
        left_class_hist_[class_id] += Index(obs_bin <= ftr_bin_);
    }

    inline void merge_bin_hist(Index& left_count_,
                               hist_type_t* left_class_hist_,
                               const hist_type_t* bin_hist_ptr,
                               Index class_count) {
        merge_stat(&left_class_hist_[0], &left_count_, bin_hist_ptr, class_count);
    }

    // regression version
    inline void add_val(hist_type_t* left_hist_,
                        Index ftr_bin_,
                        Index obs_bin,
                        Float obs_response) {
        if (obs_bin <= ftr_bin_) {
            add_val_to_hist<Float, Index>(&left_hist_[0], obs_response);
        }
    }

    inline void merge_bin_hist(hist_type_t* left_hist_,
                               const hist_type_t* bin_hist_ptr,
                               Index elem_count) {
        merge_stat(&left_hist_[0], bin_hist_ptr, elem_count);
    }

    // classififcation version
    inline void calc_imp_dec(Index& right_count_,
                             Float& left_imp_,
                             Float& right_imp_,
                             Float& imp_dec_,
                             Index left_count_,
                             const Index* node_ptr,
                             const Float* imp_list_ptr_,
                             const hist_type_t* node_class_hist_list_ptr_,
                             const hist_type_t* left_class_hist_,
                             Index class_count,
                             Index node_id) {
        Index node_row_count = node_ptr[impl_const_t::ind_grc];
        const Float* node_imp_ptr = imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float node_imp = node_imp_ptr[0];
        const Index* node_class_hist_ptr = node_class_hist_list_ptr_ + node_id * class_count;

        right_count_ = node_row_count - left_count_;

        const Float divL =
            (0 < left_count_) ? Float(1) / (Float(left_count_) * Float(left_count_)) : Float(0);
        const Float divR =
            (0 < right_count_) ? Float(1) / (Float(right_count_) * Float(right_count_)) : Float(0);

        left_imp_ = Float(1);
        right_imp_ = Float(1);

        for (Index class_id = 0; class_id < class_count; class_id++) {
            left_imp_ -=
                Float(left_class_hist_[class_id]) * Float(left_class_hist_[class_id]) * divL;
            right_imp_ -= Float(node_class_hist_ptr[class_id] - left_class_hist_[class_id]) *
                          Float(node_class_hist_ptr[class_id] - left_class_hist_[class_id]) * divR;
        }

        left_imp_ = sycl::max(left_imp_, Float(0));
        right_imp_ = sycl::max(right_imp_, Float(0));

        imp_dec_ = node_imp - (Float(left_count_) * left_imp_ + Float(right_count_) * right_imp_) /
                                  Float(node_row_count);
    }

    // regression version
    inline void calc_imp_dec(Index& left_count_,
                             Index& right_count_,
                             Float& imp_dec_,
                             const Index* node_ptr,
                             const Float* imp_list_ptr,
                             const hist_type_t* left_hist_,
                             Index node_id) {
        constexpr Index buff_size = impl_const_t::node_imp_prop_count_ + 1;
        Index node_row_count = node_ptr[impl_const_t::ind_grc];
        left_count_ = static_cast<Index>(left_hist_[0]);

        const Float* node_imp_ptr = imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;

        hist_type_t node_hist[buff_size] = { static_cast<Float>(node_row_count),
                                             node_imp_ptr[0],
                                             node_imp_ptr[1] };
        hist_type_t right_hist[buff_size] = { hist_type_t(0) };

        // getting hist for right part
        sub_stat<Float, Index, task_t>(&right_hist[0], &left_hist_[0], &node_hist[0], buff_size);

        right_count_ = node_row_count - left_count_;
        imp_dec_ = node_imp_ptr[1] - (left_hist_[2] + right_hist[2]);
    }

    inline bool test_split_is_best(Index bs_ftr_id_,
                                   Index bs_ftr_bin_,
                                   Index bs_left_count_,
                                   Float bs_imp_dec_,
                                   Index ts_ftr_id_,
                                   Index ts_ftr_bin_,
                                   Index ts_left_count_,
                                   Index ts_right_count_,
                                   Float ts_imp_dec_,
                                   Index node_id,
                                   Float node_imp,
                                   Float impurity_threshold_,
                                   Index min_observations_in_leaf_node_) {
        return (Float(0) < ts_imp_dec_ && !float_eq(node_imp, Float(0)) &&
                node_imp >= impurity_threshold_ &&
                (bs_ftr_bin_ == impl_const_t::leaf_mark_ || float_gt(ts_imp_dec_, bs_imp_dec_) ||
                 (float_eq(ts_imp_dec_, bs_imp_dec_) &&
                  (ts_ftr_id_ < bs_ftr_id_ ||
                   (bs_ftr_id_ == ts_ftr_id_ && ts_ftr_bin_ < bs_ftr_bin_)))) &&
                ts_left_count_ >= min_observations_in_leaf_node_ &&
                ts_right_count_ >= min_observations_in_leaf_node_);
    }

    // classififcation version
    inline void choose_best_split(Index& bs_ftr_id_,
                                  Index& bs_ftr_bin_,
                                  Index& bs_left_count_,
                                  Float& bs_left_imp_,
                                  Float& bs_imp_dec_,
                                  hist_type_t* bs_left_class_hist_,
                                  Index ts_ftr_id_,
                                  Index ts_ftr_bin_,
                                  Index ts_left_count_,
                                  Index ts_right_count_,
                                  Float ts_left_imp_,
                                  Float ts_imp_dec_,
                                  const hist_type_t* ts_left_class_hist_,
                                  const Float* node_imp_list_ptr,
                                  Index class_count_,
                                  Index node_id,
                                  Float impurity_threshold_,
                                  Index min_observations_in_leaf_node_) {
        // TODO move check for imp 0 to node split func
        const Float* node_imp_ptr =
            node_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;
        Float node_imp = node_imp_ptr[0];

        if (test_split_is_best(bs_ftr_id_,
                               bs_ftr_bin_,
                               bs_left_count_,
                               bs_imp_dec_,
                               ts_ftr_id_,
                               ts_ftr_bin_,
                               ts_left_count_,
                               ts_right_count_,
                               ts_imp_dec_,
                               node_id,
                               node_imp,
                               impurity_threshold_,
                               min_observations_in_leaf_node_)) {
            bs_ftr_id_ = ts_ftr_id_;
            bs_ftr_bin_ = ts_ftr_bin_;
            bs_imp_dec_ = ts_imp_dec_;

            bs_left_count_ = ts_left_count_;
            bs_left_imp_ = ts_left_imp_;
            for (Index class_id = 0; class_id < class_count_; class_id++) {
                bs_left_class_hist_[class_id] = ts_left_class_hist_[class_id];
            }
        }
    }

    // regression version
    inline void choose_best_split(Index& bs_ftr_id_,
                                  Index& bs_ftr_bin_,
                                  Index& bs_left_count_,
                                  Float& bs_imp_dec_,
                                  hist_type_t* bs_left_hist_,
                                  Index ts_ftr_id_,
                                  Index ts_ftr_bin_,
                                  Index ts_left_count_,
                                  Index ts_right_count_,
                                  Float ts_imp_dec_,
                                  const hist_type_t* ts_left_hist_,
                                  const Float* node_imp_list_ptr,
                                  Index hist_elem_count,
                                  Index node_id,
                                  Float impurity_threshold_,
                                  Index min_observations_in_leaf_node_) {
        // TODO move check for imp 0 to node split func
        const Float* node_imp_ptr =
            node_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;
        Float node_imp = node_imp_ptr[1];
        node_imp = node_imp / Float(ts_left_count_ + ts_right_count_);

        if (test_split_is_best(bs_ftr_id_,
                               bs_ftr_bin_,
                               bs_left_count_,
                               bs_imp_dec_,
                               ts_ftr_id_,
                               ts_ftr_bin_,
                               ts_left_count_,
                               ts_right_count_,
                               ts_imp_dec_,
                               node_id,
                               node_imp,
                               impurity_threshold_,
                               min_observations_in_leaf_node_)) {
            bs_ftr_id_ = ts_ftr_id_;
            bs_ftr_bin_ = ts_ftr_bin_;
            bs_imp_dec_ = ts_imp_dec_;

            bs_left_count_ = ts_left_count_;

            for (Index i = 0; i < hist_elem_count; i++) {
                bs_left_hist_[i] = ts_left_hist_[i];
            }
        }
    }

    // classififcation version
    inline void update_left_child_imp(Float* left_imp_list_ptr,
                                      hist_type_t* left_class_hist_list_ptr,
                                      Float bs_left_imp_,
                                      const hist_type_t* bs_left_class_hist_,
                                      Index node_id,
                                      Index class_count) {
        Float* left_node_imp_ptr = left_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;

        left_node_imp_ptr[0] = bs_left_imp_;
        hist_type_t* left_node_class_hist_ptr = left_class_hist_list_ptr + node_id * class_count;

        for (Index class_id = 0; class_id < class_count; class_id++) {
            left_node_class_hist_ptr[class_id] = bs_left_class_hist_[class_id];
        }
    }

    // regression version
    inline void update_left_child_imp(Float* left_imp_list_ptr,
                                      const hist_type_t* bs_left_hist,
                                      Index node_id) {
        Float* left_node_imp_ptr = left_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;

        left_node_imp_ptr[0] = bs_left_hist[1];
        left_node_imp_ptr[1] = bs_left_hist[2];
    }

    inline bool my_split_is_best_for_sbg(sycl::nd_item<2>& item,
                                         Index& bs_ftr_id_,
                                         Index& bs_ftr_bin_,
                                         Index& bs_left_count_,
                                         Float& bs_imp_dec_,
                                         Index* node_ptr,
                                         Float* node_imp_decr_ptr,
                                         Index node_id,
                                         Index index_max,
                                         bool update_imp_dec_required) {
        auto sbg = item.get_sub_group();

        const Index sub_group_local_id = sbg.get_local_id();
        const Index valNotFound = index_max;

        const Float bestImpDec = reduce(sbg, bs_imp_dec_, maximum<Float>());

        const Index impDecIsBest = float_eq(bestImpDec, bs_imp_dec_);

        const Index bestFeatureId =
            reduce(sbg, impDecIsBest ? bs_ftr_id_ : valNotFound, minimum<Index>());
        const Index bestFeatureValue =
            reduce(sbg,
                   (bestFeatureId == bs_ftr_id_ && impDecIsBest) ? bs_ftr_bin_ : valNotFound,
                   minimum<Index>());

        const bool noneSplitFoundBySubGroup =
            ((impl_const_t::leaf_mark_ == bestFeatureId) && (0 == sub_group_local_id));
        const bool mySplitIsBest = (impl_const_t::leaf_mark_ != bestFeatureId &&
                                    bs_ftr_id_ == bestFeatureId && bs_ftr_bin_ == bestFeatureValue);
        if (noneSplitFoundBySubGroup || mySplitIsBest) {
            node_ptr[impl_const_t::ind_fid] =
                bs_ftr_id_ == valNotFound ? impl_const_t::leaf_mark_ : bs_ftr_id_;
            node_ptr[impl_const_t::ind_bin] =
                bs_ftr_bin_ == valNotFound ? impl_const_t::leaf_mark_ : bs_ftr_bin_;
            node_ptr[impl_const_t::ind_lch_grc] = bs_left_count_;

            if (update_imp_dec_required) {
                if constexpr (std::is_same_v<task_t, task::classification>) {
                    node_imp_decr_ptr[0] = bs_imp_dec_;
                }
                else {
                    node_imp_decr_ptr[0] = bs_imp_dec_ / node_ptr[impl_const_t::ind_grc];
                }
            }

            return true;
        }
        return false;
    }

    // classififcation version
    inline void choose_best_split_for_sbg(sycl::nd_item<2>& item,
                                          Index& bs_ftr_id_,
                                          Index& bs_ftr_bin_,
                                          Index& bs_left_count_,
                                          Float& bs_left_imp_,
                                          Float& bs_imp_dec_,
                                          hist_type_t* bs_left_class_hist_,
                                          Index* node_ptr,
                                          Float* node_imp_decr_ptr,
                                          Float* left_imp_list_ptr,
                                          hist_type_t* left_class_hist_list_ptr,
                                          Index node_id,
                                          Index index_max,
                                          Index class_count,
                                          bool update_imp_dec_required) {
        auto sbg = item.get_sub_group();
        if (sbg.get_group_id() > 0) {
            return;
        }

        if (my_split_is_best_for_sbg(item,
                                     bs_ftr_id_,
                                     bs_ftr_bin_,
                                     bs_left_count_,
                                     bs_imp_dec_,
                                     node_ptr,
                                     node_imp_decr_ptr,
                                     node_id,
                                     index_max,
                                     update_imp_dec_required)) {
            update_left_child_imp(left_imp_list_ptr,
                                  left_class_hist_list_ptr,
                                  bs_left_imp_,
                                  bs_left_class_hist_,
                                  node_id,
                                  class_count);
        }
    }

    // regression version
    inline void choose_best_split_for_sbg(sycl::nd_item<2>& item,
                                          Index& bs_ftr_id_,
                                          Index& bs_ftr_bin_,
                                          Index& bs_left_count_,
                                          Float& bs_imp_dec_,
                                          hist_type_t* bs_left_hist_,
                                          Index* node_ptr,
                                          Float* node_imp_decr_ptr,
                                          Float* left_imp_list_ptr,
                                          Index node_id,
                                          Index index_max,
                                          bool update_imp_dec_required) {
        auto sbg = item.get_sub_group();
        if (sbg.get_group_id() > 0) {
            return;
        }

        if (my_split_is_best_for_sbg(item,
                                     bs_ftr_id_,
                                     bs_ftr_bin_,
                                     bs_left_count_,
                                     bs_imp_dec_,
                                     node_ptr,
                                     node_imp_decr_ptr,
                                     node_id,
                                     index_max,
                                     update_imp_dec_required)) {
            update_left_child_imp(left_imp_list_ptr, bs_left_hist_, node_id);
        }
    }
};

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_best_split_impl<Float, Bin, Index, Task>::compute_best_split_by_histogram(
    const context_t& ctx,
    const pr::ndarray<hist_type_t, 1>& node_hist_list,
    const pr::ndarray<Index, 1>& selected_ftr_list,
    const pr::ndarray<Index, 1>& bin_offset_list,
    const imp_data_t& imp_data_list,
    const pr::ndarray<Index, 1>& node_ind_list,
    Index node_ind_ofs,
    pr::ndarray<Index, 1>& node_list,
    imp_data_t& left_child_imp_data_list,
    pr::ndarray<Float, 1>& node_imp_dec_list,
    bool update_imp_dec_required,
    Index node_count,
    const be::event_vector& deps) {
    using split_smp_t = split_smp<Float, Index, Task>;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;
    // input asserts is going to be added
    const hist_type_t* node_hist_list_ptr = node_hist_list.get_data();
    const Index* selected_ftr_list_ptr = selected_ftr_list.get_data();

    const Index* bin_offset_list_ptr = bin_offset_list.get_data();
    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    const Index* node_indices_ptr = node_ind_list.get_data();
    Index* node_list_ptr = node_list.get_mutable_data();
    Float* node_imp_decr_list_ptr = node_imp_dec_list.get_mutable_data();

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;
    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const_t::hist_prop_count_;
    }

    const Index selected_ftr_count = ctx.selected_ftr_count_;

    constexpr Index buff_size = impl_const_t::private_hist_buff_size;

    const Index class_count = ctx.class_count_;
    const Float imp_threshold = ctx.impurity_threshold_;
    const Index min_obs_leaf = ctx.min_observations_in_leaf_node_;
    const Index index_max = ctx.index_max_;

    const Float min_imp_dec = de::limits<Float>::min();

    const Float* node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    Float* left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

    // following vars are not used for regression, but should present to compile kernel
    const Index* class_hist_list_ptr = imp_list_ptr.get_class_hist_list_ptr_or_null();
    Index* left_child_class_hist_list_ptr = left_imp_list_ptr.get_class_hist_list_ptr_or_null();

    //const kernel_context<Float, Index, Task> krn_ctx(ctx);

    auto local_size = be::device_max_sg_size(queue_);

    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    sycl::event last_event;

    last_event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }
            const Index node_idx = item.get_global_id()[1];
            const Index node_id = node_indices_ptr[node_ind_ofs + node_idx];
            Index* node_ptr = node_list_ptr + node_id * impl_const_t::node_prop_count_;

            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];

            split_smp_t sp_hlp;

            Index bs_ftr_bin = impl_const_t::leaf_mark_;
            Index bs_ftr_id = impl_const_t::leaf_mark_;
            Index bs_left_count = 0;
            hist_type_t bs_left_hist[buff_size] = { hist_type_t(0) };

            Float bs_left_imp = Float(0);
            Float bs_imp_dec = min_imp_dec;

            for (Index ftr_idx = sub_group_local_id; ftr_idx < selected_ftr_count;
                 ftr_idx += sub_group_size) {
                const Index ts_ftr_id =
                    selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];
                const Index ts_ftr_bin_count =
                    bin_offset_list_ptr[ts_ftr_id + 1] - bin_offset_list_ptr[ts_ftr_id];

                const hist_type_t* node_hist_ptr =
                    node_hist_list_ptr +
                    node_idx * selected_ftr_count * max_bin_count_among_ftrs * hist_prop_count;
                const hist_type_t* ftr_hist_ptr =
                    node_hist_ptr + ftr_idx * max_bin_count_among_ftrs * hist_prop_count;

                Index ts_left_count = 0;
                Index ts_right_count = 0;
                hist_type_t ts_left_hist[buff_size] = { 0 };

                Float ts_left_imp = Float(0);
                Float ts_right_imp = Float(0);
                Float ts_imp_dec = Float(0);

                for (Index ts_ftr_bin = 0; ts_ftr_bin < ts_ftr_bin_count; ts_ftr_bin++) {
                    const Index bin_ofs = ts_ftr_bin * hist_prop_count;

                    if constexpr (std::is_same_v<Task, task::classification>) {
                        sp_hlp.merge_bin_hist(ts_left_count,
                                              &ts_left_hist[0],
                                              ftr_hist_ptr + bin_ofs,
                                              hist_prop_count);
                    }
                    else {
                        sp_hlp.merge_bin_hist(&ts_left_hist[0],
                                              ftr_hist_ptr + bin_ofs,
                                              hist_prop_count);
                    }

                    if constexpr (std::is_same_v<Task, task::classification>) {
                        sp_hlp.calc_imp_dec(ts_right_count,
                                            ts_left_imp,
                                            ts_right_imp,
                                            ts_imp_dec,
                                            ts_left_count,
                                            node_ptr,
                                            node_imp_list_ptr,
                                            class_hist_list_ptr,
                                            &ts_left_hist[0],
                                            class_count,
                                            node_id);

                        sp_hlp.choose_best_split(bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_left_imp,
                                                 bs_imp_dec,
                                                 &bs_left_hist[0],
                                                 ts_ftr_id,
                                                 ts_ftr_bin,
                                                 ts_left_count,
                                                 ts_right_count,
                                                 ts_left_imp,
                                                 ts_imp_dec,
                                                 &ts_left_hist[0],
                                                 node_imp_list_ptr,
                                                 class_count,
                                                 node_id,
                                                 imp_threshold,
                                                 min_obs_leaf);
                    }
                    else {
                        sp_hlp.calc_imp_dec(ts_left_count,
                                            ts_right_count,
                                            ts_imp_dec,
                                            node_ptr,
                                            node_imp_list_ptr,
                                            &ts_left_hist[0],
                                            node_id);

                        sp_hlp.choose_best_split(bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_imp_dec,
                                                 &bs_left_hist[0],
                                                 ts_ftr_id,
                                                 ts_ftr_bin,
                                                 ts_left_count,
                                                 ts_right_count,
                                                 ts_imp_dec,
                                                 &ts_left_hist[0],
                                                 node_imp_list_ptr,
                                                 buff_size,
                                                 node_id,
                                                 imp_threshold,
                                                 min_obs_leaf);
                    }
                }
            }

            if constexpr (std::is_same_v<Task, task::classification>) {
                sp_hlp.choose_best_split_for_sbg(item,
                                                 bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_left_imp,
                                                 bs_imp_dec,
                                                 &bs_left_hist[0],
                                                 node_ptr,
                                                 node_imp_decr_list_ptr,
                                                 left_child_imp_list_ptr,
                                                 left_child_class_hist_list_ptr,
                                                 node_id,
                                                 index_max,
                                                 class_count,
                                                 update_imp_dec_required);
            }
            else {
                sp_hlp.choose_best_split_for_sbg(item,
                                                 bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_imp_dec,
                                                 &bs_left_hist[0],
                                                 node_ptr,
                                                 node_imp_decr_list_ptr,
                                                 left_child_imp_list_ptr,
                                                 node_id,
                                                 index_max,
                                                 update_imp_dec_required);
            }
        });
    });

    last_event.wait_and_throw();

    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_best_split_impl<Float, Bin, Index, Task>::compute_best_split_single_pass(
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    const pr::ndarray<Index, 1>& selected_ftr_list,
    const pr::ndarray<Index, 1>& bin_offset_list,
    const imp_data_t& imp_data_list,
    const pr::ndarray<Index, 1>& node_ind_list,
    Index node_ind_ofs,
    pr::ndarray<Index, 1>& node_list,
    imp_data_t& left_child_imp_data_list,
    pr::ndarray<Float, 1>& node_imp_dec_list,
    bool update_imp_dec_required,
    Index node_count,
    const be::event_vector& deps) {
    using split_smp_t = split_smp<Float, Index, Task>;

    const Bin* data_ptr = data.get_data();
    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = tree_order.get_data();

    const Index* selected_ftr_list_ptr = selected_ftr_list.get_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    const Index* node_indices_ptr = node_ind_list.get_data();
    Index* node_list_ptr = node_list.get_mutable_data();
    Float* node_imp_decr_list_ptr = node_imp_dec_list.get_mutable_data();

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    const Index column_count = ctx.column_count_;

    const Index selected_ftr_count = ctx.selected_ftr_count_;

    const kernel_context<Float, Index, Task> krn_ctx(ctx);

    const Index index_max = ctx.index_max_;

    auto local_size = be::device_max_sg_size(queue_);

    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    sycl::event last_event;

    const Index class_count = ctx.class_count_;
    const Float imp_threshold = ctx.impurity_threshold_;
    const Index min_obs_leaf = ctx.min_observations_in_leaf_node_;

    const Float min_imp_dec = de::limits<Float>::min();

    const Float* node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    Float* left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

    // following vars are not used for regression, but should present to compile kernel
    const Index* class_hist_list_ptr = imp_list_ptr.get_class_hist_list_ptr_or_null();
    Index* left_child_class_hist_list_ptr = left_imp_list_ptr.get_class_hist_list_ptr_or_null();

    constexpr Index buff_size = impl_const_t::private_hist_buff_size;

    last_event = queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](cl::sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }
            const Index node_idx = item.get_global_id()[1];
            const Index node_id = node_indices_ptr[node_ind_ofs + node_idx];
            Index* node_ptr = node_list_ptr + node_id * impl_const_t::node_prop_count_;

            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];

            const Index row_ofs = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];

            split_smp_t sp_hlp;

            Index bs_ftr_bin = impl_const_t::leaf_mark_;
            Index bs_ftr_id = impl_const_t::leaf_mark_;
            Index bs_left_count = 0;
            hist_type_t bs_left_hist[buff_size] = { hist_type_t(0) };

            Float bs_left_imp = Float(0);
            Float bs_imp_dec = min_imp_dec;

            for (Index ftr_idx = sub_group_local_id; ftr_idx < selected_ftr_count;
                 ftr_idx += sub_group_size) {
                const Index ts_ftr_id =
                    selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];

                std::uint64_t bin_map[4] = { 0 };

                // calculating classes histogram rows count <= bins num
                for (Index i = 0; i < row_count; i++) {
                    Index curr_row_id = tree_order_ptr[row_ofs + i];
                    Index ts_ftr_bin = data_ptr[curr_row_id * column_count + ts_ftr_id];

                    bool bin_not_processed = !is_bin_processed(bin_map, ts_ftr_bin);
                    if (bin_not_processed) {
                        Index ts_left_count = 0;
                        Index ts_right_count = 0;
                        hist_type_t ts_left_hist[buff_size] = { 0 };

                        Float ts_left_imp = Float(0);
                        Float ts_right_imp = Float(0);
                        Float ts_imp_dec = Float(0);

                        for (int row_idx = 0; row_idx < row_count; row_idx++) {
                            Index id = tree_order_ptr[row_ofs + row_idx];
                            Index bin = data_ptr[id * column_count + ts_ftr_id];

                            if constexpr (std::is_same_v<Task, task::classification>) {
                                sp_hlp.add_val(ts_left_count,
                                               &ts_left_hist[0],
                                               ts_ftr_bin,
                                               bin,
                                               response_ptr[id]);
                            }
                            else {
                                sp_hlp.add_val(&ts_left_hist[0], ts_ftr_bin, bin, response_ptr[id]);
                            }
                        }

                        mark_bin_processed(bin_map, ts_ftr_bin);

                        if constexpr (std::is_same_v<Task, task::classification>) {
                            sp_hlp.calc_imp_dec(ts_right_count,
                                                ts_left_imp,
                                                ts_right_imp,
                                                ts_imp_dec,
                                                ts_left_count,
                                                node_ptr,
                                                node_imp_list_ptr,
                                                class_hist_list_ptr,
                                                &ts_left_hist[0],
                                                class_count,
                                                node_id);

                            sp_hlp.choose_best_split(bs_ftr_id,
                                                     bs_ftr_bin,
                                                     bs_left_count,
                                                     bs_left_imp,
                                                     bs_imp_dec,
                                                     &bs_left_hist[0],
                                                     ts_ftr_id,
                                                     ts_ftr_bin,
                                                     ts_left_count,
                                                     ts_right_count,
                                                     ts_left_imp,
                                                     ts_imp_dec,
                                                     &ts_left_hist[0],
                                                     node_imp_list_ptr,
                                                     class_count,
                                                     node_id,
                                                     imp_threshold,
                                                     min_obs_leaf);
                        }
                        else {
                            sp_hlp.calc_imp_dec(ts_left_count,
                                                ts_right_count,
                                                ts_imp_dec,
                                                node_ptr,
                                                node_imp_list_ptr,
                                                &ts_left_hist[0],
                                                node_id);

                            sp_hlp.choose_best_split(bs_ftr_id,
                                                     bs_ftr_bin,
                                                     bs_left_count,
                                                     bs_imp_dec,
                                                     &bs_left_hist[0],
                                                     ts_ftr_id,
                                                     ts_ftr_bin,
                                                     ts_left_count,
                                                     ts_right_count,
                                                     ts_imp_dec,
                                                     &ts_left_hist[0],
                                                     node_imp_list_ptr,
                                                     buff_size,
                                                     node_id,
                                                     imp_threshold,
                                                     min_obs_leaf);
                        }
                    }
                }
            }

            if constexpr (std::is_same_v<Task, task::classification>) {
                sp_hlp.choose_best_split_for_sbg(item,
                                                 bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_left_imp,
                                                 bs_imp_dec,
                                                 &bs_left_hist[0],
                                                 node_ptr,
                                                 node_imp_decr_list_ptr,
                                                 left_child_imp_list_ptr,
                                                 left_child_class_hist_list_ptr,
                                                 node_id,
                                                 index_max,
                                                 class_count,
                                                 update_imp_dec_required);
            }
            else {
                sp_hlp.choose_best_split_for_sbg(item,
                                                 bs_ftr_id,
                                                 bs_ftr_bin,
                                                 bs_left_count,
                                                 bs_imp_dec,
                                                 &bs_left_hist[0],
                                                 node_ptr,
                                                 node_imp_decr_list_ptr,
                                                 left_child_imp_list_ptr,
                                                 node_id,
                                                 index_max,
                                                 update_imp_dec_required);
            }
        });
    });

    last_event.wait_and_throw();

    return last_event;
}

#define INSTANTIATE(F, B, I, T) template class train_best_split_impl<F, B, I, T>;

} // namespace oneapi::dal::decision_forest::backend
