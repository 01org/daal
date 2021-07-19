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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_service_kernels.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#ifdef ONEDAL_DATA_PARALLEL

#include <CL/sycl/ONEAPI/experimental/builtins.hpp>

#endif

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace be = dal::backend;
namespace pr = dal::backend::primitives;

using sycl::ONEAPI::broadcast;
using sycl::ONEAPI::reduce;
using sycl::ONEAPI::plus;
using sycl::ONEAPI::minimum;
using sycl::ONEAPI::maximum;
using sycl::ONEAPI::exclusive_scan;

using alloc = sycl::usm::alloc;
using address = sycl::access::address_space;

template <typename T>
inline T atomic_global_add(T* ptr, T operand) {
    return sycl::atomic_fetch_add<T, address::global_space>(
        { sycl::multi_ptr<T, address::global_space>{ ptr } },
        operand);
}

template <typename Float, typename Bin, typename Index, typename Task>
std::uint64_t train_service_kernels<Float, Bin, Index, Task>::get_oob_rows_required_mem_size(
    Index row_count,
    Index tree_count,
    double observations_per_tree_fraction) {
    // mem size occupied on GPU for storing OOB rows indices
    const std::uint64_t oob_rows_aprox_count =
        row_count * (Float(1) - observations_per_tree_fraction) +
        row_count * observations_per_tree_fraction * aproximate_oob_rows_fraction_;
    return sizeof(Index) * oob_rows_aprox_count * tree_count;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::initialize_tree_order(
    pr::ndarray<Index, 1>& tree_order,
    Index tree_count,
    Index row_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(tree_order.get_count() == tree_count * row_count);

    Index* tree_order_ptr = tree_order.get_mutable_data();
    const sycl::range<2> range{ de::integral_cast<size_t>(row_count),
                                de::integral_cast<size_t>(tree_count) };

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<2> id) {
            tree_order_ptr[id[1] * row_count + id[0]] = id[0];
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::split_node_list_on_groups_by_size(
    const context_t& ctx,
    const dal::backend::primitives::ndarray<Index, 1>& node_list,
    dal::backend::primitives::ndarray<Index, 1>& node_groups,
    dal::backend::primitives::ndarray<Index, 1>& node_indices,
    Index node_count,
    Index group_count,
    Index group_prop_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(node_groups.get_count() == group_count * group_prop_count);
    ONEDAL_ASSERT(node_indices.get_count() == node_count);

    const Index big_node_low_border_blocks_num = big_node_low_border_blocks_num_;
    const Index block_size = min_rows_block_;
    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node

    const Index* node_list_ptr = node_list.get_data();
    Index* node_groups_ptr = node_groups.get_mutable_data();
    Index* node_indices_ptr = node_indices.get_mutable_data();

    auto local_size = preferable_sbg_size_;
    const sycl::nd_range<1> nd_range = be::make_multiple_nd_range_1d(local_size, local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }

            //for now only 3 groups are produced, may be more required
            const Index local_id = sbg.get_local_id();
            const Index local_size = sbg.get_local_range()[0];

            Index big_node_count = 0;
            Index max_big_block_count = 1;
            Index mid_node_count = 0;
            Index max_mid_block_count = 1;

            // calculate num of big and mid nodes
            for (Index i = local_id; i < node_count; i += local_size) {
                Index row_count = node_list_ptr[i * node_prop_count + impl_const_t::ind_grc];
                Index block_count = row_count / block_size + bool(row_count % block_size);

                Index big_node = (Index)(block_count > big_node_low_border_blocks_num);
                Index mid_node =
                    (Index)(block_count <= big_node_low_border_blocks_num && block_count > 1);

                big_node_count += reduce(sbg, big_node, plus<Index>());
                mid_node_count += reduce(sbg, mid_node, plus<Index>());
                max_big_block_count =
                    sycl::max(max_big_block_count, static_cast<Index>(big_node ? block_count : 0));
                max_big_block_count = reduce(sbg, max_big_block_count, maximum<Index>());
                max_mid_block_count =
                    sycl::max(max_mid_block_count, static_cast<Index>(mid_node ? block_count : 0));
                max_mid_block_count = reduce(sbg, max_mid_block_count, maximum<Index>());
            }

            big_node_count = broadcast(sbg, big_node_count, 0);
            mid_node_count = broadcast(sbg, mid_node_count, 0);

            if (0 == local_id) {
                node_groups_ptr[0] = big_node_count;
                node_groups_ptr[1] = max_big_block_count;
                node_groups_ptr[2] = mid_node_count;
                node_groups_ptr[3] = max_mid_block_count;
                node_groups_ptr[4] = node_count - big_node_count - mid_node_count;
                node_groups_ptr[5] = 1;
            }

            Index sum_big = 0;
            Index sum_mid = 0;

            //split nodes on groups
            for (Index i = local_id; i < node_count; i += local_size) {
                Index row_count = node_list_ptr[i * node_prop_count + impl_const_t::ind_grc];
                Index block_count = row_count / block_size + bool(row_count % block_size);
                Index big_node = (Index)(block_count > big_node_low_border_blocks_num);
                Index mid_node =
                    (Index)(block_count <= big_node_low_border_blocks_num && block_count > 1);

                Index boundary_big = sum_big + exclusive_scan(sbg, big_node, plus<Index>());
                Index boundary_mid = sum_mid + exclusive_scan(sbg, mid_node, plus<Index>());
                Index pos_new = (big_node ? boundary_big
                                          : (mid_node ? big_node_count + boundary_mid
                                                      : big_node_count + mid_node_count + i -
                                                            boundary_big - boundary_mid));
                node_indices_ptr[pos_new] = i;
                sum_big += reduce(sbg, big_node, plus<Index>());
                sum_mid += reduce(sbg, mid_node, plus<Index>());
            }
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::get_split_node_count(
    const dal::backend::primitives::ndarray<Index, 1>& node_list,
    Index node_count,
    Index& split_node_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);

    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index bad_val = impl_const_t::bad_val_;

    const Index* node_list_ptr = node_list.get_data();
    auto split_node_count_buf = pr::ndarray<Index, 1>::empty(queue_, { 1 }, alloc::device);
    Index* split_node_count_ptr = split_node_count_buf.get_mutable_data();

    auto local_size = preferable_sbg_size_;
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

            Index sum = 0;
            for (Index i = local_id; i < node_count; i += local_size) {
                sum += Index(node_list_ptr[i * node_prop_count + impl_const_t::ind_fid] != bad_val);
            }

            sum = reduce(sbg, sum, plus<Index>());

            if (local_id == 0) {
                split_node_count_ptr[0] = sum;
            }
        });
    });

    auto split_node_count_host = split_node_count_buf.to_host(queue_, { event });
    split_node_count = split_node_count_host.get_data()[0];

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event
train_service_kernels<Float, Bin, Index, Task>::calculate_left_child_row_count_on_local_data(
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& tree_order,
    Index column_count,
    Index node_count,
    const be::event_vector& deps) {
    // this function is used for ditributed mode only, because for batch it is claculated during bs
    ONEDAL_ASSERT(ctx.distr_mode_);

    //DAAL_ASSERT_UNIVERSAL_BUFFER(data, uint32_t, nRows * nFeatures);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(nodeList, int32_t, nNodes * _nNodeProps);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrder, int32_t, nRows);
    //DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrderBuf, int32_t, nRows);

    // nNodes * _partitionMaxBlocksNum is used inside kernel
    //DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(int32_t, nNodes, _partitionMaxBlocksNum);

    // nodeAuxList is auxilliary buffer for synchronization of left and right boundaries of blocks (nElemsToLeft, nElemsToRight)
    // processed by subgroups in the same node
    // no mul overflow check is required due to there is already buffer of size nNodes * _nNodeProps
    ONEDAL_ASSERT(aux_node_buffer_prop_count_ <= impl_const_t::node_prop_count_);

    const Index nNodeProp = impl_const_t::node_prop_count_; // num of split attributes for node
    const Index leafMark = impl_const_t::leaf_mark_;
    const Index maxBlocksNum = partition_max_block_count_;
    const Index minBlockSize = partition_min_block_size_;

    const Bin* data_ptr = data.get_data();
    Index* node_list_ptr = node_list.get_mutable_data();
    const Index* tree_order_ptr = tree_order.get_data();

    auto local_size = preferable_partition_group_size_;
    const sycl::nd_range<1> nd_range =
        be::make_multiple_nd_range_1d(preferable_partition_groups_count_ * local_size, local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            const Index sub_group_size = sbg.get_local_range()[0];
            const Index work_group_size = item.get_local_range()[0];
            const Index sub_groups_in_work_group_num =
                work_group_size / sub_group_size; // num of subgroups for current node processing

            const Index sub_group_local_id = sbg.get_local_id();
            const Index work_group_local_id = item.get_local_id()[0];

            const Index sub_group_id = item.get_group().get_id(0) * sub_groups_in_work_group_num +
                                       work_group_local_id / sub_group_size;
            const Index sub_groups_num =
                item.get_group_range(0) *
                sub_groups_in_work_group_num; // num of subgroups for current node processing

            const Index totalBlocksNum = node_count * maxBlocksNum;

            for (Index blockIndGlob = sub_group_id; blockIndGlob < totalBlocksNum;
                 blockIndGlob += sub_groups_num) {
                const Index nodeId = blockIndGlob / maxBlocksNum;
                const Index blockInd = blockIndGlob % maxBlocksNum;

                Index* node = node_list_ptr + nodeId * nNodeProp;
                const Index offset = node[impl_const_t::ind_ofs];
                const Index nRows = node[impl_const_t::ind_lrc];
                const Index featId = node[impl_const_t::ind_fid];
                const Index splitVal = node[impl_const_t::ind_bin];

                Index nodeBlocks =
                    nRows / minBlockSize ? sycl::min(nRows / minBlockSize, maxBlocksNum) : 1;

                // if blockInd assigned for this sbg less than current node's block count -> sbg will just go to the next node
                if (featId != leafMark && blockInd < nodeBlocks) // split node
                {
                    const Index blockSize =
                        nodeBlocks > 1 ? nRows / nodeBlocks + bool(nRows % nodeBlocks) : nRows;

                    const Index iEnd = sycl::min((blockInd + 1) * blockSize, nRows);
                    const Index iStart = sycl::min(blockInd * blockSize, iEnd);
                    const Index rowsForGroup = iEnd - iStart;

                    if (rowsForGroup > 0) {
                        Index groupRowsToRight = 0;
                        for (Index i = iStart + sub_group_local_id; i < iEnd; i += sub_group_size) {
                            const Index id = tree_order_ptr[offset + i];
                            const Index toRight =
                                Index(static_cast<Index>(data_ptr[id * column_count + featId]) >
                                      splitVal);
                            groupRowsToRight += reduce(sbg, toRight, plus<Index>());
                        }

                        if (0 == sub_group_local_id) {
                            atomic_global_add(node + impl_const_t::ind_lch_lrc,
                                              rowsForGroup - groupRowsToRight);
                        }
                    }
                }
            }
        });
    });

    event.wait_and_throw();

    return event;
}
template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::do_level_partition_by_groups(
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndarray<Index, 1>& node_list,
    pr::ndarray<Index, 1>& tree_order,
    pr::ndarray<Index, 1>& tree_order_buf,
    Index data_row_count,
    Index data_selected_row_count,
    Index data_column_count,
    Index node_count,
    Index tree_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(data.get_count() == data_row_count * data_column_count);
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(tree_order.get_count() == data_selected_row_count * tree_count);
    ONEDAL_ASSERT(tree_order_buf.get_count() == data_selected_row_count * tree_count);

    // node_count * partition_max_block_count_ is used inside kernel
    de::check_mul_overflow(node_count, partition_max_block_count_);

    // node_aux_list is auxilliary buffer for synchronization of left and right boundaries of blocks (elems_to_left_count, elems_to_right_count)
    // processed by subgroups in the same node
    // no mul overflow check is required due to there is already buffer of size node_count * impl_const_t::node_prop_count_
    ONEDAL_ASSERT(aux_node_buffer_prop_count_ <= impl_const_t::node_prop_count_);

    auto [node_aux_list, last_event] =
        pr::ndarray<Index, 1>::zeros(queue_,
                                     { node_count * aux_node_buffer_prop_count_ },
                                     alloc::device);
    last_event.wait_and_throw();

    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index leaf_mark = impl_const_t::leaf_mark_;
    const Index aux_node_buffer_prop_count =
        aux_node_buffer_prop_count_; // num of auxilliary attributes for node
    const Index max_block_count = partition_max_block_count_;
    const Index min_block_size = partition_min_block_size_;

    const Bin* data_ptr = data.get_data();
    const Index* node_list_ptr = node_list.get_data();
    Index* node_aux_list_ptr = node_aux_list.get_mutable_data();
    const Index* tree_order_ptr = tree_order.get_data();
    Index* tree_order_buf_ptr = tree_order_buf.get_mutable_data();

    bool distr_mode = ctx.distr_mode_;

    auto local_size = preferable_partition_group_size_;
    const sycl::nd_range<1> nd_range =
        be::make_multiple_nd_range_1d(preferable_partition_groups_count_ * local_size, local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            const Index sub_group_size = sbg.get_local_range()[0];
            const Index work_group_size = item.get_local_range()[0];
            const Index sub_groups_in_work_group_num =
                work_group_size / sub_group_size; // num of subgroups for current node processing

            const Index sub_group_local_id = sbg.get_local_id();
            const Index work_group_local_id = item.get_local_id()[0];

            const Index sub_group_id = item.get_group().get_id(0) * sub_groups_in_work_group_num +
                                       work_group_local_id / sub_group_size;
            const Index sub_groups_num =
                item.get_group_range(0) *
                sub_groups_in_work_group_num; // num of subgroups for current node processing

            const Index total_block_count = node_count * max_block_count;

            for (Index block_ind_glob = sub_group_id; block_ind_glob < total_block_count;
                 block_ind_glob += sub_groups_num) {
                const Index node_id = block_ind_glob / max_block_count;
                const Index block_ind = block_ind_glob % max_block_count;

                const Index* node = node_list_ptr + node_id * node_prop_count;
                const Index offset = node[impl_const_t::ind_ofs];
                const Index row_count =
                    distr_mode ? node[impl_const_t::ind_lrc] : node[impl_const_t::ind_grc];
                const Index feat_id = node[impl_const_t::ind_fid];
                const Index feat_bin = node[impl_const_t::ind_bin];
                const Index left_ch_row_count =
                    distr_mode
                        ? node[impl_const_t::ind_lch_lrc]
                        : node[impl_const_t::ind_lch_grc]; // num of items in the Left part of node

                Index node_block_count =
                    row_count / min_block_size
                        ? sycl::min(row_count / min_block_size, max_block_count)
                        : 1;

                // if block_ind assigned for this sbg less than current node's block count -> sbg will just go to the next node
                if (feat_id != leaf_mark && block_ind < node_block_count) // split node
                {
                    Index* node_aux = node_aux_list_ptr + node_id * aux_node_buffer_prop_count;

                    const Index block_size =
                        node_block_count > 1
                            ? row_count / node_block_count + bool(row_count % node_block_count)
                            : row_count;

                    const Index ind_end = sycl::min((block_ind + 1) * block_size, row_count);
                    const Index ind_start = sycl::min(block_ind * block_size, ind_end);
                    const Index group_row_count = ind_end - ind_start;

                    Index group_left_boundary = 0;
                    Index group_right_boundary = 0;

                    if (node_block_count > 1 && group_row_count > 0) {
                        Index group_row_to_right_count = 0;
                        for (Index i = ind_start + sub_group_local_id; i < ind_end;
                             i += sub_group_size) {
                            const Index id = tree_order_ptr[offset + i];
                            const Index to_right =
                                Index(static_cast<Index>(
                                          data_ptr[id * data_column_count + feat_id]) > feat_bin);
                            group_row_to_right_count += reduce(sbg, to_right, plus<Index>());
                        }

                        if (0 == sub_group_local_id) {
                            group_left_boundary =
                                atomic_global_add(node_aux + 0,
                                                  group_row_count - group_row_to_right_count);
                            group_right_boundary =
                                atomic_global_add(node_aux + 1, group_row_to_right_count);
                        }
                        group_left_boundary = broadcast(sbg, group_left_boundary, 0);
                        group_right_boundary = broadcast(sbg, group_right_boundary, 0);
                    }

                    Index group_row_to_right_count = 0;
                    for (Index i = ind_start + sub_group_local_id; i < ind_end;
                         i += sub_group_size) {
                        const Index id = tree_order_ptr[offset + i];
                        const Index to_right =
                            Index(static_cast<Index>(data_ptr[id * data_column_count + feat_id]) >
                                  feat_bin);
                        const Index boundary =
                            group_row_to_right_count + exclusive_scan(sbg, to_right, plus<Index>());
                        const Index pos_new =
                            (to_right ? left_ch_row_count + group_right_boundary + boundary
                                      : group_left_boundary + i - ind_start - boundary);
                        tree_order_buf_ptr[offset + pos_new] = id;
                        group_row_to_right_count += reduce(sbg, to_right, plus<Index>());
                    }
                }
            }
        });
    });

    event.wait_and_throw();

    std::swap(tree_order, tree_order_buf);
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::update_mdi_var_importance(
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Float, 1>& node_imp_decrease_list,
    pr::ndarray<Float, 1>& res_var_imp,
    Index data_column_count,
    Index node_count,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(node_imp_decrease_list.get_count() == node_count);
    ONEDAL_ASSERT(res_var_imp.get_count() == data_column_count);

    Index local_size = preferable_group_size_;
    //calculating local size in way to have all subgroups for node in one group to use local buffer
    while (local_size > node_count && local_size > preferable_sbg_size_) {
        local_size >>= 1;
    }

    const Index* node_list_ptr = node_list.get_data();
    const Float* node_imp_decrease_list_ptr = node_imp_decrease_list.get_data();
    Float* res_var_imp_ptr = res_var_imp.get_mutable_data();

    const sycl::nd_range<2> nd_range =
        be::make_multiple_nd_range_2d({ local_size, data_column_count }, { local_size, 1 });

    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index leaf_mark = impl_const_t::leaf_mark_;
    const Index max_sub_groups_num =
        max_sbg_count_per_group_; //need to calculate it via device info

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        sycl::accessor<Float, 1, sycl::access::mode::read_write, sycl::access::target::local> buf(
            max_sub_groups_num,
            cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();

            const Index local_id = item.get_local_id()[0];
            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];
            const Index local_size = item.get_local_range()[0];
            const Index n_sub_groups =
                local_size / sub_group_size; // num of subgroups for current node processing
            const Index sub_group_id = local_id / sub_group_size;

            const Index buf_idx =
                item.get_global_id()[1] %
                (max_sub_groups_num / n_sub_groups); // local buffer is shared between 16 sub groups
            const Index ftr_id = item.get_global_id()[1];

            const Index sbg_elem_count =
                node_count / n_sub_groups + bool(node_count % n_sub_groups);

            const Index ind_start = sub_group_id * sbg_elem_count;
            const Index ind_end = sycl::min((sub_group_id + 1) * sbg_elem_count, node_count);

            Float ftr_imp = Float(0);

            for (Index node_idx = ind_start + sub_group_local_id; node_idx < ind_end;
                 node_idx += sub_group_size) {
                Index split_ftr_id =
                    node_list_ptr[node_idx * node_prop_count + impl_const_t::ind_fid];
                ftr_imp += reduce(sbg,
                                  ((split_ftr_id != leaf_mark && ftr_id == split_ftr_id)
                                       ? node_imp_decrease_list_ptr[node_idx]
                                       : Float(0)),
                                  plus<Float>());
            }

            if (0 == sub_group_local_id) {
                if (1 == n_sub_groups) {
                    res_var_imp_ptr[ftr_id] += ftr_imp;
                }
                else {
                    buf[buf_idx + sub_group_id] = ftr_imp;
                }
            }

            item.barrier(sycl::access::fence_space::local_space);
            if (1 < n_sub_groups && 0 == sub_group_id) {
                // first sub group for current node reduces over local buffer if required
                Float ftr_imp = (sub_group_local_id < n_sub_groups)
                                    ? buf[buf_idx + sub_group_local_id]
                                    : (Float)0;
                Float total_ftr_imp = reduce(sbg, ftr_imp, plus<Float>());

                if (0 == local_id) {
                    res_var_imp_ptr[ftr_id] += total_ftr_imp;
                }
            }
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::mark_present_rows(
    const pr::ndarray<Index, 1>& rowsList,
    pr::ndarray<Index, 1>& rowsBuffer,
    Index nRows,
    Index nTrees,
    Index tree_idx,
    Index localSize,
    Index nSubgroupSums,
    const be::event_vector& deps) {
    //DAAL_ITTNOTIFY_SCOPED_TASK(compute.mark_present_rows);

    ONEDAL_ASSERT(rowsList.get_count() == nRows * nTrees);
    ONEDAL_ASSERT(rowsBuffer.get_count() == nRows * nTrees);

    const Index* rows_list_ptr = rowsList.get_data();
    Index* rows_buffer_ptr = rowsBuffer.get_mutable_data();
    const Index itemPresentMark = 1;

    const sycl::nd_range<1> nd_range =
        be::make_multiple_nd_range_1d(localSize * nSubgroupSums, localSize);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();

            const Index n_groups = item.get_group_range(0);
            const Index n_sub_groups = sbg.get_group_range()[0];
            const Index n_total_sub_groups = n_sub_groups * n_groups;
            const Index elems_for_sbg =
                nRows / n_total_sub_groups + bool(nRows % n_total_sub_groups);

            const Index local_size = sbg.get_local_range()[0];

            const Index local_id = sbg.get_local_id();
            const Index sub_group_id = sbg.get_group_id();
            const Index group_id = item.get_group().get_id(0) * n_sub_groups + sub_group_id;

            const Index ind_start = group_id * elems_for_sbg;
            const Index ind_end = sycl::min((group_id + 1) * elems_for_sbg, nRows);

            for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                rows_buffer_ptr[nRows * tree_idx + rows_list_ptr[nRows * tree_idx + i]] =
                    itemPresentMark;
            }
        });
    });

    event.wait_and_throw();

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::count_absent_rows_for_blocks(
    const pr::ndarray<Index, 1>& rowsBuffer,
    pr::ndarray<Index, 1>& partial_sum,
    Index nRows,
    Index nTrees,
    Index tree_idx,
    Index localSize,
    Index nSubgroupSums,
    const be::event_vector& deps) {
    ONEDAL_ASSERT(rowsBuffer.get_count() == nRows * nTrees);
    ONEDAL_ASSERT(partial_sum.get_count() == nSubgroupSums);

    const Index* rows_buffer_ptr = rowsBuffer.get_data();
    Index* partial_sum_ptr = partial_sum.get_mutable_data();
    const Index itemAbsentMark = -1;

    const sycl::nd_range<1> nd_range =
        be::make_multiple_nd_range_1d(localSize * nSubgroupSums, localSize);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();

            const Index n_groups = item.get_group_range(0);
            const Index n_sub_groups = sbg.get_group_range()[0];
            const Index n_total_sub_groups = n_sub_groups * n_groups;
            const Index elems_for_sbg =
                nRows / n_total_sub_groups + bool(nRows % n_total_sub_groups);

            const Index local_size = sbg.get_local_range()[0];

            const Index local_id = sbg.get_local_id();
            const Index sub_group_id = sbg.get_group_id();
            const Index group_id = item.get_group().get_id(0) * n_sub_groups + sub_group_id;

            const Index ind_start = group_id * elems_for_sbg;
            const Index ind_end = sycl::min((group_id + 1) * elems_for_sbg, nRows);

            Index subSum = 0;

            for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                subSum += Index(itemAbsentMark == rows_buffer_ptr[nRows * tree_idx + i]);
            }

            Index sum = reduce(sbg, subSum, plus<Index>());

            if (local_id == 0) {
                partial_sum_ptr[group_id] = sum;
            }
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::count_absent_rows_total(
    const pr::ndarray<Index, 1>& partial_sum,
    pr::ndarray<Index, 1>& partial_prefix_sum,
    pr::ndarray<Index, 1>& oob_rows_num_list,
    Index nTrees,
    Index tree_idx,
    Index localSize,
    Index nSubgroupSums,
    const be::event_vector& deps) {
    //ONEDAL_ASSERT(rowsBuffer.get_count() == nRows * nTrees);
    ONEDAL_ASSERT(partial_sum.get_count() == nSubgroupSums);
    ONEDAL_ASSERT(partial_prefix_sum.get_count() == nSubgroupSums * nTrees);
    ONEDAL_ASSERT(oob_rows_num_list.get_count() == nTrees + 1);

    const Index* partial_sum_ptr = partial_sum.get_data();
    Index* partial_prefix_sum_ptr = partial_prefix_sum.get_mutable_data();
    Index* total_sum_ptr = oob_rows_num_list.get_mutable_data();

    const sycl::nd_range<1> nd_range =
        be::make_multiple_nd_range_1d(localSize * nSubgroupSums, localSize);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();

            if (sbg.get_group_id() > 0)
                return;
            const Index local_size = sbg.get_local_range()[0];
            const Index local_id = sbg.get_local_id();

            Index sum = 0;

            for (Index i = local_id; i < nSubgroupSums; i += local_size) {
                Index value = partial_sum_ptr[i];
                Index boundary = exclusive_scan(sbg, value, plus<Index>());
                partial_prefix_sum_ptr[nSubgroupSums * tree_idx + i] = sum + boundary;
                sum += reduce(sbg, value, plus<Index>());
            }

            if (local_id == 0) {
                total_sum_ptr[tree_idx + 1] = total_sum_ptr[tree_idx] + sum;
            }
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::fill_oob_rows_list_by_blocks(
    const pr::ndarray<Index, 1>& rowsBuffer,
    const pr::ndarray<Index, 1>& partial_prefix_sum,
    const pr::ndarray<Index, 1>& oob_row_num_list,
    pr::ndarray<Index, 1>& oob_row_list,
    Index nRows,
    Index nTrees,
    Index tree_idx,
    Index total_oob_row_num,
    Index localSize,
    Index nSubgroupSums,
    const be::event_vector& deps) {
    //DAAL_ITTNOTIFY_SCOPED_TASK(compute.fillOOBRowsListByBlocks);

    ONEDAL_ASSERT(rowsBuffer.get_count() == nRows * nTrees);
    ONEDAL_ASSERT(partial_prefix_sum.get_count() == nSubgroupSums * nTrees);
    ONEDAL_ASSERT(oob_row_num_list.get_count() == nTrees + 1);
    ONEDAL_ASSERT(oob_row_list.get_count() == total_oob_row_num);

    const Index* rows_buffer_ptr = rowsBuffer.get_data();
    const Index* partial_prefix_sum_ptr = partial_prefix_sum.get_data();
    const Index* oob_row_num_list_ptr = oob_row_num_list.get_data();
    Index* oob_row_list_ptr = oob_row_list.get_mutable_data();

    const Index itemAbsentMark = -1;

    const sycl::nd_range<1> nd_range =
        be::make_multiple_nd_range_1d(localSize * nSubgroupSums, localSize);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();

            const Index n_groups = item.get_group_range(0);
            const Index n_sub_groups = sbg.get_group_range()[0];
            const Index n_total_sub_groups = n_sub_groups * n_groups;
            const Index elems_for_sbg =
                nRows / n_total_sub_groups + bool(nRows % n_total_sub_groups);

            const Index local_size = sbg.get_local_range()[0];

            const Index local_id = sbg.get_local_id();
            const Index sub_group_id = sbg.get_group_id();
            const Index group_id = item.get_group().get_id(0) * n_sub_groups + sub_group_id;

            const Index ind_start = group_id * elems_for_sbg;
            const Index ind_end = sycl::min((group_id + 1) * elems_for_sbg, nRows);

            const int oobRowsListOffset = oob_row_num_list_ptr[tree_idx];

            int groupOffset = partial_prefix_sum_ptr[n_groups * tree_idx + group_id];
            int sum = 0;

            for (int i = ind_start + local_id; i < ind_end; i += local_size) {
                int oobRow = int(itemAbsentMark == rows_buffer_ptr[nRows * tree_idx + i]);
                int pos = groupOffset + sum + exclusive_scan(sbg, oobRow, plus<Index>());
                if (oobRow) {
                    oob_row_list_ptr[oobRowsListOffset + pos] = i;
                }
                sum += reduce(sbg, oobRow, plus<Index>());
            }
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::get_oob_row_list(
    const pr::ndarray<Index, 1>& rowsList,
    pr::ndarray<Index, 1>& oobRowsNumList,
    pr::ndarray<Index, 1>& oobRowsList,
    Index nRows,
    Index nTrees,
    const be::event_vector& deps) {
    const Index absentMark = -1;
    const Index localSize = preferable_sbg_size_;
    const Index nSubgroupSums = max_local_sums_ * localSize < nRows
                                    ? max_local_sums_
                                    : (nRows / localSize + !(nRows / localSize));

    ONEDAL_ASSERT(rowsList.get_count() == nRows * nTrees);
    ONEDAL_ASSERT(oobRowsNumList.get_count() == nTrees + 1);
    // oobRowsList will be created here

    sycl::event::wait_and_throw(deps);

    //sycl::event last_event;

    auto [rowsBuffer, last_event] = pr::ndarray<Index, 1>::full(
        queue_,
        { nRows * nTrees },
        absentMark,
        alloc::device); // it is filled with marks Present/Absent for each rows
    last_event.wait_and_throw();
    auto partialSums = pr::ndarray<Index, 1>::empty(queue_, { nSubgroupSums }, alloc::device);
    auto partialPrefixSums =
        pr::ndarray<Index, 1>::empty(queue_, { nSubgroupSums * nTrees }, alloc::device);
    Index totalOOBRowsNum = 0;

    last_event = oobRowsNumList.fill(queue_, 0);

    for (Index tree_idx = 0; tree_idx < nTrees; tree_idx++) {
        last_event = mark_present_rows(rowsList,
                                       rowsBuffer,
                                       nRows,
                                       nTrees,
                                       tree_idx,
                                       localSize,
                                       nSubgroupSums,
                                       { last_event });
        last_event = count_absent_rows_for_blocks(rowsBuffer,
                                                  partialSums,
                                                  nRows,
                                                  nTrees,
                                                  tree_idx,
                                                  localSize,
                                                  nSubgroupSums,
                                                  { last_event });
        last_event = count_absent_rows_total(partialSums,
                                             partialPrefixSums,
                                             oobRowsNumList,
                                             nTrees,
                                             tree_idx,
                                             localSize,
                                             nSubgroupSums,
                                             { last_event });
    }

    auto nOOBRowsHost = oobRowsNumList.to_host(queue_, { last_event });
    const Index* nOOBRowsHost_ptr = nOOBRowsHost.get_data();
    totalOOBRowsNum = nOOBRowsHost_ptr[nTrees];

    if (totalOOBRowsNum > 0) {
        // assign buffer of required size to the input oobRowsList buffer
        oobRowsList = pr::ndarray<Index, 1>::empty(queue_, { totalOOBRowsNum }, alloc::device);

        //be::event_vector vec_event(nTrees);

        for (Index tree_idx = 0; tree_idx < nTrees; tree_idx++) {
            Index nOOBRows = nOOBRowsHost_ptr[tree_idx + 1] - nOOBRowsHost_ptr[tree_idx];

            if (nOOBRows > 0) {
                last_event = fill_oob_rows_list_by_blocks(rowsBuffer,
                                                          partialPrefixSums,
                                                          oobRowsNumList,
                                                          oobRowsList,
                                                          nRows,
                                                          nTrees,
                                                          tree_idx,
                                                          totalOOBRowsNum,
                                                          localSize,
                                                          nSubgroupSums,
                                                          { last_event });
                last_event.wait_and_throw();
            }
        }
    }

    return last_event;
}

#define INSTANTIATE(F, B, I, T) template class train_service_kernels<F, B, I, T>;

} // namespace oneapi::dal::decision_forest::backend
