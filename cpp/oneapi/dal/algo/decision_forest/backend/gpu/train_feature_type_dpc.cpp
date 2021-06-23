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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_feature_type.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#ifdef ONEDAL_DATA_PARALLEL

#include <CL/sycl/ONEAPI/experimental/builtins.hpp>

#endif

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace be = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::extract_column(
    const pr::ndarray<Float, 2>& data_nd,
    pr::ndarray<Float, 1>& values_nd,
    pr::ndarray<Index, 1>& indices_nd,
    Index feature_id,
    const dal::backend::event_vector& deps) {
    ONEDAL_ASSERT(data_nd.get_count() == row_count_ * column_count_);
    ONEDAL_ASSERT(values_nd.get_count() == row_count_);
    ONEDAL_ASSERT(indices_nd.get_count() == row_count_);

    const Float* data = data_nd.get_data();

    Float* values = values_nd.get_mutable_data();
    Index* indices = indices_nd.get_mutable_data();
    auto column_count = column_count_;

    const sycl::range<1> range = de::integral_cast<size_t>(row_count_);

    auto event = queue_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> idx) {
            values[idx] = data[idx * column_count + feature_id];
            indices[idx] = idx;
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::collect_bin_borders(
    const pr::ndarray<Float, 1>& values_nd,
    const pr::ndarray<Index, 1>& bin_offsets_nd,
    pr::ndarray<Float, 1>& bin_borders_nd,
    const dal::backend::event_vector& deps) {
    ONEDAL_ASSERT(values_nd.get_count() == row_count_);
    ONEDAL_ASSERT(bin_offsets_nd.get_count() <= max_bins_);
    ONEDAL_ASSERT(bin_borders_nd.get_count() <= max_bins_);

    const sycl::range<1> range = de::integral_cast<size_t>(max_bins_);

    const Float* values = values_nd.get_data();
    const Index* bin_offsets = bin_offsets_nd.get_data();
    Float* bin_borders = bin_borders_nd.get_mutable_data();

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            bin_borders[idx] = values[bin_offsets[idx]];
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::compute_bins(
    const pr::ndarray<Float, 1>& values_nd,
    const pr::ndarray<Index, 1>& indices_nd,
    const pr::ndarray<Float, 1>& bin_borders_nd,
    const pr::ndarray<Bin, 1>& bins_nd,
    Index bin_count,
    size_t local_size,
    size_t local_blocks_count,
    const dal::backend::event_vector& deps) {
    ONEDAL_ASSERT(values_nd.get_count() == row_count_);
    ONEDAL_ASSERT(indices_nd.get_count() == row_count_);
    ONEDAL_ASSERT(bin_borders_nd.get_count() <= max_bins_);
    ONEDAL_ASSERT(bins_nd.get_count() == row_count_);

    const sycl::nd_range<1> nd_range =
        be::make_multiple_nd_range_1d(de::check_mul_overflow(local_size, local_blocks_count),
                                      local_size);

    const Index row_count = row_count_;
    const Float* values = values_nd.get_data();
    const Index* indices = indices_nd.get_data();
    const Float* bin_borders = bin_borders_nd.get_data();
    Bin* bins = bins_nd.get_mutable_data();

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            const std::uint32_t n_groups = item.get_group_range(0);
            const std::uint32_t n_sub_groups = sbg.get_group_range()[0];
            const std::uint32_t n_total_sub_groups = n_sub_groups * n_groups;
            const Index elems_for_sbg =
                row_count / n_total_sub_groups + bool(row_count % n_total_sub_groups);
            const std::uint32_t local_size = sbg.get_local_range()[0];

            const std::uint32_t local_id = sbg.get_local_id();
            const std::uint32_t sub_group_id = sbg.get_group_id();
            const std::uint32_t group_id = item.get_group().get_id(0) * n_sub_groups + sub_group_id;

            Index ind_start = group_id * elems_for_sbg;
            Index ind_end =
                sycl::min(static_cast<Index>((group_id + 1) * elems_for_sbg), row_count);

            Index cur_bin = 0;

            for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                Float value = values[i];
                while (bin_borders[cur_bin] < value)
                    cur_bin++;
                bins[indices[i]] = cur_bin;
            }
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index>
//std::tuple<pr::ndarray<Index, 1>, sycl::event> indexed_features<Float, Bin, Index>::compute_bins(
sycl::event indexed_features<Float, Bin, Index>::compute_bins(
    const pr::ndarray<Float, 1>& values_nd,
    const pr::ndarray<Index, 1>& indices_nd,
    pr::ndarray<Bin, 1>& bins_nd,
    feature_entry& entry,
    const dal::backend::event_vector& deps) {
    ONEDAL_ASSERT(values_nd.get_count() == row_count_);
    ONEDAL_ASSERT(indices_nd.get_count() == row_count_);
    ONEDAL_ASSERT(bins_nd.get_count() == row_count_);

    sycl::event::wait_and_throw(deps);

    const Index max_bins = std::min(max_bins_, row_count_);
    const Index local_size = preferable_sbg_size_;
    const Index local_block_count = max_local_block_count_ * local_size < row_count_
                                        ? max_local_block_count_
                                        : (row_count_ / local_size) + bool(row_count_ % local_size);

    auto bin_offsets_nd_host = pr::ndarray<Index, 1>::empty({ max_bins });
    auto bin_borders_nd_device =
        pr::ndarray<Float, 1>::empty(queue_, { max_bins }, sycl::usm::alloc::device);

    auto bin_offsets = bin_offsets_nd_host.get_mutable_data();
    Index offset = 0;
    for (Index i = 0; i < max_bins; i++) {
        offset += (row_count_ + i) / max_bins;
        ONEDAL_ASSERT(offset > 0); // max_bins = min(max_bins_, row_count_) => offset > 0
        bin_offsets[i] = offset - 1;
    }

    bin_offsets_nd_ = bin_offsets_nd_host.to_device(queue_);
    auto last_event =
        collect_bin_borders(values_nd, bin_offsets_nd_, bin_borders_nd_device, { deps });

    Index bin_count = 0;
    auto bin_borders_nd_host = bin_borders_nd_device.to_host(queue_, { last_event });
    auto bin_borders = bin_borders_nd_host.get_mutable_data();
    for (Index i = 0; i < max_bins; i++) {
        if (0 == bin_count || (bin_count > 0 && bin_borders[i] != bin_borders[bin_count - 1])) {
            bin_borders[bin_count] = bin_borders[i];
            bin_count++;
        }
    }
    last_event = bin_borders_nd_device.assign(queue_, bin_borders_nd_host);

    last_event = compute_bins(values_nd,
                              indices_nd,
                              bin_borders_nd_device,
                              bins_nd,
                              bin_count,
                              local_size,
                              local_block_count,
                              { last_event });

    entry.bin_count_ = bin_count;
    entry.bin_borders_nd_ = bin_borders_nd_device;

    return last_event;
}

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::store_column(
    const pr::ndarray<Bin, 1>& column_data_nd,
    pr::ndarray<Bin, 2>& full_data_nd,
    Index column_idx,
    Index column_count,
    const dal::backend::event_vector& deps) {
    ONEDAL_ASSERT(column_data_nd.get_count() == row_count_);
    ONEDAL_ASSERT(full_data_nd.get_count() == row_count_ * column_count_);

    const Bin* column_data = column_data_nd.get_data();
    Bin* full_data = full_data_nd.get_mutable_data();

    const sycl::range<1> range = de::integral_cast<size_t>(column_data_nd.get_dimension(0));

    auto event = queue_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> idx) {
            full_data[idx * column_count + column_idx] = column_data[idx];
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index>
indexed_features<Float, Bin, Index>::indexed_features(sycl::queue& q,
                                                      std::int64_t min_bin_size,
                                                      std::int64_t max_bins)
        : queue_(q) {
    min_bin_size_ = de::integral_cast<Index>(min_bin_size);
    max_bins_ = de::integral_cast<Index>(max_bins);
}

template <typename Float, typename Bin, typename Index>
std::int64_t indexed_features<Float, Bin, Index>::get_required_mem_size(std::int64_t row_count,
                                                                        std::int64_t column_count,
                                                                        std::int64_t max_bins) {
    std::int64_t required_mem = 0;
    required_mem += sizeof(Bin) * (column_count + 1); // bin_offsets
    required_mem +=
        sizeof(Bin) * row_count * column_count; // data vs ftrs bin map table (full_data_nd)
    required_mem +=
        sizeof(Float) * column_count * std::min(max_bins, row_count); // bin_borders for each column

    return required_mem;
}

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::operator()(
    const table& tbl,
    const dal::backend::event_vector& deps) {
    sycl::event::wait_and_throw(deps);

    const auto data_nd_ = pr::table2ndarray<Float>(queue_, tbl, sycl::usm::alloc::device);

    if (tbl.get_row_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_rows());
    }
    if (tbl.get_column_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_columns());
    }

    row_count_ = de::integral_cast<Index>(tbl.get_row_count());
    column_count_ = de::integral_cast<Index>(tbl.get_column_count());
    total_bins_ = 0;

    //allocating buffers
    full_data_nd_ =
        pr::ndarray<Bin, 2>::empty(queue_, { row_count_, column_count_ }, sycl::usm::alloc::device);

    entries_.resize(column_count_);

    auto values_nd = pr::ndarray<Float, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);
    auto indices_nd =
        pr::ndarray<Index, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);
    std::vector<pr::ndarray<Bin, 1>> column_bin_vec_;
    column_bin_vec_.resize(column_count_);

    for (Index i = 0; i < column_count_; i++) {
        column_bin_vec_[i] =
            pr::ndarray<Bin, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);
    }

    pr::radix_sort_indices_inplace<Float, Index> sort{ queue_ };

    sycl::event last_event;

    for (Index i = 0; i < column_count_; i++) {
        last_event = extract_column(data_nd_, values_nd, indices_nd, i, { last_event });
        last_event = sort(values_nd, indices_nd, { last_event });
        last_event =
            compute_bins(values_nd, indices_nd, column_bin_vec_[i], entries_[i], { last_event });
    }

    last_event.wait_and_throw();

    auto bin_offsets_nd_host = pr::ndarray<Index, 1>::empty({ column_count_ + 1 });
    auto bin_offsets = bin_offsets_nd_host.get_mutable_data();

    Index total = 0;
    for (Index i = 0; i < column_count_; i++) {
        last_event =
            store_column(column_bin_vec_[i], full_data_nd_, i, column_count_, { last_event });
        bin_offsets[i] = total;
        entries_[i].offset_ = total;
        total += entries_[i].bin_count_;
    }

    bin_offsets[column_count_] = total;
    total_bins_ = total;

    bin_offsets_nd_ = bin_offsets_nd_host.to_device(queue_);

    return last_event;
}

#define INSTANTIATE(F, B) template class indexed_features<F, B>;

INSTANTIATE(float, std::uint32_t);
INSTANTIATE(float, std::uint8_t);
INSTANTIATE(double, std::uint32_t);
INSTANTIATE(double, std::uint8_t);

} // namespace oneapi::dal::decision_forest::backend
