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
#include "oneapi/dal/backend/primitives/sort/sort.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float, typename Bin, typename Index = std::uint32_t>
class indexed_features {
public:
    struct feature_entry {
        Index bin_count_ = 0;
        Index offset_ = 0;
        dal::backend::primitives::ndarray<Float, 1> bin_borders_nd_; //right bin borders
    };

    indexed_features(sycl::queue& q, std::int64_t min_bin_size, std::int64_t max_bins);
    ~indexed_features() = default;

    static std::int64_t get_required_mem_size(std::int64_t row_count,
                                              std::int64_t column_count,
                                              std::int64_t max_bins);

    sycl::event operator()(const table& tbl, const dal::backend::event_vector& deps = {});

    Index get_bin_count(std::int64_t column_idx) const {
        return entries_[column_idx].bin_count_;
    }
    Index get_total_bin_count() const {
        return total_bins_;
    }

    dal::backend::primitives::ndarray<Float, 1> get_bin_borders(Index column_idx) {
        return entries_[column_idx].bin_borders_nd_;
    }

    dal::backend::primitives::ndview<Index, 1> get_bin_offsets() {
        return bin_offsets_nd_;
    }

    dal::backend::primitives::ndview<Bin, 2> get_full_data() {
        return full_data_nd_;
    }

    Index get_row_count() const {
        return row_count_;
    }
    Index get_column_count() const {
        return column_count_;
    }

private:
    sycl::event extract_column(const dal::backend::primitives::ndarray<Float, 2>& data_nd,
                               dal::backend::primitives::ndarray<Float, 1>& values_nd,
                               dal::backend::primitives::ndarray<Index, 1>& indices_nd,
                               Index feature_id,
                               const dal::backend::event_vector& deps = {});
    sycl::event collect_bin_borders(
        const dal::backend::primitives::ndarray<Float, 1>& values_nd,
        const dal::backend::primitives::ndarray<Index, 1>& bin_offsets_nd,
        dal::backend::primitives::ndarray<Float, 1>& bin_borders_nd,
        const dal::backend::event_vector& deps = {});
    sycl::event compute_bins(const dal::backend::primitives::ndarray<Float, 1>& values_nd,
                             const dal::backend::primitives::ndarray<Index, 1>& indices_nd,
                             const dal::backend::primitives::ndarray<Float, 1>& bin_borders_nd,
                             const dal::backend::primitives::ndarray<Bin, 1>& bins_nd,
                             Index bin_count,
                             size_t local_size,
                             size_t local_blocks_count,
                             const dal::backend::event_vector& deps = {});
    sycl::event compute_bins(const dal::backend::primitives::ndarray<Float, 1>& values_nd,
                             const dal::backend::primitives::ndarray<Index, 1>& indices_nd,
                             dal::backend::primitives::ndarray<Bin, 1>& bins_nd,
                             feature_entry& entry,
                             const dal::backend::event_vector& deps);

    sycl::event store_column(const dal::backend::primitives::ndarray<Bin, 1>& column_data_nd,
                             dal::backend::primitives::ndarray<Bin, 2>& full_data_nd,
                             Index column_idx,
                             Index column_count,
                             const dal::backend::event_vector& deps);

    sycl::queue queue_;

    dal::backend::primitives::ndarray<Bin, 2> full_data_nd_;
    dal::backend::primitives::ndarray<Index, 1> bin_offsets_nd_;

    std::vector<feature_entry> entries_;

    Index row_count_ = 0;
    Index column_count_ = 0;
    Index total_bins_ = 0;
    Index min_bin_size_ = 0;
    Index max_bins_ = 0;

    static constexpr inline std::uint32_t preferable_sbg_size_ = 16;
    static constexpr inline std::uint32_t max_local_block_count_ = 1024;
};

#endif

} // namespace oneapi::dal::decision_forest::backend
