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

#include "oneapi/dal/algo/svm/backend/gpu/misc.hpp"
#include "oneapi/dal/backend/primitives/selection/select_flagged.hpp"

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;

inline std::int64_t propose_working_set_size(const sycl::queue& queue,
                                             const std::int64_t n_vectors) {
    const std::int64_t max_wg_size = dal::backend::device_max_wg_size(queue);
    return std::min(dal::backend::down_pow2<std::uint32_t>(n_vectors),
                    dal::backend::down_pow2<std::uint32_t>(max_wg_size));
}

template <typename Float>
class working_set_selector {
public:
    working_set_selector(const sycl::queue& queue,
                         const pr::ndarray<Float, 1>& y,
                         const Float C,
                         const std::int64_t n_vectors,
                         const std::int64_t n_ws = 0)
            : queue_(queue),
              n_selected_(0),
              n_vectors_(n_vectors),
              n_ws_(n_ws),
              C_(C),
              labels_(y) {
        ONEDAL_ASSERT(n_vectors > 0);
        ONEDAL_ASSERT(n_vectors <= dal::detail::limits<std::uint32_t>::max());
        sorted_f_inices_ =
            pr::ndarray<std::uint32_t, 1>::empty(queue_, { n_vectors_ }, sycl::usm::alloc::device);
        auto [indicator, indicator_event] =
            pr::ndarray<std::uint32_t, 1>::zeros(queue_, { n_vectors_ }, sycl::usm::alloc::device);

        if (!n_ws_) {
            n_ws_ = propose_working_set_size(queue_, n_vectors);
        }
        values_sort_ =
            pr::ndarray<Float, 1>::empty(queue_, { n_vectors_ }, sycl::usm::alloc::device);
        buff_indices_ =
            pr::ndarray<std::uint32_t, 1>::empty(queue_, { n_vectors_ }, sycl::usm::alloc::device);

        indicator_event.wait_and_throw();
        indicator_ = indicator;
    }

    sycl::event copy_last_to_first(pr::ndview<std::uint32_t, 1>& ws_indices,
                                   const dal::backend::event_vector& deps = {}) {
        ONEDAL_ASSERT(ws_indices.get_dimension(0) == n_ws_);
        ONEDAL_ASSERT(ws_indices.has_mutable_data());
        const std::int64_t q = n_ws_ / 2;
        std::uint32_t* ws_indices_ptr = ws_indices.get_mutable_data();
        auto copy_event =
            dal::backend::copy(queue_, ws_indices_ptr, ws_indices_ptr + q, n_ws_ - q, deps);
        n_selected_ = q;
        return copy_event;
    }

    sycl::event select(const pr::ndview<Float, 1>& alpha,
                       const pr::ndview<Float, 1>& f,
                       pr::ndview<std::uint32_t, 1>& ws_indices,
                       const dal::backend::event_vector& deps = {}) {
        ONEDAL_ASSERT(labels_.get_dimension(0) == alpha.get_dimension(0));
        ONEDAL_ASSERT(labels_.get_dimension(0) == f.get_dimension(0));
        ONEDAL_ASSERT(alpha.get_dimension(0) == f.get_dimension(0));
        ONEDAL_ASSERT(ws_indices.get_dimension(0) == n_ws_);
        ONEDAL_ASSERT(ws_indices.has_mutable_data());

        auto arg_sort_event = arg_sort(queue_, f, values_sort_, sorted_f_inices_, n_vectors_, deps);

        sycl::event copy_event;

        {
            const std::int64_t n_need_select = (n_ws_ - n_selected_) / 2;

            auto check_upper_event =
                check_upper(queue_, labels_, alpha, indicator_, C_, n_vectors_, { arg_sort_event });

            /* Reset indicator for busy Indices */
            if (n_selected_ > 0) {
                reset_indicator_with_zeros(ws_indices,
                                           indicator_,
                                           n_selected_,
                                           { check_upper_event })
                    .wait_and_throw();
            }
            std::int64_t n_upper_select = 0;
            auto select_event = pr::select_flagged_index<std::uint32_t, std::uint32_t>{
                queue_
            }(indicator_, sorted_f_inices_, buff_indices_, n_upper_select, { check_upper_event });

            const std::int64_t n_copy = std::min(n_upper_select, n_need_select);

            std::uint32_t* ws_indices_ptr = ws_indices.get_mutable_data();
            const std::uint32_t* buff_indices_ptr = buff_indices_.get_data();

            if (n_copy > 0)
                copy_event = dal::backend::copy(queue_,
                                                ws_indices_ptr + n_selected_,
                                                buff_indices_ptr,
                                                n_copy,
                                                { select_event });

            n_selected_ += n_copy;
        }

        {
            const std::int64_t n_need_select = n_ws_ - n_selected_;

            auto check_lower_event =
                check_lower(queue_, labels_, alpha, indicator_, C_, n_vectors_, { copy_event });

            /* Reset indicator for busy Indices */
            if (n_selected_ > 0) {
                reset_indicator_with_zeros(ws_indices,
                                           indicator_,
                                           n_selected_,
                                           { check_lower_event })
                    .wait_and_throw();
            }

            std::int64_t n_lower_select = 0;
            auto select_event = pr::select_flagged_index<std::uint32_t, std::uint32_t>{
                queue_
            }(indicator_, sorted_f_inices_, buff_indices_, n_lower_select, { check_lower_event });

            const std::int64_t n_copy = std::min(n_lower_select, n_need_select);

            std::uint32_t* ws_indices_ptr = ws_indices.get_mutable_data();
            const std::uint32_t* buff_indices_ptr = buff_indices_.get_data();

            if (n_copy > 0)
                copy_event = dal::backend::copy(queue_,
                                                ws_indices_ptr + n_selected_,
                                                buff_indices_ptr + n_lower_select - n_copy,
                                                n_copy,
                                                { select_event });

            n_selected_ += n_copy;
        }

        if (n_selected_ < n_ws_) {
            const std::int64_t n_need_select = n_ws_ - n_selected_;

            auto check_upper_event =
                check_upper(queue_, labels_, alpha, indicator_, C_, n_vectors_, { copy_event });

            /* Reset indicator for busy Indices */
            if (n_selected_ > 0) {
                reset_indicator_with_zeros(ws_indices,
                                           indicator_,
                                           n_selected_,
                                           { check_upper_event })
                    .wait_and_throw();
            }

            std::int64_t n_upper_select = 0;
            auto select_event = pr::select_flagged_index<std::uint32_t, std::uint32_t>{
                queue_
            }(indicator_, sorted_f_inices_, buff_indices_, n_upper_select, { check_upper_event });

            const std::int64_t n_copy = std::min(n_upper_select, n_need_select);

            std::uint32_t* ws_indices_ptr = ws_indices.get_mutable_data();
            const std::uint32_t* buff_indices_ptr = buff_indices_.get_data();

            if (n_copy > 0)
                copy_event = dal::backend::copy(queue_,
                                                ws_indices_ptr + n_selected_,
                                                buff_indices_ptr,
                                                n_copy,
                                                { select_event });

            n_selected_ += n_copy;
        }
        ONEDAL_ASSERT(n_selected_ == n_ws_);

        n_selected_ = 0;
        return copy_event;
    }

private:
    sycl::event reset_indicator_with_zeros(const pr::ndview<std::uint32_t, 1>& idx,
                                           pr::ndview<std::uint32_t, 1>& indicator,
                                           const std::int64_t n,
                                           const dal::backend::event_vector& deps = {}) {
        ONEDAL_ASSERT(idx.get_dimension(0) == n_ws_);
        ONEDAL_ASSERT(indicator.get_dimension(0) == n_vectors_);
        ONEDAL_ASSERT(indicator.has_mutable_data());

        const std::uint32_t* idx_ptr = idx.get_data();
        std::uint32_t* indicator_ptr = indicator.get_mutable_data();

        const auto wg_size = std::min(dal::backend::propose_wg_size(queue_), n);
        const auto range = dal::backend::make_multiple_nd_range_1d(n, wg_size);

        auto reset_indicator_with_zeros_event = queue_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                const std::uint32_t i = item.get_global_id(0);
                indicator_ptr[idx_ptr[i]] = 0;
            });
        });

        return reset_indicator_with_zeros_event;
    }

    sycl::queue queue_;

    std::int64_t n_selected_;
    std::int64_t n_vectors_;
    std::int64_t n_ws_;
    Float C_;

    pr::ndarray<std::uint32_t, 1> sorted_f_inices_;
    pr::ndarray<std::uint32_t, 1> indicator_;
    pr::ndarray<std::uint32_t, 1> buff_indices_;
    pr::ndarray<Float, 1> values_sort_;
    pr::ndarray<Float, 1> labels_;
};

#define INSTANTIATE_WORKING_SET(F) template class ONEDAL_EXPORT working_set_selector<F>;

INSTANTIATE_WORKING_SET(float);
INSTANTIATE_WORKING_SET(double);

} // namespace oneapi::dal::svm::backend
