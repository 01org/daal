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

#include "oneapi/dal/algo/dbscan/compute_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::dbscan {

template <typename Task>
class detail::v1::compute_input_impl : public base {
public:
    compute_input_impl(const table& data) : data(data) {}
    compute_input_impl(const table& data, const table& weights) : data(data), weights(weights) {}

    table data;
    table weights;
};

template <typename Task>
class detail::v1::compute_result_impl : public base {
public:
    table responses;
    table core_flags;
    table core_observation_indices;
    table core_observations;
    std::int64_t cluster_count = 0;
};

using detail::v1::compute_input_impl;
using detail::v1::compute_result_impl;

namespace v1 {

template <typename Task>
compute_input<Task>::compute_input(const table& data, const table& weights)
        : impl_(new compute_input_impl<Task>(data, weights)) {}

template <typename Task>
const table& compute_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
const table& compute_input<Task>::get_weights() const {
    return impl_->weights;
}

template <typename Task>
void compute_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
void compute_input<Task>::set_weights_impl(const table& value) {
    impl_->weights = value;
}

template <typename Task>
compute_result<Task>::compute_result() : impl_(new compute_result_impl<Task>{}) {}

template <typename Task>
const table& compute_result<Task>::get_responses() const {
    return impl_->responses;
}

template <typename Task>
const table& compute_result<Task>::get_core_flags() const {
    return impl_->core_flags;
}

template <typename Task>
const table& compute_result<Task>::get_core_observation_indices() const {
    return impl_->core_observation_indices;
}

template <typename Task>
const table& compute_result<Task>::get_core_observations() const {
    return impl_->core_observations;
}

template <typename Task>
std::int64_t compute_result<Task>::get_cluster_count() const {
    return impl_->cluster_count;
}

template <typename Task>
void compute_result<Task>::set_responses_impl(const table& value) {
    impl_->responses = value;
}

template <typename Task>
void compute_result<Task>::set_core_flags_impl(const table& value) {
    impl_->core_flags = value;
}

template <typename Task>
void compute_result<Task>::set_core_observation_indices_impl(const table& value) {
    impl_->core_observation_indices = value;
}

template <typename Task>
void compute_result<Task>::set_core_observations_impl(const table& value) {
    impl_->core_observations = value;
}

template <typename Task>
void compute_result<Task>::set_cluster_count_impl(std::int64_t value) {
    if (value < 0) {
        //        throw domain_error(dal::detail::error_messages::cluster_count_lt_zero());
    }
    impl_->cluster_count = value;
}

template class ONEDAL_EXPORT compute_input<task::clustering>;
template class ONEDAL_EXPORT compute_result<task::clustering>;

} // namespace v1
} // namespace oneapi::dal::dbscan
