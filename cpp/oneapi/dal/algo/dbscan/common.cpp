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

#include "oneapi/dal/algo/dbscan/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::dbscan {
namespace detail {
namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    std::int64_t min_observations;
    double epsilon;
    bool mem_save_mode;
};

template <typename Task>
class model_impl : public base {
public:
    table centroids;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
std::int64_t descriptor_base<Task>::get_min_observations() const {
    return impl_->min_observations;
}

template <typename Task>
double descriptor_base<Task>::get_epsilon() const {
    return impl_->epsilon;
}

template <typename Task>
bool descriptor_base<Task>::get_mem_save_mode() const {
    return impl_->mem_save_mode;
}

template <typename Task>
void descriptor_base<Task>::set_epsilon_impl(double value) {
    if (value <= 0) {
        throw domain_error(dal::detail::error_messages::cluster_count_leq_zero());
    }
    impl_->epsilon = value;
}

template <typename Task>
void descriptor_base<Task>::set_min_observations_impl(std::int64_t value) {
    impl_->min_observations = value;
}

template <typename Task>
void descriptor_base<Task>::set_mem_save_mode_impl(bool value) {
    impl_->mem_save_mode = value;
}

template class ONEDAL_EXPORT descriptor_base<task::clustering>;

} // namespace v1
} // namespace detail

} // namespace oneapi::dal::dbscan
