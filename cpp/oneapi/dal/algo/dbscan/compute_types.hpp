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

#include "oneapi/dal/algo/dbscan/common.hpp"

namespace oneapi::dal::dbscan {

namespace detail {
namespace v1 {
template <typename Task>
class compute_input_impl;

template <typename Task>
class compute_result_impl;
} // namespace v1

using v1::compute_input_impl;
using v1::compute_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::clustering`.
template <typename Task = task::by_default>
class compute_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data` and
    /// :literal:`weights`
    compute_input(const table& data = {}, const table& weights = {});

    /// An $n \\times p$ table with the data to be clustered, where each row
    /// stores one feature vector.
    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

    /// A single column table with the weights, where each row
    /// stores one weight per observation.
    const table& get_weights() const;

    auto& set_weights(const table& data) {
        set_weights_impl(data);
        return *this;
    }

protected:
    void set_data_impl(const table& data);
    void set_weights_impl(const table& weights);

private:
    dal::detail::pimpl<detail::compute_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::clustering`.
template <typename Task = task::by_default>
class compute_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    compute_result();

    /// The number of clusters found by the algorithm.
    /// @invariant :expr:`cluster_count >= 0`
    std::int64_t get_cluster_count() const;

    auto& set_cluster_count(std::int64_t value) {
        set_cluster_count_impl(value);
        return *this;
    }

    /// An $n \\times 1$ table with the responses $y_i$ assigned to the
    /// samples $x_i$ in the input data
    /// @remark default = table{}
    const table& get_responses() const;

    auto& set_responses(const table& value) {
        set_responses_impl(value);
        return *this;
    }

    /// An $n \\times 1$ table with the core flags $y_i$ assigned to the
    /// samples $x_i$ in the input data
    const table& get_core_flags() const;

    auto& set_core_flags(const table& value) {
        set_core_flags_impl(value);
        return *this;
    }

    /// An $m \\times 1$ table with the indices of core observations in
    /// the input data. $m is a number of core observations
    const table& get_core_observation_indices() const;

    auto& set_core_observation_indices(const table& value) {
        set_core_observation_indices_impl(value);
        return *this;
    }

    /// An $m \\times p$ table with the core observations in
    /// the input data. $m is a number of core observations
    const table& get_core_observations() const;

    auto& set_core_observations(const table& value) {
        set_core_observations_impl(value);
        return *this;
    }

protected:
    void set_cluster_count_impl(std::int64_t);
    void set_responses_impl(const table&);
    void set_core_flags_impl(const table&);
    void set_core_observation_indices_impl(const table&);
    void set_core_observations_impl(const table&);

private:
    dal::detail::pimpl<detail::compute_result_impl<Task>> impl_;
};

} // namespace v1

using v1::compute_input;
using v1::compute_result;

} // namespace oneapi::dal::dbscan
