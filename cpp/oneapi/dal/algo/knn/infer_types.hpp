/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/knn/common.hpp"

namespace oneapi::dal::knn {

namespace detail {
namespace v1 {
template <typename Task>
class infer_input_impl;

template <typename Task>
class infer_result_impl;
} // namespace v1

using v1::infer_input_impl;
using v1::infer_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::classification` or :expr:`task::search`.
template <typename Task = task::by_default>
class infer_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`model`
    /// and :literal:`data` property values
    infer_input(const table& data, const model<Task>& model);

    /// The dataset for inference $X'$
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

    /// The trained k-NN model
    /// @remark default = model<Task>{}
    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& m) {
        set_model_impl(m);
        return *this;
    }

protected:
    void set_data_impl(const table& data);
    void set_model_impl(const model<Task>& model);

private:
    dal::detail::pimpl<detail::infer_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::classification` or :expr:`task::search`.
template <typename Task = task::by_default>
class infer_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    infer_result();

    /// The predicted labels
    /// @remark default = table{}
    [[deprecated]] const table& get_labels() const {
        return get_responses();
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    [[deprecated]] auto& set_labels(const table& value) {
        return set_responses(value);
    }

    /// The predicted responses
    /// @remark default = table{}
    const table& get_responses() const;

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_responses(const table& value) {
        set_responses_impl(value);
        return *this;
    }

    /// Indices of nearest neighbors
    /// @remark default = table{}
    const table& get_indices() const;

    auto& set_indices(const table& value) {
        set_indices_impl(value);
        return *this;
    }

    /// Distances to nearest neighbors
    /// @remark default = table{}
    const table& get_distances() const;

    auto& set_distances(const table& value) {
        set_distances_impl(value);
        return *this;
    }

    /// Result options that indicates availability of the properties
    /// @remark default = default_result_options<Task>
    const result_option_id& get_result_options() const;

    auto& set_result_options(const result_option_id& value) {
        set_result_options_impl(value);
        return *this;
    }

protected:
    void set_responses_impl(const table&);
    void set_indices_impl(const table&);
    void set_distances_impl(const table&);
    void set_result_options_impl(const result_option_id&);
    const table& get_responses_impl() const;

private:
    dal::detail::pimpl<detail::infer_result_impl<Task>> impl_;
};

} // namespace v1

using v1::infer_input;
using v1::infer_result;

} // namespace oneapi::dal::knn
