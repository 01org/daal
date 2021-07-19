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
class train_input_impl;

template <typename Task>
class train_result_impl;
} // namespace v1

using v1::train_input_impl;
using v1::train_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::classification` or :expr:`task::search`.
template <typename Task = task::by_default>
class train_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data`
    /// and :literal:`responses` property values
    train_input(const table& data, const table& responses);

    train_input(const table& data);

    /// The training set X
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

    /// Vector of labels y for the training set X
    /// @remark default = table{}
    [[deprecated]] const table& get_labels() const {
        return get_responses();
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    [[deprecated]] auto& set_labels(const table& value) {
        return set_responses(value);
    }

    /// Vector of responses y for the training set X
    /// @remark default = table{}
    const table& get_responses() const;

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_responses(const table& responses) {
        set_data_impl(responses);
        return *this;
    }

protected:
    void set_data_impl(const table& data);
    void set_responses_impl(const table& responses);

private:
    dal::detail::pimpl<detail::train_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::classification` or :expr:`task::search`.
template <typename Task = task::by_default>
class train_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    train_result();

    /// The trained k-NN model
    /// @remark default = model<Task>{}
    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

protected:
    void set_model_impl(const model<Task>&);

private:
    dal::detail::pimpl<detail::train_result_impl<Task>> impl_;
};

} // namespace v1

using v1::train_input;
using v1::train_result;

} // namespace oneapi::dal::knn
