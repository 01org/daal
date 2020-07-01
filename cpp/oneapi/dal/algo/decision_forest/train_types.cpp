/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/train_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::decision_forest {

class detail::train_input_impl : public base {
public:
    train_input_impl(const table& data, const table& labels) : data(data), labels(labels) {}

    table data;
    table labels;
};

class detail::train_result_impl : public base {
public:
    model trained_model;

    table oob_err;
    table oob_per_observation_err;
    table variable_importance;
};

using detail::train_input_impl;
using detail::train_result_impl;

train_input::train_input(const table& data, const table& labels)
        : impl_(new train_input_impl(data, labels)) {}

table train_input::get_data() const {
    return impl_->data;
}

table train_input::get_labels() const {
    return impl_->labels;
}

void train_input::set_data_impl(const table& value) {
    impl_->data = value;
}

void train_input::set_labels_impl(const table& value) {
    impl_->labels = value;
}

/* train_result implementation*/
train_result::train_result() : impl_(new train_result_impl{}) {}

model train_result::get_model() const {
    return impl_->trained_model;
}

table train_result::get_oob_err() const {
    return impl_->oob_err;
}

table train_result::get_oob_per_observation_err() const {
    return impl_->oob_per_observation_err;
}

table train_result::get_var_importance() const {
    return impl_->variable_importance;
}

void train_result::set_model_impl(const model& value) {
    impl_->trained_model = value;
}

void train_result::set_oob_err_impl(const table& value) {
    impl_->oob_err = value;
}

void train_result::set_oob_per_observation_err_impl(const table& value) {
    impl_->oob_per_observation_err = value;
}

void train_result::set_var_importance_impl(const table& value) {
    impl_->variable_importance = value;
}

} // namespace oneapi::dal::decision_forest
