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

#include <pybind11/pybind11.h>

#include "oneapi/dal/train.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/compute.hpp"

#define ONEDAL_PARAM_DISPATCH_VALUE(value, value_case, next_dispatch, ...) \
    if (value == value_case) {                                             \
        return next_dispatch<__VA_ARGS__>(params);                         \
    }                                                                      \
    else

#define ONEDAL_PARAM_DISPATCH_SECTION_END(name) \
    { throw std::runtime_error("Invalid value for parameter <" #name ">"); }

#define ONEDAL_DECLARE_PARAMS_DISPATCHER_CTOR()                             \
    using Task = typename Input::task_t;                                    \
                                                                            \
    Policy policy;                                                          \
    Input input;                                                            \
                                                                            \
    params_dispatcher(const Policy& policy, const Input& input, const Ops&) \
            : policy(policy),                                               \
              input(input) {}

#define ONEDAL_DECLARE_PARAMS_DISPATCHER_DISPATCH(next) \
    auto dispatch(const pybind11::dict& params) {       \
        return next(params);                            \
    }

#define ONEDAL_DECLARE_PARAMS_DISPATCHER_DISPATCH_FPTYPE(next)      \
    auto dispatch_fptype(const pybind11::dict& params) {            \
        auto fptype = params["fptype"].cast<std::string>();         \
        ONEDAL_PARAM_DISPATCH_VALUE(fptype, "float", next, float)   \
        ONEDAL_PARAM_DISPATCH_VALUE(fptype, "double", next, double) \
        ONEDAL_PARAM_DISPATCH_SECTION_END(fptype)                   \
    }

#define ONEDAL_DECLARE_PARAMS_DISPATCHER_RUN_BODY(...)   \
    {                                                    \
        auto desc = get_descriptor<__VA_ARGS__>(params); \
        return Ops{}.call(policy, desc, input);          \
    }

namespace oneapi::dal::python {

struct train_ops {
    template <typename... Args>
    auto call(Args&&... args) {
        return dal::train(std::forward<Args>(args)...);
    }
};

struct infer_ops {
    template <typename... Args>
    auto call(Args&&... args) {
        return dal::infer(std::forward<Args>(args)...);
    }
};

struct compute_ops {
    template <typename... Args>
    auto call(Args&&... args) {
        return dal::compute(std::forward<Args>(args)...);
    }
};

} // namespace oneapi::dal::python
