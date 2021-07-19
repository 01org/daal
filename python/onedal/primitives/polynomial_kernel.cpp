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

#include "onedal/primitives/kernel_functions.hpp"
#include "onedal/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

template <typename Float, typename Method, typename Task>
auto get_descriptor(const py::dict& params) {
    using namespace polynomial_kernel;
    return get_kernel_descriptor<descriptor<Float, Method, Task>>(params);
}

template <typename Policy, typename Input, typename Ops>
struct params_dispatcher {
    ONEDAL_DECLARE_PARAMS_DISPATCHER_CTOR()
    ONEDAL_DECLARE_PARAMS_DISPATCHER_DISPATCH(dispatch_fptype)
    ONEDAL_DECLARE_PARAMS_DISPATCHER_DISPATCH_FPTYPE(dispatch_method)

    template <typename Float>
    auto dispatch_method(const py::dict& params) {
        using namespace polynomial_kernel;

        auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", run, Float, method::dense, Task)
        ONEDAL_PARAM_DISPATCH_SECTION_END(method)
    }

    template <typename Float, typename Method, typename Task>
    auto run(const py::dict& params) {
        ONEDAL_DECLARE_PARAMS_DISPATCHER_RUN_BODY(Float, Method, Task)
    }
};

template <typename Policy, typename Input, typename Result>
void init_compute_ops(py::module_& m) {
    m.def("compute",
          [](const Policy& policy, const py::dict& params, const table& x, const table& y) {
              return params_dispatcher{ policy, Input{ x, y }, compute_ops{} }.dispatch(params);
          });
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_kernel_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);

ONEDAL_PY_INIT_MODULE(polynomial_kernel) {
    using namespace dal::detail;
    using namespace polynomial_kernel;
    using input_t = compute_input<task::compute>;
    using result_t = compute_result<task::compute>;
    using policy_list = types<host_policy, data_parallel_policy>;

    auto sub = m.def_submodule("polynomial_kernel");
    ONEDAL_PY_INSTANTIATE(init_kernel_result, sub, result_t);
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list, input_t, result_t);
}

} // namespace oneapi::dal::python
