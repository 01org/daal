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

#include "oneapi/dal/io/csv/common.hpp"
#include "oneapi/dal/io/csv/detail/read_graph_kernel.hpp"

namespace oneapi::dal::csv::detail {

template <typename Policy, typename DataSource, typename Descriptor>
struct backend_base {
    virtual typename Descriptor::object_t operator()(const Policy &ctx,
                                                     const DataSource &ds,
                                                     const Descriptor &desc) = 0;
    virtual ~backend_base() = default;
};

template <typename Policy, typename DataSource, typename Descriptor>
struct backend_default : public backend_base<Policy, DataSource, Descriptor> {
    static_assert(dal::detail::is_one_of_v<Policy, dal::detail::host_policy>,
                  "Host policy only is supported.");

    virtual typename Descriptor::object_t operator()(const Policy &ctx,
                                                     const DataSource &ds,
                                                     const Descriptor &desc) {
        return read_graph_default_kernel(ctx, ds, desc);
    }
};

template <typename Policy, typename DataSource, typename Descriptor>
dal::detail::shared<backend_base<Policy, DataSource, Descriptor>> get_backend(
    const DataSource &ds,
    const Descriptor &desc) {
    return std::make_shared<backend_default<Policy, DataSource, Descriptor>>();
}

} // namespace oneapi::dal::csv::detail
