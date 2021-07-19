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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/io/csv/read_types.hpp"
#include "oneapi/dal/io/csv/detail/select_kernel.hpp"

namespace oneapi::dal::csv::detail {
namespace v1 {

template <typename Object, typename Policy, typename... Options>
struct read_ops_dispatcher;

template <typename Graph, typename Allocator>
struct read_ops_dispatcher<Graph, dal::detail::host_policy, Allocator> {
    Graph operator()(const dal::detail::host_policy& policy,
                     const data_source_base& ds,
                     const read_args<Graph, Allocator>& args) const {
        static auto impl =
            get_backend<dal::detail::host_policy, data_source_base, read_args<Graph, Allocator>>(
                ds,
                args);
        return (*impl)(policy, ds, args);
    }
};

template <>
struct read_ops_dispatcher<table, dal::detail::host_policy> {
    table operator()(const dal::detail::host_policy& policy,
                     const data_source_base& ds,
                     const read_args<table>& args) const;
};

template <typename Object, typename DataSource, typename... Allocator>
struct read_ops;

template <typename Object, typename Allocator>
struct read_ops<Object, data_source, Allocator> {
    using args_t = read_args<Object, Allocator>;
    using result_t = Object;
    using allocator_t = Allocator;

    void check_preconditions(const data_source_base& ds, const args_t& args) const {}

    void check_postconditions(const data_source_base& ds,
                              const args_t& args,
                              const result_t& result) const {}

    template <typename Policy>
    auto operator()(const Policy& ctx, const data_source_base& ds, const args_t& args) const {
        check_preconditions(ds, args);
        auto result = read_ops_dispatcher<Object, Policy, Allocator>()(ctx, ds, args);
        check_postconditions(ds, args, result);
        return result;
    }
};

template <typename Object>
struct read_ops<Object, data_source> {
    using args_t = read_args<Object, std::allocator<float>>;
    using result_t = Object;

    void check_preconditions(const data_source_base& ds, const args_t& args) const {}

    void check_postconditions(const data_source_base& ds,
                              const args_t& args,
                              const result_t& result) const {}

    template <typename Policy>
    auto operator()(const Policy& ctx, const data_source_base& ds, const args_t& args) const {
        check_preconditions(ds, args);
        auto result = read_ops_dispatcher<Object, Policy, std::allocator<float>>()(ctx, ds, args);
        check_postconditions(ds, args, result);
        return result;
    }
};

template <>
struct read_ops<table, data_source> {
    using args_t = read_args<table>;
    using result_t = table;

    void check_preconditions(const data_source_base& ds, const args_t& args) const {}

    void check_postconditions(const data_source_base& ds,
                              const args_t& args,
                              const result_t& result) const {}

    template <typename Policy>
    auto operator()(const Policy& ctx, const data_source_base& ds, const args_t& args) const {
        check_preconditions(ds, args);
        const auto result = read_ops_dispatcher<table, Policy>()(ctx, ds, args);
        check_postconditions(ds, args, result);
        return result;
    }
};

} // namespace v1

using v1::read_ops;

} // namespace oneapi::dal::csv::detail
