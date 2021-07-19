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

#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T, typename Ops, bool IsInput = std::is_same_v<T, typename Ops::input_t>>
struct ops_input_dispatcher;

template <typename T, typename Ops>
struct ops_input_dispatcher<T, Ops, /* IsInput = */ true> {
    template <typename... Args>
    auto operator()(Args&&... args) {
        return Ops{}(std::forward<Args>(args)...);
    }
};

template <typename T, typename Ops>
struct ops_input_dispatcher<T, Ops, /* IsInput = */ false> {
    template <typename Policy, typename Descriptor, typename... Args>
    auto operator()(Policy&& policy, Descriptor&& desc, Args&&... args) {
        using input_t = typename Ops::input_t;
        return Ops{}(std::forward<Policy>(policy),
                     std::forward<Descriptor>(desc),
                     input_t{ std::forward<Args>(args)... });
    }
};

template <typename T, typename Ops, bool IsInput = std::is_same_v<T, typename Ops::args_t>>
struct ops_args_dispatcher;

template <typename T, typename Ops>
struct ops_args_dispatcher<T, Ops, /* IsInput = */ true> {
    template <typename... Args>
    auto operator()(Args&&... args) {
        return Ops{}(std::forward<Args>(args)...);
    }
};

template <typename T, typename Ops>
struct ops_args_dispatcher<T, Ops, /* IsInput = */ false> {
    template <typename Policy, typename Descriptor, typename... Args>
    auto operator()(Policy&& policy, Descriptor&& desc, Args&&... args) {
        using args_t = typename Ops::args_t;
        return Ops{}(std::forward<Policy>(policy),
                     std::forward<Descriptor>(desc),
                     args_t{ std::forward<Args>(args)... });
    }
};

template <typename T, template <typename> typename Ops, bool IsPolicy = is_execution_policy_v<T>>
struct ops_policy_dispatcher;

template <typename T, template <typename> typename Ops>
struct ops_policy_dispatcher<T, Ops, /* IsPolicy = */ true> {
    template <typename Policy, typename Descriptor, typename Head, typename... Tail>
    auto operator()(Policy&& policy, Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Descriptor>>;
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(std::forward<Policy>(policy),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename T, template <typename> typename Ops>
struct ops_policy_dispatcher<T, Ops, /* IsPolicy = */ false> {
    template <typename Descriptor, typename Head, typename... Tail>
    auto operator()(Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Descriptor>>;
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(host_policy::get_default(),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename Object,
          typename T,
          template <typename, typename>
          typename Ops,
          bool IsPolicy = is_execution_policy_v<T>>
struct ops_policy_dispatcher_object;

template <typename Object, typename T, template <typename, typename> typename Ops>
struct ops_policy_dispatcher_object<Object, T, Ops, /* IsPolicy = */ true> {
    template <typename Policy, typename Descriptor>
    auto operator()(Policy&& policy, Descriptor&& desc) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using input_t = typename ops_t::args_t;
        return ops_t{}(std::forward<Policy>(policy), std::forward<Descriptor>(desc), input_t{});
    }

    template <typename Policy, typename Descriptor, typename Head, typename... Tail>
    auto operator()(Policy&& policy, Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(std::forward<Policy>(policy),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename Object, typename T, template <typename, typename> typename Ops>
struct ops_policy_dispatcher_object<Object, T, Ops, /* IsPolicy = */ false> {
    template <typename Descriptor>
    auto operator()(Descriptor&& desc) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using input_t = typename ops_t::args_t;
        return ops_t{}(host_policy::get_default(), std::forward<Descriptor>(desc), input_t{});
    }

    template <typename Descriptor, typename Head, typename... Tail>
    auto operator()(Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using dispatcher_t = ops_args_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(host_policy::get_default(),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename Object,
          typename T,
          typename Allocator,
          template <typename, typename, typename>
          typename Ops,
          bool IsPolicy = is_execution_policy_v<T>>
struct ops_policy_dispatcher_object_allocator;

template <typename Object,
          typename T,
          typename Allocator,
          template <typename, typename, typename>
          typename Ops>
struct ops_policy_dispatcher_object_allocator<Object, T, Allocator, Ops, /* IsPolicy = */ true> {
    template <typename Policy, typename Descriptor>
    auto operator()(Policy&& policy, Descriptor&& desc) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>, Allocator>;
        using input_t = typename ops_t::args_t;
        return ops_t{}(std::forward<Policy>(policy), std::forward<Descriptor>(desc), input_t{});
    }

    template <typename Policy, typename Descriptor, typename Head, typename... Tail>
    auto operator()(Policy&& policy, Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>, Allocator>;
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(std::forward<Policy>(policy),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename Object,
          typename T,
          typename Allocator,
          template <typename, typename, typename>
          typename Ops>
struct ops_policy_dispatcher_object_allocator<Object, T, Allocator, Ops, /* IsPolicy = */ false> {
    template <typename DataSource>
    auto operator()(DataSource&& ds) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<DataSource>, Allocator>;
        using input_t = typename ops_t::args_t;
        return ops_t{}(host_policy::get_default(), std::forward<DataSource>(ds), input_t{});
    }

    template <typename Descriptor, typename Head, typename... Tail>
    auto operator()(Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>, Allocator>;
        using dispatcher_t = ops_args_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(host_policy::get_default(),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

} // namespace v1

using v1::ops_input_dispatcher;
using v1::ops_args_dispatcher;
using v1::ops_policy_dispatcher;
using v1::ops_policy_dispatcher_object;
using v1::ops_policy_dispatcher_object_allocator;

} // namespace oneapi::dal::detail
