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

#include "oneapi/dal/detail/ops_dispatcher.hpp"
#include "oneapi/dal/detail/common.hpp"
#pragma warning(disable : 1011)

namespace oneapi::dal::csv {
struct read_args_tag;
}

namespace oneapi::dal::detail {

namespace v1 {

template <typename Object, typename Descriptor, typename Tag, typename... Options>
struct read_ops;

template <typename Object, typename Descriptor, typename... Options>
using tagged_read_ops = read_ops<Object, Descriptor, typename Descriptor::tag_t, Options...>;

template <typename Object, typename Head, typename... Tail>
auto read_dispatch(Head&& head, Tail&&... tail) {
    using dispatcher_t = ops_policy_dispatcher_object<Object, std::decay_t<Head>, tagged_read_ops>;
    return dispatcher_t{}(std::forward<Head>(head), std::forward<Tail>(tail)...);
}

template <typename Object, typename Descriptor, typename Head, typename... Tail>
auto read_dispatch(Descriptor&& desc, Head&& head, Tail&&... tail) {
    using head_t = std::decay_t<Head>;
    if constexpr (oneapi::dal::detail::is_tagged_v<head_t>) {
        static_assert(is_tag_one_of_v<head_t, csv::read_args_tag>);
        using allocator_t = typename head_t::allocator_t;
        using dispatcher_t = ops_policy_dispatcher_object_allocator<Object,
                                                                    std::decay_t<Head>,
                                                                    allocator_t,
                                                                    tagged_read_ops>;

        return dispatcher_t{}(desc, std::forward<Head>(head), std::forward<Tail>(tail)...);
    }
    else {
        using dispatcher_t =
            ops_policy_dispatcher_object<Object, std::decay_t<Head>, tagged_read_ops>;
        return dispatcher_t{}(desc, std::forward<Head>(head), std::forward<Tail>(tail)...);
    }
}

} // namespace v1

using v1::read_dispatch;

} // namespace oneapi::dal::detail
