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

#include "oneapi/dal/io/csv/detail/read_ops.hpp"
#include "oneapi/dal/io/csv/read_types.hpp"
#include "oneapi/dal/read.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename Object, typename DataSource, typename Tag, typename... Options>
struct read_ops;

template <typename Object, typename DataSource, typename Head>
struct read_ops<Object, DataSource, dal::csv::detail::data_source_tag, Head>
        : dal::csv::detail::read_ops<Object, DataSource, Head> {};

template <typename Object, typename DataSource>
struct read_ops<Object, DataSource, dal::csv::detail::data_source_tag>
        : dal::csv::detail::read_ops<Object, DataSource> {};

} // namespace v1
} // namespace oneapi::dal::detail
