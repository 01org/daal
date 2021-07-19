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

#include <string>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/graph/common.hpp"

namespace oneapi::dal::csv {

namespace detail {
namespace v1 {

struct data_source_tag {};
class data_source_impl;

class ONEDAL_EXPORT data_source_base : public base {
public:
    using tag_t = data_source_tag;

    explicit data_source_base(const char* file_name);

    char get_delimiter() const {
        return get_delimiter_impl();
    }

    bool get_parse_header() const {
        return get_parse_header_impl();
    }

    std::string get_file_name() const {
        return std::string(get_file_name_impl());
    }

protected:
    char get_delimiter_impl() const;
    bool get_parse_header_impl() const;
    const char* get_file_name_impl() const;

    void set_delimiter_impl(char value);
    void set_parse_header_impl(bool value);
    void set_file_name_impl(const char*);

    dal::detail::pimpl<data_source_impl> impl_;
};

} // namespace v1

using v1::data_source_tag;
using v1::data_source_impl;
using v1::data_source_base;

} // namespace detail

namespace v1 {

class data_source : public detail::data_source_base {
public:
    explicit data_source(const char* file_name) : data_source_base(file_name) {}

    explicit data_source(const std::string& file_name) : data_source_base(file_name.c_str()) {}

    auto& set_delimiter(char value) {
        set_delimiter_impl(value);
        return *this;
    }

    auto& set_parse_header(bool value) {
        set_parse_header_impl(value);
        return *this;
    }

    auto& set_file_name(const char* value) {
        set_file_name_impl(value);
        return *this;
    }

    auto& set_file_name(const std::string& value) {
        set_file_name_impl(value.c_str());
        return *this;
    }
};

} // namespace v1

using v1::data_source;

} // namespace oneapi::dal::csv

namespace oneapi::dal::preview {
enum class read_mode { table, edge_list, weighted_edge_list };
} // namespace oneapi::dal::preview
