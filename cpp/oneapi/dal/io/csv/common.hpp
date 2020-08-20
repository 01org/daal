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

#pragma once

#include <cstring>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::csv {

namespace detail {
struct data_source_tag {};
class data_source_impl;
} // namespace detail

class ONEAPI_DAL_EXPORT data_source : public base {
public:
    using tag_t = detail::data_source_tag;

    data_source(const char *file_name);
    data_source(const std::string &file_name);

    char get_delimiter() const;
    bool get_parse_header() const;
    const char *get_file_name() const;

    auto &set_delimiter(char value) {
        set_delimiter_impl(value);
        return *this;
    }

    auto &set_parse_header(bool value) {
        set_parse_header_impl(value);
        return *this;
    }

    auto &set_file_name(const char *value) {
        set_file_name_impl(value);
        return *this;
    }

private:
    void set_delimiter_impl(char value);
    void set_parse_header_impl(bool value);
    void set_file_name_impl(const char *);

    dal::detail::pimpl<detail::data_source_impl> impl_;
};

} // namespace oneapi::dal::csv
