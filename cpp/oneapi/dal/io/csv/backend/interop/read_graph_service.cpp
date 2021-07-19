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

#include "oneapi/dal/io/csv/detail/read_graph_service.hpp"
#include "src/externals/service_service.h"

namespace oneapi::dal::preview::read_graph::detail {

ONEDAL_EXPORT std::int32_t daal_string_to_int(const char* nptr, char** endptr) {
    return daal::internal::Service<>::serv_string_to_int(nptr, endptr);
}

ONEDAL_EXPORT double daal_string_to_double(const char* nptr, char** endptr) {
    return daal::internal::Service<>::serv_string_to_double(nptr, endptr);
}

} // namespace oneapi::dal::preview::read_graph::detail
