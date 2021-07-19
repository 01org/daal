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

#include "oneapi/dal/algo/kmeans_init/compute.hpp"
#include "oneapi/dal/table/homogen.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::kmeans_init::test {

namespace te = dal::test::engine;

template <typename TestType>
class kmeans_init_badarg_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    auto get_descriptor() const {
        return kmeans_init::descriptor<Float, Method, kmeans_init::task::init>{};
    }
    bool not_available_on_device() {
        constexpr bool is_plus_plus_dense =
            std::is_same_v<Method, kmeans_init::method::plus_plus_dense>;
        constexpr bool is_parallel_plus_dense =
            std::is_same_v<Method, kmeans_init::method::parallel_plus_dense>;
        return this->get_policy().is_gpu() && (is_plus_plus_dense || is_parallel_plus_dense);
    }
};

using kmeans_init_types = _TE_COMBINE_TYPES_2((float, double),
                                              (kmeans_init::method::dense,
                                               kmeans_init::method::random_dense,
                                               kmeans_init::method::plus_plus_dense,
                                               kmeans_init::method::parallel_plus_dense));

#define KMEANS_INIT_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(kmeans_init_badarg_test, name, "[kmeans][badarg]", kmeans_init_types)

KMEANS_INIT_BADARG_TEST("accepts positive cluster_count") {
    REQUIRE_NOTHROW(this->get_descriptor().set_cluster_count(1));
}

KMEANS_INIT_BADARG_TEST("throws if cluster_count is negative") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_cluster_count(-1), domain_error);
}

KMEANS_INIT_BADARG_TEST("throws if cluster_count is zero") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_cluster_count(0), domain_error);
}

KMEANS_INIT_BADARG_TEST("throws if  data is empty") {
    const auto desc = this->get_descriptor().set_cluster_count(2);

    REQUIRE_THROWS_AS(this->compute(desc, table{}), domain_error);
}

KMEANS_INIT_BADARG_TEST("throws if  cluster count leads to overflow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };
    const auto data_table = homogen_table::wrap(data, row_count, column_count);

    const auto desc = this->get_descriptor().set_cluster_count(0x7FFFFFFFFFFFFFFF);
    REQUIRE_THROWS_AS(this->compute(desc, data_table), range_error);
}

} // namespace oneapi::dal::kmeans_init::test
