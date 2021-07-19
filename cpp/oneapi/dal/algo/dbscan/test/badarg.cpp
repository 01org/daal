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

#include <array>

#include "oneapi/dal/algo/dbscan/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::dbscan::test {

namespace te = dal::test::engine;

template <typename Method>
class dbscan_badarg_test : public te::algo_fixture {
public:
    static constexpr std::int64_t row_count = 5;
    static constexpr std::int64_t bad_weight_element_count = 2;

    auto get_descriptor() const {
        return dbscan::descriptor<float, Method>(1.0, 2);
    }

    table get_data() const {
        return homogen_table::wrap(compute_data_.data(), compute_data_.size(), 1);
    }

    table get_weights() const {
        return homogen_table::wrap(weights_.data(), weights_.size(), 1);
    }

    table get_bad_weights() const {
        return homogen_table::wrap(bad_weights_.data(), bad_weights_.size(), 1);
    }

    table get_two_column_weights() const {
        return homogen_table::wrap(two_column_weights_.data(), two_column_weights_.size() / 2, 2);
    }

private:
    static constexpr std::array<float, row_count> compute_data_ = { 1.0, 1.0, 2.0, 2.0, 1.0 };

    static constexpr std::array<float, row_count> weights_ = { 1.0, 1.0, 2.0, 2.0, 1.0 };

    static constexpr std::array<float, 2 * row_count> two_column_weights_ = { 1.0, 1.0, 2.0, 2.0,
                                                                              1.0, 1.0, 1.0, 2.0,
                                                                              2.0, 1.0 };

    static constexpr std::array<float, bad_weight_element_count> bad_weights_ = { 1.0, 1.0 };
};

#define DBSCAN_BADARG_TEST(name) \
    TEMPLATE_TEST_M(dbscan_badarg_test, name, "[dbscan][badarg]", method::brute_force)

DBSCAN_BADARG_TEST("accepts positive min observations") {
    REQUIRE_NOTHROW(this->get_descriptor().set_min_observations(1));
}

DBSCAN_BADARG_TEST("throws if epsilon is negative") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_epsilon(-1.0), domain_error);
}

DBSCAN_BADARG_TEST("accepts positive epsilon") {
    REQUIRE_NOTHROW(this->get_descriptor().set_epsilon(1.0));
}

// Is it reasonable in case of negative weights?
/*
DBSCAN_BADARG_TEST("throws if min_observatons is negative") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_min_observations(-1), domain_error);
}
*/

DBSCAN_BADARG_TEST("throws if weights row count does not match data row count") {
    REQUIRE_THROWS_AS(
        this->compute(this->get_descriptor(), this->get_data(), this->get_bad_weights()),
        invalid_argument);
}

DBSCAN_BADARG_TEST("throws if data is empty") {
    REQUIRE_THROWS_AS(this->compute(this->get_descriptor(), table{}, this->get_weights()),
                      invalid_argument);
}

DBSCAN_BADARG_TEST("accepts empty weights") {
    REQUIRE_NOTHROW(this->compute(this->get_descriptor(), this->get_data(), table{}));
}

DBSCAN_BADARG_TEST("throws if weights column count does not equal 1") {
    REQUIRE_THROWS_AS(
        this->compute(this->get_descriptor(), this->get_data(), this->get_two_column_weights()),
        invalid_argument);
}

DBSCAN_BADARG_TEST("accepts weights matching data dimension") {
    REQUIRE_NOTHROW(this->compute(this->get_descriptor(), this->get_data(), this->get_weights()));
}

} // namespace oneapi::dal::dbscan::test
