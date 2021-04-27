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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/mocks.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename T>
class array_serialization_test : te::policy_fixture {
public:
    array<T> get_empty_array() {
        return array<T>{};
    }

    array<T> get_host_backed_array(std::int64_t count, int seed = 7777) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> distr{ -10, 10 };

        auto data = array<T>::empty(count);
        T* data_ptr = data.get_mutable_data();
        for (std::int64_t i = 0; i < count; i++) {
            data_ptr[i] = T(distr(rng));
        }

        return data;
    }

#ifdef ONEDAL_DATA_PARALLEL
    array<T> get_device_backed_array(std::int64_t count, int seed = 7777) {
        auto q = this->get_queue();
        const auto host_data = get_host_backed_array(count, seed);
        return la::matrix<T>::wrap(host_data).to_device(q).get_array();
    }
#endif

    void check_array_invariants(const array<T>& deserialized) {
        if (deserialized.get_count() > 0) {
            // We assume deserialized always has mutable data
            REQUIRE(deserialized.has_mutable_data());
            REQUIRE(deserialized.get_data() != nullptr);
            REQUIRE(deserialized.get_mutable_data() != nullptr);
            REQUIRE(deserialized.get_mutable_data() == deserialized.get_data());
        }
    }

    void compare_arrays(const array<T>& original, const array<T>& deserialized) {
        REQUIRE(original.get_count() == deserialized.get_count());

        if (original.get_count() > 0) {
            const auto original_host = la::matrix<T>::wrap(original).to_host().get_array();
            const auto deserialized_host = la::matrix<T>::wrap(deserialized).to_host().get_array();

            for (std::int64_t i = 0; i < original.get_count(); i++) {
                REQUIRE(deserialized_host[i] == original_host[i]);
            }
        }
    }

    void serialize_deserialize(const array<T>& original, array<T> deserialized = {}) {
        te::mock_archive_state state;

        INFO("serialize") {
            te::mock_output_archive ar(state);
            detail::serialize(original, ar);
        }

        INFO("deserialize") {
            te::mock_input_archive ar(state);
            detail::deserialize(deserialized, ar);

            check_array_invariants(deserialized);
            compare_arrays(original, deserialized);
        }
    }
};

using array_types = std::tuple<std::int32_t, float, double>;

TEMPLATE_LIST_TEST_M(array_serialization_test,
                     "serialize-deserialize empty array",
                     "[empty]",
                     array_types) {
    const auto empty_array = this->get_empty_array();
    this->serialize_deserialize(empty_array);
}

TEMPLATE_LIST_TEST_M(array_serialization_test,
                     "serialize-deserialize host array",
                     "[host]",
                     array_types) {
    const std::int64_t count = GENERATE(1, 10, 1000);
    const auto host_array = this->get_host_backed_array(count);

    this->serialize_deserialize(host_array);
}

#ifdef ONEDAL_DATA_PARALLEL
TEMPLATE_LIST_TEST_M(array_serialization_test,
                     "serialize device array, deserialize to host array",
                     "[device]",
                     array_types) {
    const std::int64_t count = GENERATE(1, 10, 1000);
    const auto device_array = this->get_device_backed_array(count);

    this->serialize_deserialize(device_array);
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
TEMPLATE_LIST_TEST_M(array_serialization_test,
                     "serialize/deserialize device array",
                     "[device]",
                     array_types) {
    const std::int64_t count = GENERATE(1, 10, 1000);
    const auto device_array = this->get_device_backed_array(count);

    this->serialize_deserialize(device_array);
}
#endif

} // namespace oneapi::dal::test