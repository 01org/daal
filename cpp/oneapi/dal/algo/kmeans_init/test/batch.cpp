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

#include <unordered_set>

#include "oneapi/dal/algo/kmeans_init/compute.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::kmeans_init::test {

namespace te = dal::test::engine;

template <typename TestType>
class kmeans_init_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor(std::int64_t cluster_count) const {
        return kmeans_init::descriptor<Float, Method>{ cluster_count };
    }

    bool not_available_on_device() {
        constexpr bool is_plus_plus_dense =
            std::is_same_v<Method, kmeans_init::method::plus_plus_dense>;
        constexpr bool is_parallel_plus_dense =
            std::is_same_v<Method, kmeans_init::method::parallel_plus_dense>;
        return this->get_policy().is_gpu() && (is_plus_plus_dense || is_parallel_plus_dense);
    }

    constexpr bool is_dense() const {
        return std::is_same_v<Method, method::dense> ||
               std::is_same_v<Method, method::random_dense> ||
               std::is_same_v<Method, method::plus_plus_dense> ||
               std::is_same_v<Method, method::parallel_plus_dense>;
    }

    void dense_checks(std::int64_t cluster_count, const table& data) {
        CAPTURE(cluster_count);

        INFO("create descriptor")
        const auto desc = get_descriptor(cluster_count);

        INFO("compute");
        const auto compute_results = this->compute(desc, data);
        check_results(cluster_count, data, compute_results.get_centroids());
    }

    void check_results(std::int64_t cluster_count, const table& data, const table& centroids) {
        const std::int64_t row_count = data.get_row_count();
        const std::int64_t column_count = data.get_column_count();

        ONEDAL_ASSERT(centroids.get_row_count() == cluster_count);
        ONEDAL_ASSERT(centroids.get_column_count() == column_count);

        const auto data_array = row_accessor<const float>(data).pull();
        const auto centroid_array = row_accessor<const float>(centroids).pull();

        std::unordered_set<std::int64_t> indices;
        for (std::int64_t i = 0; i < cluster_count; i++) {
            for (std::int64_t j = 0; j < row_count; j++) {
                bool match = true;
                for (std::int64_t k = 0; k < column_count; k++) {
                    if (data_array[j * column_count + k] != centroid_array[i * column_count + k]) {
                        match = false;
                        break;
                    }
                }
                if (match && indices.insert(j).second) {
                    break;
                }
            }
        }
        REQUIRE(dal::detail::integral_cast<std::int64_t>(indices.size()) == cluster_count);
    }
};

using kmeans_init_types = _TE_COMBINE_TYPES_2((float, double),
                                              (kmeans_init::method::dense,
                                               kmeans_init::method::random_dense,
                                               kmeans_init::method::plus_plus_dense,
                                               kmeans_init::method::parallel_plus_dense));

TEMPLATE_LIST_TEST_M(kmeans_init_batch_test,
                     "kmeans init dense test",
                     "[kmeans_init][batch]",
                     kmeans_init_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(!this->is_dense());
    SKIP_IF(this->not_float64_friendly());
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t cluster_count = 4;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };
    const auto data_table = homogen_table::wrap(data, row_count, column_count);

    this->dense_checks(cluster_count, data_table);
}

} // namespace oneapi::dal::kmeans_init::test
