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

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "oneapi/dal/algo/knn/train.hpp"
#include "oneapi/dal/algo/knn/infer.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/metrics/classification.hpp"

namespace oneapi::dal::knn::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename TestType>
class knn_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor(std::int64_t override_class_count,
                        std::int64_t override_neighbor_count) const {
        return knn::descriptor<Float, Method, knn::task::classification>(override_class_count,
                                                                         override_neighbor_count)
            .set_result_options(knn::result_options::responses | knn::result_options::indices |
                                knn::result_options::distances);
    }

    template <typename Distance,
              typename M = Method,
              typename = oneapi::dal::knn::detail::enable_if_brute_force_t<M>>
    auto get_descriptor(std::int64_t override_class_count,
                        std::int64_t override_neighbor_count,
                        const Distance& distance) const {
        return knn::
            descriptor<Float, knn::method::brute_force, knn::task::classification, Distance>(
                   override_class_count,
                   override_neighbor_count,
                   distance)
                .set_result_options(knn::result_options::responses | knn::result_options::indices |
                                    knn::result_options::distances);
    }

    static constexpr bool is_kd_tree = std::is_same_v<Method, knn::method::kd_tree>;
    static constexpr bool is_brute_force = std::is_same_v<Method, knn::method::brute_force>;

    bool not_available_on_device() {
        return (this->get_policy().is_gpu() && is_kd_tree);
    }

    Float classification(const table& train_data,
                         const table& train_responses,
                         const table& infer_data,
                         const table& infer_responses,
                         const std::int64_t n_classes,
                         const std::int64_t n_neighbors = 1,
                         const Float tolerance = Float(1.e-5)) {
        INFO("check if data shape is expected")
        REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        REQUIRE(train_responses.get_column_count() == 1);
        REQUIRE(infer_responses.get_column_count() == 1);
        REQUIRE(infer_data.get_row_count() == infer_responses.get_row_count());
        REQUIRE(train_data.get_row_count() == train_responses.get_row_count());

        const auto knn_desc = this->get_descriptor(n_classes, n_neighbors);

        auto train_result = this->train(knn_desc, train_data, train_responses);
        auto train_model = train_result.get_model();
        auto infer_result = this->infer(knn_desc, infer_data, train_model);
        auto [prediction] = this->unpack_result(infer_result);

        const auto score_table = te::accuracy_score<Float>(infer_responses, prediction, tolerance);
        const auto score = row_accessor<const Float>(score_table).pull({ 0, -1 })[0];
        return score;
    }

    template <typename Distance,
              typename M = Method,
              typename = oneapi::dal::knn::detail::enable_if_brute_force_t<M>>
    Float classification(const table& train_data,
                         const table& train_responses,
                         const table& infer_data,
                         const table& infer_responses,
                         const std::int64_t n_classes,
                         const std::int64_t n_neighbors,
                         const Distance& distance,
                         const Float tolerance = Float(1.e-5)) {
        INFO("check if data shape is expected")
        REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        REQUIRE(train_responses.get_column_count() == 1);
        REQUIRE(infer_responses.get_column_count() == 1);
        REQUIRE(infer_data.get_row_count() == infer_responses.get_row_count());
        REQUIRE(train_data.get_row_count() == train_responses.get_row_count());

        const auto knn_desc = this->get_descriptor(n_classes, n_neighbors, distance);

        auto train_result = this->train(knn_desc, train_data, train_responses);
        auto train_model = train_result.get_model();
        auto infer_result = this->infer(knn_desc, infer_data, train_model);
        auto [prediction] = this->unpack_result(infer_result);

        const auto score_table = te::accuracy_score<Float>(infer_responses, prediction, tolerance);
        const auto score = row_accessor<const Float>(score_table).pull({ 0, -1 })[0];
        return score;
    }

    void exact_nearest_indices_check(const table& train_data,
                                     const table& infer_data,
                                     const knn::infer_result<>& result) {
        check_nans(result);

        const auto [responses] = unpack_result(result);

        const auto gtruth = naive_knn_search(train_data, infer_data);

        INFO("check if data shape is expected")
        REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        REQUIRE(infer_data.get_row_count() == responses.get_row_count());
        REQUIRE(responses.get_column_count() == 1);
        REQUIRE(infer_data.get_row_count() == gtruth.get_row_count());
        REQUIRE(train_data.get_row_count() == gtruth.get_column_count());

        const auto m = infer_data.get_row_count();

        const auto indices = naive_knn_search(train_data, infer_data);

        for (std::int64_t j = 0; j < m; ++j) {
            const auto gt_indices_row = row_accessor<const Float>(indices).pull({ j, j + 1 });
            const auto te_indices_row = row_accessor<const Float>(responses).pull({ j, j + 1 });
            const auto l = gt_indices_row[0];
            const auto r = te_indices_row[0];
            if (l != r) {
                CAPTURE(l, r);
                FAIL("Indices of nearest neighbors are unequal");
            }
        }
    }

    static auto naive_knn_search(const table& train_data, const table& infer_data) {
        const auto distances_matrix = distances(train_data, infer_data);
        const auto indices_matrix = argsort(distances_matrix);

        return indices_matrix;
    }

    static auto distances(const table& train_data, const table& infer_data) {
        const auto m = train_data.get_row_count();
        const auto n = infer_data.get_row_count();
        const auto d = infer_data.get_column_count();

        auto distances_arr = array<Float>::zeros(m * n);
        auto* distances_ptr = distances_arr.get_mutable_data();

        for (std::int64_t j = 0; j < n; ++j) {
            const auto queue_row = row_accessor<const Float>(infer_data).pull({ j, j + 1 });
            for (std::int64_t i = 0; i < m; ++i) {
                const auto train_row = row_accessor<const Float>(train_data).pull({ i, i + 1 });
                for (std::int64_t s = 0; s < d; ++s) {
                    const auto diff = queue_row[s] - train_row[s];
                    distances_ptr[j * m + i] += diff * diff;
                }
            }
        }
        return de::homogen_table_builder{}.reset(distances_arr, n, m).build();
    }

    static auto argsort(const table& distances) {
        const auto n = distances.get_row_count();
        const auto m = distances.get_column_count();

        auto indices = array<std::int32_t>::zeros(m * n);
        auto indices_ptr = indices.get_mutable_data();
        for (std::int64_t j = 0; j < n; ++j) {
            const auto dist_row = row_accessor<const Float>(distances).pull({ j, j + 1 });
            auto idcs_row = &indices_ptr[j * m];
            std::iota(idcs_row, idcs_row + m, std::int32_t(0));
            const auto compare = [&](std::int32_t x, std::int32_t y) -> bool {
                return dist_row[x] < dist_row[y];
            };
            std::sort(idcs_row, idcs_row + m, compare);
        }
        return de::homogen_table_builder{}.reset(indices, n, m).build();
    }

    static auto arange(std::int64_t from, std::int64_t to) {
        auto indices_arr = array<std::int32_t>::zeros(to - from);
        auto* indices_ptr = indices_arr.get_mutable_data();
        std::iota(indices_ptr, indices_ptr + to - from, std::int32_t(from));
        return de::homogen_table_builder{}.reset(indices_arr, to - from, 1).build();
    }

    static auto arange(std::int64_t to) {
        return arange(0, to);
    }

    void check_nans(const knn::infer_result<>& result) {
        const auto [responses] = unpack_result(result);

        INFO("check if there is no NaN in responses")
        REQUIRE(te::has_no_nans(responses));
    }

    static auto unpack_result(const knn::infer_result<>& result) {
        const auto responses = result.get_responses();
        return std::make_tuple(responses);
    }
};

using knn_types = COMBINE_TYPES((float, double), (knn::method::brute_force, knn::method::kd_tree));
using knn_bf_types = COMBINE_TYPES((float, double), (knn::method::brute_force));

#define KNN_SMALL_TEST(name)                                               \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                   \
                         name,                                             \
                         "[small-dataset][knn][integration][batch][test]", \
                         knn_types)

#define KNN_SYNTHETIC_TEST(name)                                               \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                       \
                         name,                                                 \
                         "[synthetic-dataset][knn][integration][batch][test]", \
                         knn_types)

#define KNN_EXTERNAL_TEST(name)                                               \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                      \
                         name,                                                \
                         "[external-dataset][knn][integration][batch][test]", \
                         knn_types)

#define KNN_BF_EXTERNAL_TEST(name)                                            \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                      \
                         name,                                                \
                         "[external-dataset][knn][integration][batch][test]", \
                         knn_bf_types)

KNN_SMALL_TEST("knn nearest points test predefined 7x5x2") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    constexpr std::int64_t train_row_count = 7;
    constexpr std::int64_t infer_row_count = 5;
    constexpr std::int64_t column_count = 2;

    CAPTURE(train_row_count, infer_row_count, column_count);

    constexpr std::int64_t train_element_count = train_row_count * column_count;
    constexpr std::int64_t infer_element_count = infer_row_count * column_count;

    constexpr std::array<float, train_element_count> train = { -2.f, -1.f, -1.f,   -1.f,   -1.f,
                                                               -2.f, +1.f, +1.f,   +1.f,   +2.f,
                                                               +2.f, +1.f, +100.f, -1024.f };

    constexpr std::array<float, infer_element_count> infer = { +2.f, +1.f, -1.f, +3.f, -1.f,
                                                               -1.f, +1.f, +2.f, +1.f, +2.f };

    const auto x_train_table = homogen_table::wrap(train.data(), train_row_count, column_count);
    const auto x_infer_table = homogen_table::wrap(infer.data(), infer_row_count, column_count);
    const auto y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SYNTHETIC_TEST("knn nearest points test random uniform 513x301x17") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    constexpr std::int64_t train_row_count = 513;
    constexpr std::int64_t infer_row_count = 301;
    constexpr std::int64_t column_count = 17;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.3, 1.));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SYNTHETIC_TEST("knn nearest points test random uniform 16390x20x5") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    constexpr std::int64_t train_row_count = 16390;
    constexpr std::int64_t infer_row_count = 20;
    constexpr std::int64_t column_count = 5;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.3, 1.));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_EXTERNAL_TEST("knn classification hepmass 50kx10k") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    constexpr double target_score = 0.8;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_BF_EXTERNAL_TEST("knn classification hepmass 50kx10k with Minkowski distance (p = 2.5)") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    constexpr double target_score = 0.8;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::minkowski_distance::descriptor<>;
    const double minkowski_degree = 2.5;
    const auto distance_desc = distance_t(minkowski_degree);
    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors,
                                            distance_desc);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_BF_EXTERNAL_TEST("knn classification hepmass 50kx10k with Chebyshev distance") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    constexpr double target_score = 0.8;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::chebyshev_distance::descriptor<>;
    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors,
                                            distance_t{});
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

} // namespace oneapi::dal::knn::test
