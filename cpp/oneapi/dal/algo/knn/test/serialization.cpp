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

#include "oneapi/dal/algo/knn/infer.hpp"
#include "oneapi/dal/algo/knn/train.hpp"
#include "oneapi/dal/table/homogen.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::knn::test {

namespace te = dal::test::engine;

template <typename TestType>
class knn_serialization_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = std::tuple_element_t<2, TestType>;
    using descriptor_t = descriptor<float_t, method_t, task_t>;

    static constexpr bool is_kd_tree = std::is_same_v<method_t, knn::method::kd_tree>;
    static constexpr bool is_brute_force = std::is_same_v<method_t, knn::method::brute_force>;
    static constexpr bool is_classification = std::is_same_v<task_t, knn::task::classification>;
    static constexpr bool is_search = std::is_same_v<task_t, knn::task::search>;

    bool not_available_on_device() {
        return (this->get_policy().is_gpu() && is_kd_tree);
    }

    void set_class_count(std::int64_t class_count) {
        class_count_ = class_count;
    }

    void set_neighbor_count(std::int64_t neighbor_count) {
        neighbor_count_ = neighbor_count;
    }

    std::tuple<table, table> get_train_data() {
        // TODO: Replace by classification dataset generator

        constexpr std::int64_t row_count = 21;
        constexpr std::int64_t feature_count = 3;
        static const float_t x_train[] = {
            -0.543,  0.6576,  0.2046, //
            0.33,    -1.4263, 1.3322, //
            -0.1936, -0.1364, -0.6573, //
            1.1793,  -1.0809, 0.7298, //
            -2.8212, -0.4471, -0.5333, //
            1.122,   0.4834,  0.2969, //
            -2.0703, -1.7256, 0.6822, //
            0.2969,  -1.004,  -0.5835, //
            -1.1765, -2.2248, 0.7409, //
            -1.6578, -2.9339, 1.0975, //
            0.4779,  0.3422,  -1.4021, //
            -0.6799, 0.7451,  -0.7207, //
            -0.086,  -2.1414, -1.8213, //
            -2.3201, -0.8528, -1.3286, //
            -0.2933, 0.5726,  -0.0968, //
            0.0617,  1.0032,  -0.4763, //
            -0.6552, -1.4016, 2.3133, //
            -0.2874, -0.9861, -0.2129, //
            0.1375,  -0.326,  -1.3081, //
            -1.8112, 0.2395,  -0.2247, //
            1.7241,  -0.4099, -1.7277 //
        };

        static const std::int32_t y_train_two_cls[] = {
            1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
        };
        static const std::int32_t y_train_three_cls[] = {
            1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 2, 1, 0, 2, 2, 2, 0, 1, 0, 0, 1,
        };

        ONEDAL_ASSERT(class_count_ == 2 || class_count_ == 3);
        const std::int32_t* y_train = (class_count_ == 2) ? y_train_two_cls : y_train_three_cls;

        return { homogen_table::wrap(x_train, row_count, feature_count),
                 homogen_table::wrap(y_train, row_count, 1) };
    }

    table get_test_data() {
        constexpr std::int64_t row_count = 9;
        constexpr std::int64_t feature_count = 3;
        static const float_t x_test[] = {
            0.4213,  1.5162,  0.1232, //
            -0.9252, -0.514,  -0.4708, //
            0.87,    0.8338,  3.4986, //
            1.1788,  -0.5077, 4.5583, //
            -1.4726, 2.2684,  -1.8172, //
            -0.0091, -1.0776, 1.5261, //
            -0.5016, -2.9235, 0.3886, //
            0.4889,  -0.2868, 0.6805, //
            -2.0861, 0.9992,  -1.9605, //
        };
        return homogen_table::wrap(x_test, row_count, feature_count);
    }

    descriptor_t get_descriptor() {
        ONEDAL_ASSERT(class_count_ > 0);
        ONEDAL_ASSERT(neighbor_count_ > 0);
        return descriptor_t(class_count_, neighbor_count_);
    }

    model<task_t> train_model() {
        const auto [x_train, y_train] = this->get_train_data();
        return this->train(this->get_descriptor(), x_train, y_train).get_model();
    }

    infer_result<task_t> run_inference(const model<task_t>& m) {
        return this->infer(this->get_descriptor(), this->get_test_data(), m);
    }

    void compare_infer_results(const infer_result<task_t>& actual,
                               const infer_result<task_t>& reference) {
        if constexpr (is_classification) {
            INFO("compare responses") {
                te::check_if_tables_equal<float_t>(actual.get_responses(),
                                                   reference.get_responses());
            }
        }
        if constexpr (is_search) {
            INFO("compare indices") {
                te::check_if_tables_equal<float_t>(actual.get_indices(), reference.get_indices());
            }
            INFO("compare distances") {
                te::check_if_tables_equal<float_t>(actual.get_distances(),
                                                   reference.get_distances());
            }
        }
    }

    void run_test() {
        INFO("training");
        const auto model = train_model();

        INFO("serialization");
        const auto deserialized_model = te::serialize_deserialize(model);

        INFO("inference");
        const auto actual = run_inference(model);
        const auto expected = run_inference(deserialized_model);
        compare_infer_results(actual, expected);
    }

private:
    std::int64_t class_count_ = -1;
    std::int64_t neighbor_count_ = -1;
};

using knn_types = COMBINE_TYPES((float, double),
                                (knn::method::kd_tree, knn::method::brute_force),
                                (knn::task::classification, knn::task::search));

TEMPLATE_LIST_TEST_M(knn_serialization_test,
                     "serialize/deserialize knn models",
                     "[cls]",
                     knn_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->not_available_on_device());

    const std::int64_t class_count = GENERATE(2, 3);
    const std::int64_t neighbor_count = GENERATE(2, 3, 15);

    this->set_class_count(class_count);
    this->set_neighbor_count(neighbor_count);
    this->run_test();
}

} // namespace oneapi::dal::knn::test
