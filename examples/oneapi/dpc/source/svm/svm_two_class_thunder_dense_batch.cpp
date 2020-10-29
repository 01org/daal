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

#include <CL/sycl.hpp>

#define ONEDAL_DATA_PARALLEL
#include "oneapi/dal/algo/svm.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace onedal = oneapi::dal;

void run(sycl::queue &q) {
    const std::string train_data_file_name  = get_data_path("svm_two_class_train_dense_data.csv");
    const std::string train_label_file_name = get_data_path("svm_two_class_train_dense_label.csv");
    const std::string test_data_file_name   = get_data_path("svm_two_class_test_dense_data.csv");
    const std::string test_label_file_name  = get_data_path("svm_two_class_test_dense_label.csv");

    const auto x_train = onedal::read<onedal::table>(q, onedal::csv::data_source{train_data_file_name});
    const auto y_train = onedal::read<onedal::table>(q, onedal::csv::data_source{train_label_file_name});

    const auto kernel_desc = onedal::linear_kernel::descriptor{}.set_scale(1.0).set_shift(0.0);
    const auto svm_desc    = onedal::svm::descriptor{ kernel_desc }
                              .set_c(1.0)
                              .set_accuracy_threshold(0.001)
                              .set_max_iteration_count(100)
                              .set_cache_size(200.0)
                              .set_tau(1e-6);

    const auto result_train = onedal::train(q, svm_desc, x_train, y_train);

    std::cout << "Bias:\n" << result_train.get_bias() << std::endl;
    std::cout << "Support indices:\n" << result_train.get_support_indices() << std::endl;

    const auto x_test = onedal::read<onedal::table>(q, onedal::csv::data_source{test_data_file_name});
    const auto y_true = onedal::read<onedal::table>(onedal::csv::data_source{test_label_file_name});

    const auto result_test = onedal::infer(q, svm_desc, result_train.get_model(), x_test);

    std::cout << "Decision function result:\n"
              << result_test.get_decision_function() << std::endl;
    std::cout << "Labels result:\n" << result_test.get_labels() << std::endl;
    std::cout << "Labels true:\n" << y_true << std::endl;
}

int main(int argc, char const *argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
