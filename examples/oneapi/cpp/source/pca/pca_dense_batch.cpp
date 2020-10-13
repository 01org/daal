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

#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

template <typename Method>
void run(const dal::table& x_train, const std::string& method_name) {
    const auto pca_desc = dal::pca::descriptor<float, Method>()
        .set_component_count(5)
        .set_deterministic(true);

    const auto result_train = dal::train(pca_desc, x_train);

    std::cout << method_name << std::endl << std::endl;

    std::cout << "Eigenvectors:" << std::endl
              << result_train.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:" << std::endl
              << result_train.get_eigenvalues() << std::endl;

    const auto result_infer = dal::infer(pca_desc, result_train.get_model(), x_train);

    std::cout << "Transformed data:" << std::endl
              << result_infer.get_transformed_data() << std::endl;
}

int main(int argc, char const *argv[]) {
    const std::string train_data_file_name = get_data_path("pca_normalized.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });

    run<dal::pca::method::cov>(x_train, "Training method: Covariance");
    run<dal::pca::method::svd>(x_train, "Training method: SVD");

    return 0;
}
