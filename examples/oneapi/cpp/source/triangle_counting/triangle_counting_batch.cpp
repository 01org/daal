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

#include <memory>

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/triangle_counting.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"

#include <chrono>
#include <vector>
#include <algorithm>

namespace dal = oneapi::dal;
using namespace dal::preview::triangle_counting;

using namespace std::chrono;
using namespace std;
using namespace oneapi::dal::preview;

int main(int argc, char** argv) {
    const auto filename = get_data_path(argv[1]);
    const int num_trials_custom = 5;
    // read the graph
    const dal::preview::graph_csv_data_source ds(filename);
    const dal::preview::load_graph::descriptor<> d;
    const auto my_graph = dal::preview::load_graph::load(d, ds);
    std::allocator<char> alloc;
    // set algorithm parameters
    const auto tc_desc =
        descriptor<float, method::ordered_count, task::local_and_global, std::allocator<char>>(
            alloc);
    vector<double> time;
    double median;
    for (int i = 0; i < num_trials_custom; i++) {
        auto start = high_resolution_clock::now();
        const auto result_vertex_ranking2 = dal::preview::vertex_ranking(tc_desc, my_graph);
        auto stop = high_resolution_clock::now();

        time.push_back(
            std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count());
        cout << i << " iter: " << time.back() << endl;
    }
    sort(time.begin(), time.end());
    median = time[time.size() / 2];
    cout << "Median: " << median << endl;
    // compute local and global triangles
    const auto result_vertex_ranking = dal::preview::vertex_ranking(tc_desc, my_graph);

    // extract the result
    std::cout << "Global triangles: " << result_vertex_ranking.get_global_rank() << std::endl;
    /*   std::cout << "Local triangles: " << std::endl;

    auto local_triangles_table = result_vertex_ranking.get_ranks();
    const auto& local_triangles = static_cast<const dal::homogen_table&>(local_triangles_table);
    const auto local_triangles_data = local_triangles.get_data<std::int64_t>();
    for (auto i = 0; i < local_triangles_table.get_row_count(); i++) {
        std::cout << i << ":\t" << local_triangles_data[i] << std::endl;
    }
*/
    const auto tc_desc2 =
        descriptor<float, method::ordered_count, task::global, std::allocator<char>>(alloc);
    const auto result_vertex_ranking3 = dal::preview::vertex_ranking(tc_desc2, my_graph);
    std::cout << "Global triangles from global task: " << result_vertex_ranking3.get_global_rank()
              << std::endl;
    return 0;
}
