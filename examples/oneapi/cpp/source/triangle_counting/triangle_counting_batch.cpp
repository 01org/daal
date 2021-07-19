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
#include "oneapi/dal/io/csv.hpp"

namespace dal = oneapi::dal;
using namespace dal::preview::triangle_counting;

int main(int argc, char** argv) {
    const auto filename = get_data_path("graph.csv");

    // read the graph
    using graph_t = dal::preview::undirected_adjacency_vector_graph<>;
    auto graph = dal::read<graph_t>(csv::data_source{ filename });
    std::allocator<char> alloc;
    // set algorithm parameters
    const auto tc_desc =
        descriptor<float, method::ordered_count, task::local_and_global, std::allocator<char>>(
            alloc);

    // compute local and global triangles
    const auto result_vertex_ranking = dal::preview::vertex_ranking(tc_desc, graph);

    // extract the result
    std::cout << "Global triangles: " << result_vertex_ranking.get_global_rank() << std::endl;
    std::cout << "Local triangles: " << std::endl;

    auto local_triangles_table = result_vertex_ranking.get_ranks();
    const auto& local_triangles = static_cast<const dal::homogen_table&>(local_triangles_table);
    const auto local_triangles_data = local_triangles.get_data<std::int64_t>();
    for (auto i = 0; i < local_triangles_table.get_row_count(); i++) {
        std::cout << i << ":\t" << local_triangles_data[i] << std::endl;
    }

    return 0;
}
