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

#include <iostream>

#include "example_util/utils.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/csv.hpp"

namespace dal = oneapi::dal;

using namespace dal;

int main(int argc, char **argv) {
    std::cout << "__________________" << std::endl;
    std::cout << "   Hello World" << std::endl;
    std::cout << "__________________" << std::endl;
    const auto filename = get_data_path("graph.csv");

    std::cout << "TTT" << std::endl;
    read<table>(csv::data_source{ filename });

    std::cout << "T" << std::endl;
    read<table>(csv::data_source{ filename });

    std::cout << "T" << std::endl;
    read<table>(csv::data_source{ filename }, preview::read_mode::table);
    // read<table>(csv::data_source{ filename }, preview::read_mode::edge_list);

    using graph_t = dal::preview::undirected_adjacency_vector_graph<>;
    {
        std::cout << "G" << std::endl;
        auto my_graph = read<graph_t>(csv::data_source{ filename });
        std::cout << "Number of vertices: " << dal::preview::get_vertex_count(my_graph)
                  << std::endl;
        std::cout << "Number of edges: " << dal::preview::get_edge_count(my_graph) << std::endl;
    }

    {
        std::cout << "G2" << std::endl;
        auto my_graph = read<graph_t>(csv::data_source{ filename }, preview::read_mode::edge_list);
        std::cout << "Number of vertices: " << dal::preview::get_vertex_count(my_graph)
                  << std::endl;
        std::cout << "Number of edges: " << dal::preview::get_edge_count(my_graph) << std::endl;
    }
    // // read<graph_t>(csv::data_source{ filename }, preview::read_mode::table);

    {
        std::cout << "T" << std::endl;
        const auto read_args = csv::read_args<table>{}.set_read_mode(preview::read_mode::table);
        read<table>(csv::data_source{ filename }, read_args);
    }

    {
        std::cout << "GGG" << std::endl;
        std::allocator<int> my_allocator;
        const auto read_args = csv::read_args<graph_t>{}
                                   .set_read_mode(preview::read_mode::edge_list)
                                   .set_allocator(my_allocator);

        auto my_graph = read<graph_t>(csv::data_source{ filename }, read_args);
        std::cout << "Number of vertices: " << dal::preview::get_vertex_count(my_graph)
                  << std::endl;
        std::cout << "Number of edges: " << dal::preview::get_edge_count(my_graph) << std::endl;
    }

    return 0;
}
