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
#include "oneapi/dal/io/csv.hpp"

#include <stdlib.h> // size_t, malloc, free
#include <new> // bad_alloc, bad_array_new_length
template <class T>
struct Mallocator {
    typedef T value_type;
    Mallocator() noexcept {} // default ctor not required
    template <class U>
    Mallocator(const Mallocator<U> &) noexcept {}
    template <class U>
    bool operator==(const Mallocator<U> &) const noexcept {
        return true;
    }
    template <class U>
    bool operator!=(const Mallocator<U> &) const noexcept {
        return false;
    }

    T *allocate(const size_t n) const {
        if (n == 0) {
            return nullptr;
        }
        if (n > static_cast<size_t>(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void *const pv = malloc(n * sizeof(T));
        if (!pv) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(pv);
    }
    void deallocate(T *const p, size_t) const noexcept {
        free(p);
    }
};

namespace dal = oneapi::dal;

using namespace dal;

int main(int argc, char **argv) {
    const auto filename = get_data_path("graph.csv");

    using graph_t = preview::undirected_adjacency_vector_graph<>;

    Mallocator<char> mallocator;
    auto read_args = csv::read_args<graph_t, Mallocator<char>>{ mallocator }.set_read_mode(
        preview::read_mode::edge_list);

    {
        auto graph = read<graph_t, csv::data_source, csv::read_args<graph_t, Mallocator<char>>>(
            csv::data_source{ filename },
            std::move(read_args));
    }
    // {
    //     auto graph = read<graph_t, csv::data_source, csv::read_args<graph_t, Mallocator<char>>>(
    //         csv::data_source{ filename },
    //         mallocator,
    //         read_mode::edge_list);
    // }

    // auto graph = read<graph_t>(csv::data_source{ filename });

    // std::cout << "Number of vertices: " << preview::get_vertex_count(graph) << std::endl;
    // std::cout << "Number of edges: " << preview::get_edge_count(graph) << std::endl;

    // preview::vertex_type<graph_t> vertex_id = 0;
    // std::cout << "Degree of " << vertex_id << ": " << preview::get_vertex_degree(graph, vertex_id)
    //           << std::endl;

    // for (preview::vertex_size_type<graph_t> i = 0; i < preview::get_vertex_count(graph); ++i) {
    //     std::cout << "Neighbors of " << i << ": ";
    //     const auto neigh = preview::get_vertex_neighbors(graph, i);
    //     for (auto u = neigh.first; u != neigh.second; ++u) {
    //         std::cout << *u << " ";
    //     }
    //     std::cout << std::endl;
    // }
    return 0;
}
