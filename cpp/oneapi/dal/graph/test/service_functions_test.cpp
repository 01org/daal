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

#include <array>

#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::graph::test {

class graph_base_data {
public:
    graph_base_data() = default;

    std::int64_t get_correct_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_correct_edge_count() const {
        return edge_count;
    }

    std::int64_t get_neighbors_count() const {
        return neighbors_count;
    }

protected:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t neighbors_count;
};

class common_graph_type : public graph_base_data {
public:
    common_graph_type() {
        vertex_count = 7;
        edge_count = 8;
        neighbors_count = 16;
    }

    std::array<std::int32_t, 7> degrees = { 1, 3, 4, 2, 3, 1, 2 };
    std::array<std::int32_t, 16> cols = { 1, 0, 2, 4, 1, 3, 4, 5, 2, 6, 1, 2, 6, 2, 3, 4 };
    std::array<std::int64_t, 8> rows = { 0, 1, 4, 8, 10, 13, 14, 16 };
};

class two_vertices_graph_type : public graph_base_data {
public:
    two_vertices_graph_type() {
        vertex_count = 2;
        edge_count = 1;
        neighbors_count = 2;
    }

    std::array<std::int32_t, 2> degrees = { 1, 1 };
    std::array<std::int32_t, 2> cols = { 1, 0 };
    std::array<std::int64_t, 3> rows = { 0, 1, 2 };
};

class acyclic_graph_type : public graph_base_data {
public:
    acyclic_graph_type() {
        vertex_count = 8;
        edge_count = 7;
        neighbors_count = 14;
    }

    std::array<std::int32_t, 8> degrees = { 3, 1, 3, 3, 1, 1, 1, 1 };
    std::array<std::int32_t, 14> cols = { 1, 2, 4, 0, 0, 3, 6, 2, 5, 7, 0, 3, 2, 3 };
    std::array<std::int64_t, 9> rows = { 0, 3, 4, 7, 10, 11, 12, 13, 14 };
};

class complete_graph_type : public graph_base_data {
public:
    complete_graph_type() {
        vertex_count = 5;
        edge_count = 10;
        neighbors_count = 20;
    }

    std::array<std::int32_t, 5> degrees = { 4, 4, 4, 4, 4 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3
    };
    std::array<std::int64_t, 6> rows = { 0, 4, 8, 12, 16, 20 };
};

class service_functions_test {
public:
    using my_graph_type = dal::preview::undirected_adjacency_vector_graph<>;

    template <typename GraphType>
    auto create_graph() {
        GraphType graph_data;
        my_graph_type my_graph;

        auto &graph_impl = oneapi::dal::detail::get_impl(my_graph);
        auto &vertex_allocator = graph_impl._vertex_allocator;
        auto &edge_allocator = graph_impl._edge_allocator;

        const std::int64_t vertex_count = graph_data.get_correct_vertex_count();
        const std::int64_t edge_count = graph_data.get_correct_edge_count();
        const std::int64_t cols_count = edge_count * 2;
        const std::int64_t rows_count = vertex_count + 1;

        std::int32_t *degrees_ =
            std::allocator_traits<std::allocator<char>>::rebind_traits<std::int32_t>::allocate(
                vertex_allocator,
                vertex_count);
        std::int32_t *cols_ =
            std::allocator_traits<std::allocator<char>>::rebind_traits<std::int32_t>::allocate(
                vertex_allocator,
                cols_count);
        std::int64_t *rows_ =
            std::allocator_traits<std::allocator<char>>::rebind_traits<std::int64_t>::allocate(
                edge_allocator,
                rows_count);
        std::int32_t *rows_vertex_ =
            std::allocator_traits<std::allocator<char>>::rebind_traits<std::int32_t>::allocate(
                vertex_allocator,
                rows_count);

        std::int32_t *degrees = new (degrees_) std::int32_t[vertex_count];
        std::int32_t *cols = new (cols_) std::int32_t[cols_count];
        std::int64_t *rows = new (rows_) std::int64_t[rows_count];
        std::int32_t *rows_vertex = new (rows_vertex_) std::int32_t[rows_count];

        for (int i = 0; i < vertex_count; i++) {
            degrees[i] = graph_data.degrees[i];
        }

        for (int i = 0; i < cols_count; i++) {
            cols[i] = graph_data.cols[i];
        }
        for (int i = 0; i < rows_count; i++) {
            rows[i] = graph_data.rows[i];
            rows_vertex[i] = graph_data.rows[i];
        }
        graph_impl.set_topology(vertex_count, edge_count, rows, cols, degrees);
        graph_impl.get_topology()._rows_vertex =
            oneapi::dal::preview::detail::container<std::int32_t>::wrap(rows_vertex, rows_count);
        return my_graph;
    }

    template <typename GraphType, std::size_t SIZE>
    void check_vertex_degree(const GraphType &graph,
                             const std::array<std::int32_t, SIZE> &correct_degrees,
                             std::int64_t vertex_count) {
        int correct_degree_count = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (dal::preview::get_vertex_degree(graph, i) == correct_degrees[i])
                correct_degree_count++;
        }
        REQUIRE(correct_degree_count == vertex_count);
    }

    template <typename GraphType, std::size_t SIZE>
    void check_vertex_neighbors(const GraphType &graph,
                                const std::array<std::int32_t, SIZE> &correct_neighbors,
                                std::int64_t vertex_count,
                                std::int64_t neighbors_count) {
        int correct_neighbors_count = 0;

        std::int64_t neighbor_index = 0;

        for (std::int64_t j = 0; j < vertex_count; j++) {
            const auto [start, end] = dal::preview::get_vertex_neighbors(graph, j);
            for (auto i = start; i != end; ++i) {
                if (*i == correct_neighbors[neighbor_index])
                    correct_neighbors_count++;
                neighbor_index++;
            }
        }

        REQUIRE(correct_neighbors_count == neighbors_count);
    }
};

TEST_M(service_functions_test, "check service functions") {
    common_graph_type common_graph;
    two_vertices_graph_type two_vertices_graph;
    acyclic_graph_type acyclic_graph;
    complete_graph_type complete_graph;

    const auto my_common_graph = create_graph<common_graph_type>();
    const auto my_two_vertices_graph = create_graph<two_vertices_graph_type>();
    const auto my_acyclic_graph = create_graph<acyclic_graph_type>();
    const auto my_complete_graph = create_graph<complete_graph_type>();
    my_graph_type my_empty_graph;

    SECTION("check get_vertex_count") {
        REQUIRE(dal::preview::get_vertex_count(my_common_graph) ==
                common_graph.get_correct_vertex_count());
        REQUIRE(dal::preview::get_vertex_count(my_two_vertices_graph) ==
                two_vertices_graph.get_correct_vertex_count());

        REQUIRE(dal::preview::get_vertex_count(my_two_vertices_graph) ==
                two_vertices_graph.get_correct_vertex_count());
        REQUIRE(dal::preview::get_vertex_count(my_acyclic_graph) ==
                acyclic_graph.get_correct_vertex_count());
        REQUIRE(dal::preview::get_vertex_count(my_complete_graph) ==
                complete_graph.get_correct_vertex_count());
        REQUIRE(dal::preview::get_vertex_count(my_empty_graph) == 0);
    }
    SECTION("check get_edge_count") {
        REQUIRE(dal::preview::get_edge_count(my_common_graph) ==
                common_graph.get_correct_edge_count());
        REQUIRE(dal::preview::get_edge_count(my_two_vertices_graph) ==
                two_vertices_graph.get_correct_edge_count());
        REQUIRE(dal::preview::get_edge_count(my_acyclic_graph) ==
                acyclic_graph.get_correct_edge_count());
        REQUIRE(dal::preview::get_edge_count(my_complete_graph) ==
                complete_graph.get_correct_edge_count());
        REQUIRE(dal::preview::get_edge_count(my_empty_graph) == 0);
    }
    SECTION("check get_vertex_degree") {
        this->check_vertex_degree(my_common_graph,
                                  common_graph.degrees,
                                  common_graph.get_correct_vertex_count());
        this->check_vertex_degree(my_two_vertices_graph,
                                  two_vertices_graph.degrees,
                                  two_vertices_graph.get_correct_vertex_count());
        this->check_vertex_degree(my_acyclic_graph,
                                  acyclic_graph.degrees,
                                  acyclic_graph.get_correct_vertex_count());
        this->check_vertex_degree(my_complete_graph,
                                  complete_graph.degrees,
                                  complete_graph.get_correct_vertex_count());
        REQUIRE_THROWS_AS(dal::preview::get_vertex_degree(my_empty_graph, 0), out_of_range);
    }
    SECTION("check get_vertex_neighbors") {
        this->check_vertex_neighbors(my_common_graph,
                                     common_graph.cols,
                                     common_graph.get_correct_vertex_count(),
                                     common_graph.get_neighbors_count());

        this->check_vertex_neighbors(my_two_vertices_graph,
                                     two_vertices_graph.cols,
                                     two_vertices_graph.get_correct_vertex_count(),
                                     two_vertices_graph.get_neighbors_count());
        this->check_vertex_neighbors(my_acyclic_graph,
                                     acyclic_graph.cols,
                                     acyclic_graph.get_correct_vertex_count(),
                                     acyclic_graph.get_neighbors_count());
        this->check_vertex_neighbors(my_complete_graph,
                                     complete_graph.cols,
                                     complete_graph.get_correct_vertex_count(),
                                     complete_graph.get_neighbors_count());
        REQUIRE_THROWS_AS(dal::preview::get_vertex_neighbors(my_empty_graph, 0), out_of_range);
    }
}

} // namespace oneapi::dal::graph::test
