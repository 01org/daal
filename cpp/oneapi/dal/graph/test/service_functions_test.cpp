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

#include <array>

#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/graph/directed_adjacency_vector_graph.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_builder.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::graph::test {
class graph_base_data {
public:
    graph_base_data() = default;

    std::int64_t get_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_edge_count() const {
        return edge_count;
    }

    std::int64_t get_cols_count() const {
        return cols_count;
    }

    std::int64_t get_rows_count() const {
        return rows_count;
    }

protected:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t cols_count;
    std::int64_t rows_count;
};

class complete_graph_5_type : public graph_base_data {
public:
    complete_graph_5_type() {
        vertex_count = 5;
        edge_count = 10;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 5> degrees = { 4, 4, 4, 4, 4 };
    std::array<std::int32_t, 20> cols = {
        1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3
    };
    std::array<std::int64_t, 6> rows = { 0, 4, 8, 12, 16, 20 };
};

class d_complete_graph_5_type : public complete_graph_5_type {
public:
    d_complete_graph_5_type() {
        edge_count = 20;
        cols_count = edge_count;
    }
    std::array<std::int32_t, 5> degrees = { 4, 4, 4, 4, 4 };
    std::array<double, 20> edge_weights_double = { 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    std::array<std::int32_t, 20> edge_weights_int = { 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
};

class graph_with_isolated_vertices_10_type : public graph_base_data {
public:
    graph_with_isolated_vertices_10_type() {
        vertex_count = 10;
        edge_count = 11;
        cols_count = edge_count * 2;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 10> degrees = { 5, 3, 2, 0, 3, 4, 0, 2, 0, 3 };
    std::array<std::int32_t, 22> cols = { 1, 2, 4, 5, 7, 0, 5, 9, 0, 7, 0,
                                          5, 9, 0, 1, 4, 9, 0, 2, 1, 4, 5 };
    std::array<std::int64_t, 11> rows = { 0, 5, 8, 10, 10, 13, 17, 17, 19, 19, 22 };
};

class d_graph_with_isolated_vertices_10_type : public graph_base_data {
public:
    d_graph_with_isolated_vertices_10_type() {
        vertex_count = 10;
        edge_count = 11;
        cols_count = edge_count;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 10> degrees = { 2, 2, 1, 0, 1, 0, 0, 2, 0, 3 };
    std::array<std::int32_t, 11> cols = { 4, 5, 0, 5, 0, 5, 0, 2, 1, 4, 5 };
    std::array<std::int64_t, 11> rows = { 0, 2, 4, 5, 5, 6, 6, 6, 8, 8, 11 };
};

class empty_graph_type : public graph_base_data {
public:
    empty_graph_type() {
        vertex_count = 10;
        edge_count = 0;
        cols_count = 0;
        rows_count = vertex_count + 1;
    }

    std::array<std::int32_t, 10> degrees = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    std::array<std::int32_t, 0> cols = {};
    std::array<std::int64_t, 11> rows = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
};

class service_functions_test {
public:
    template <typename Graph, typename GraphType>
    auto create_graph() {
        GraphType graph_data;
        Graph g;
        auto &graph_impl = oneapi::dal::detail::get_impl(g);
        auto &vertex_allocator = graph_impl._vertex_allocator;
        auto &edge_allocator = graph_impl._edge_allocator;

        const std::int64_t vertex_count = graph_data.get_vertex_count();
        const std::int64_t edge_count = graph_data.get_edge_count();
        const std::int64_t cols_count = graph_data.get_cols_count();
        const std::int64_t rows_count = graph_data.get_rows_count();

        std::int32_t *degrees =
            oneapi::dal::preview::detail::allocate(vertex_allocator, vertex_count);
        std::int32_t *cols = oneapi::dal::preview::detail::allocate(vertex_allocator, cols_count);
        std::int64_t *rows = oneapi::dal::preview::detail::allocate(edge_allocator, rows_count);
        std::int32_t *rows_vertex =
            oneapi::dal::preview::detail::allocate(vertex_allocator, rows_count);

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
        graph_impl.set_topology(vertex_count, edge_count, rows, cols, cols_count, degrees);
        graph_impl.get_topology()._rows_vertex =
            oneapi::dal::preview::detail::container<std::int32_t>::wrap(rows_vertex, rows_count);
        return g;
    }

    template <typename UndirectedGraphType>
    void check_vertex_count_undir() {
        UndirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::undirected_adjacency_vector_graph<>, UndirectedGraphType>();
        REQUIRE(dal::preview::get_vertex_count(g) == graph_data.get_vertex_count());
    }

    template <typename DirectedGraphType>
    void check_vertex_count_dir() {
        DirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::directed_adjacency_vector_graph<>, DirectedGraphType>();
        REQUIRE(dal::preview::get_vertex_count(g) == graph_data.get_vertex_count());
    }

    template <typename UndirectedGraphType>
    auto check_edge_count_undir() {
        UndirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::undirected_adjacency_vector_graph<>, UndirectedGraphType>();
        REQUIRE(dal::preview::get_edge_count(g) == graph_data.get_edge_count());
    }

    template <typename DirectedGraphType>
    auto check_edge_count_dir() {
        DirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::directed_adjacency_vector_graph<>, DirectedGraphType>();
        REQUIRE(dal::preview::get_edge_count(g) == graph_data.get_edge_count());
    }

    template <typename UndirectedGraphType>
    void check_vertex_degree() {
        UndirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::undirected_adjacency_vector_graph<>, UndirectedGraphType>();
        std::int64_t vertex_count = graph_data.get_vertex_count();
        int correct_degree_count = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (dal::preview::get_vertex_degree(g, i) == graph_data.degrees[i])
                correct_degree_count++;
        }
        REQUIRE(correct_degree_count == vertex_count);
    }

    template <typename UndirectedGraphType>
    void check_vertex_degree(std::int64_t u) {
        UndirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::undirected_adjacency_vector_graph<>, UndirectedGraphType>();
        REQUIRE(dal::preview::get_vertex_degree(g, u) == graph_data.degrees[u]);
    }

    template <typename DirectedGraphType>
    void check_vertex_outward_degree() {
        DirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::directed_adjacency_vector_graph<>, DirectedGraphType>();
        std::int64_t vertex_count = graph_data.get_vertex_count();
        int correct_degree_count = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (dal::preview::get_vertex_outward_degree(g, i) == graph_data.degrees[i])
                correct_degree_count++;
        }
        REQUIRE(correct_degree_count == vertex_count);
    }

    template <typename DirectedGraphType>
    void check_vertex_outward_degree(std::int64_t u) {
        DirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::directed_adjacency_vector_graph<>, DirectedGraphType>();
        REQUIRE(dal::preview::get_vertex_outward_degree(g, u) == graph_data.degrees[u]);
    }

    template <typename UndirectedGraphType>
    void check_vertex_neighbors() {
        UndirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::undirected_adjacency_vector_graph<>, UndirectedGraphType>();
        std::int64_t vertex_count = graph_data.get_vertex_count();
        std::int64_t neighbors_count = graph_data.get_cols_count();

        std::int64_t correct_neighbors_count = 0;
        std::int64_t neighbor_index = 0;

        for (std::int64_t j = 0; j < vertex_count; j++) {
            const auto [start, end] = dal::preview::get_vertex_neighbors(g, j);
            for (auto i = start; i != end; ++i) {
                if (*i == graph_data.cols[neighbor_index])
                    correct_neighbors_count++;
                neighbor_index++;
            }
        }
        REQUIRE(correct_neighbors_count == neighbors_count);
    }

    template <typename UndirectedGraphType>
    void check_vertex_neighbors(std::int64_t u) {
        UndirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::undirected_adjacency_vector_graph<>, UndirectedGraphType>();
        const auto [start, end] = dal::preview::get_vertex_neighbors(g, u);
        std::int64_t correct_neighbors_count = 0;
        std::int64_t neighbor_index = graph_data.rows[u];
        std::int64_t neighbors_count = graph_data.degrees[u];
        for (auto i = start; i != end; ++i) {
            if (*i == graph_data.cols[neighbor_index]) {
                correct_neighbors_count++;
            }
            neighbor_index++;
        }
        REQUIRE(neighbor_index == graph_data.rows[u] + neighbors_count);
        REQUIRE(correct_neighbors_count == neighbors_count);
    }

    template <typename DirectedGraphType>
    void check_vertex_outward_neighbors() {
        DirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::directed_adjacency_vector_graph<>, DirectedGraphType>();
        std::int64_t vertex_count = graph_data.get_vertex_count();
        std::int64_t neighbors_count = graph_data.get_cols_count();

        std::int64_t correct_neighbors_count = 0;
        std::int64_t neighbor_index = 0;

        for (std::int64_t j = 0; j < vertex_count; j++) {
            const auto [start, end] = dal::preview::get_vertex_outward_neighbors(g, j);
            for (auto i = start; i != end; ++i) {
                if (*i == graph_data.cols[neighbor_index])
                    correct_neighbors_count++;
                neighbor_index++;
            }
        }
        REQUIRE(correct_neighbors_count == neighbors_count);
    }

    template <typename DirectedGraphType>
    void check_vertex_outward_neighbors(std::int64_t u) {
        DirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::directed_adjacency_vector_graph<>, DirectedGraphType>();
        const auto [start, end] = dal::preview::get_vertex_outward_neighbors(g, u);
        std::int64_t correct_neighbors_count = 0;
        std::int64_t neighbor_index = graph_data.rows[u];
        std::int64_t neighbors_count = graph_data.degrees[u];
        for (auto i = start; i != end; ++i) {
            if (*i == graph_data.cols[neighbor_index])
                correct_neighbors_count++;
            neighbor_index++;
        }
        REQUIRE(neighbor_index == graph_data.rows[u] + neighbors_count);
        REQUIRE(correct_neighbors_count == neighbors_count);
    }

    template <typename DirectedGraphType>
    void check_edge_value_int_dir() {
        DirectedGraphType graph_data;
        const auto graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
            std::int32_t,
            std::int32_t,
            oneapi::dal::preview::empty_value,
            std::int32_t,
            std::allocator<char>>(graph_data.get_vertex_count(),
                                  graph_data.get_edge_count(),
                                  graph_data.rows.data(),
                                  graph_data.cols.data(),
                                  graph_data.edge_weights_int.data());
        const auto &g = graph_builder.get_graph();
        std::int64_t vertex_count = graph_data.get_vertex_count();
        std::int64_t edge_count = graph_data.get_edge_count();
        std::int64_t correct_value_count = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            for (std::int64_t j = graph_data.rows[i]; j < graph_data.rows[i + 1]; ++j) {
                if (dal::preview::get_edge_value(g, i, graph_data.cols[j]) ==
                    graph_data.edge_weights_int[j])
                    correct_value_count++;
            }
        }
        REQUIRE(correct_value_count == edge_count);
    }

    template <typename DirectedGraphType>
    void check_edge_value_double_dir() {
        DirectedGraphType graph_data;
        const auto graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
            std::int32_t,
            double,
            oneapi::dal::preview::empty_value,
            std::int32_t,
            std::allocator<char>>(graph_data.get_vertex_count(),
                                  graph_data.get_edge_count(),
                                  graph_data.rows.data(),
                                  graph_data.cols.data(),
                                  graph_data.edge_weights_double.data());
        const auto &g = graph_builder.get_graph();
        std::int64_t vertex_count = graph_data.get_vertex_count();
        std::int64_t edge_count = graph_data.get_edge_count();
        std::int64_t correct_value_count = 0;
        for (std::int64_t i = 0; i < vertex_count; i++) {
            for (std::int64_t j = graph_data.rows[i]; j < graph_data.rows[i + 1]; ++j) {
                if (Approx(dal::preview::get_edge_value(g, i, graph_data.cols[j])) ==
                    graph_data.edge_weights_double[j])
                    correct_value_count++;
            }
        }
        REQUIRE(correct_value_count == edge_count);
    }

    template <typename DirectedGraphType>
    void check_edge_value_empty_value_dir() {
        DirectedGraphType graph_data;
        const auto g =
            create_graph<dal::preview::directed_adjacency_vector_graph<>, DirectedGraphType>();
        REQUIRE_NOTHROW(dal::preview::get_edge_value(g, 0, 1));
    }
};

#define SERVICE_FUNCTIONS_TEST(name) TEST_M(service_functions_test, name, "[service_functions]")

SERVICE_FUNCTIONS_TEST("Check get_vertex_count on undirected complete graph") {
    this->check_vertex_count_undir<complete_graph_5_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_vertex_count on directed complete graph") {
    this->check_vertex_count_dir<d_complete_graph_5_type>();
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_count on undirected graph with several isolated vertices") {
    this->check_vertex_count_undir<graph_with_isolated_vertices_10_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_vertex_count on directed graph with several isolated vertices") {
    this->check_vertex_count_dir<d_graph_with_isolated_vertices_10_type>();
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_count on undirected empty graph (all vertices are isolated)") {
    this->check_vertex_count_undir<empty_graph_type>();
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_count on directed empty graph (all vertices are isolated)") {
    this->check_vertex_count_dir<empty_graph_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_vertex_count on undirected null graph") {
    dal::preview::undirected_adjacency_vector_graph<> null_graph;
    REQUIRE(dal::preview::get_vertex_count(null_graph) == 0);
}

SERVICE_FUNCTIONS_TEST("Check get_vertex_count on directed null graph") {
    dal::preview::directed_adjacency_vector_graph<> null_graph;
    REQUIRE(dal::preview::get_vertex_count(null_graph) == 0);
}

SERVICE_FUNCTIONS_TEST("Check get_edge_count on undirected complete graph") {
    this->check_edge_count_undir<complete_graph_5_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_edge_count on directed complete graph") {
    this->check_edge_count_dir<d_complete_graph_5_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_edge_count on undirected graph with several isolated vertices") {
    this->check_edge_count_undir<graph_with_isolated_vertices_10_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_edge_count on directed graph with several isolated vertices") {
    this->check_edge_count_dir<d_graph_with_isolated_vertices_10_type>();
}

SERVICE_FUNCTIONS_TEST(
    "Check get_edge_count on undirected empty graph (all vertices are isolated)") {
    this->check_edge_count_undir<empty_graph_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_edge_count on directed empty graph (all vertices are isolated)") {
    this->check_edge_count_dir<empty_graph_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_edge_count on undirected null graph") {
    dal::preview::undirected_adjacency_vector_graph<> null_graph;
    REQUIRE(dal::preview::get_edge_count(null_graph) == 0);
}

SERVICE_FUNCTIONS_TEST("Check get_edge_count on directed null graph") {
    dal::preview::directed_adjacency_vector_graph<> null_graph;
    REQUIRE(dal::preview::get_edge_count(null_graph) == 0);
}

SERVICE_FUNCTIONS_TEST("Check get_vertex_degree on undirected complete graph") {
    this->check_vertex_degree<complete_graph_5_type>();
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_degree on undirected graph with several isolated vertices, non-isolated input vertex") {
    for (int i = 0; i < 10; ++i) {
        if ((i != 3) && (i != 6) && (i != 8))
            this->check_vertex_degree<graph_with_isolated_vertices_10_type>(i);
    }
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_degree on undirected graph with several isolated vertices, isolated input vertex") {
    this->check_vertex_degree<graph_with_isolated_vertices_10_type>(3);
    this->check_vertex_degree<graph_with_isolated_vertices_10_type>(6);
    this->check_vertex_degree<graph_with_isolated_vertices_10_type>(8);
}

SERVICE_FUNCTIONS_TEST("Check get_vertex_degree on undirected empty graph") {
    this->check_vertex_degree<empty_graph_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_vertex_outward_degree on directed complete graph") {
    this->check_vertex_outward_degree<d_complete_graph_5_type>();
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_outward_degree on directed graph with several isolated vertices, non-isolated input vertex, outward_degree != 0") {
    for (int i = 0; i < 5; ++i) {
        this->check_vertex_outward_degree<d_graph_with_isolated_vertices_10_type>(i);
    }
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_outward_degree on directed graph with several isolated vertices, non-isolated input vertex, outward_degree = 0") {
    this->check_vertex_outward_degree<d_graph_with_isolated_vertices_10_type>(5);
    this->check_vertex_outward_degree<d_graph_with_isolated_vertices_10_type>(7);
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_outward_degree on directed graph with several isolated vertices, isolated input vertex") {
    this->check_vertex_outward_degree<d_graph_with_isolated_vertices_10_type>(3);
    this->check_vertex_outward_degree<d_graph_with_isolated_vertices_10_type>(6);
    this->check_vertex_outward_degree<d_graph_with_isolated_vertices_10_type>(8);
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_outward_degree on directed empty graph (all vertices are isolated)") {
    this->check_vertex_outward_degree<empty_graph_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_vertex_neighbors on undirected complete graph") {
    this->check_vertex_neighbors<complete_graph_5_type>();
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_neighbors on undirected graph with several isolated vertices, non-isolated input vertex") {
    for (int i = 0; i < 10; ++i) {
        if ((i != 3) && (i != 6) && (i != 8))
            this->check_vertex_neighbors<graph_with_isolated_vertices_10_type>(i);
    }
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_neighbors on undirected graph with several isolated vertices, isolated input vertex") {
    this->check_vertex_neighbors<graph_with_isolated_vertices_10_type>(3);
    this->check_vertex_neighbors<graph_with_isolated_vertices_10_type>(6);
    this->check_vertex_neighbors<graph_with_isolated_vertices_10_type>(8);
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_neighbors on undirected empty graph (all vertices are isolated)") {
    this->check_vertex_neighbors<empty_graph_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_vertex_outward_neighbors on directed complete graph") {
    this->check_vertex_outward_neighbors<d_complete_graph_5_type>();
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_outward_neighbors on directed graph with several isolated vertices, non-isolated input vertex, outward_degree != 0") {
    for (int i = 0; i < 5; ++i) {
        this->check_vertex_outward_neighbors<d_graph_with_isolated_vertices_10_type>(i);
    }
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_outward_neighbors on directed graph with several isolated vertices, non-isolated input vertex, outward_degree = 0") {
    this->check_vertex_outward_neighbors<d_graph_with_isolated_vertices_10_type>(5);
    this->check_vertex_outward_neighbors<d_graph_with_isolated_vertices_10_type>(7);
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_outward_neighbors on directed graph with several isolated vertices, isolated input vertex") {
    this->check_vertex_outward_neighbors<d_graph_with_isolated_vertices_10_type>(3);
    this->check_vertex_outward_neighbors<d_graph_with_isolated_vertices_10_type>(6);
    this->check_vertex_outward_neighbors<d_graph_with_isolated_vertices_10_type>(8);
}

SERVICE_FUNCTIONS_TEST(
    "Check get_vertex_outward_neighbors on directed empty graph (all vertices are isolated)") {
    this->check_vertex_outward_neighbors<empty_graph_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_edge_value on directed graph with empty_value weights") {
    this->check_edge_value_empty_value_dir<d_complete_graph_5_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_edge_value on directed graph with double weights") {
    this->check_edge_value_double_dir<d_complete_graph_5_type>();
}

SERVICE_FUNCTIONS_TEST("Check get_edge_value on directed graph with int32_t weights") {
    this->check_edge_value_int_dir<d_complete_graph_5_type>();
}

} // namespace oneapi::dal::graph::test
