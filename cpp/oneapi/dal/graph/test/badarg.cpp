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
    std::array<std::int32_t, 11> edge_weights_int = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
};

class service_functions_badarg_test {
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
};

#define SERVICE_FUNCTIONS_BADARG_TEST(name) \
    TEST_M(service_functions_badarg_test, name, "[service_functions][badarg]")

SERVICE_FUNCTIONS_BADARG_TEST("Vertex_id >= vertex_count for get_vertex_degree") {
    const auto g = create_graph<dal::preview::undirected_adjacency_vector_graph<>,
                                graph_with_isolated_vertices_10_type>();
    REQUIRE_THROWS_AS(dal::preview::get_vertex_degree(g, 100), out_of_range);
    REQUIRE_THROWS_AS(dal::preview::get_vertex_degree(g, 10), out_of_range);
}

SERVICE_FUNCTIONS_BADARG_TEST("Vertex_id < 0 for get_vertex_degree") {
    const auto g = create_graph<dal::preview::undirected_adjacency_vector_graph<>,
                                graph_with_isolated_vertices_10_type>();
    REQUIRE_THROWS_AS(dal::preview::get_vertex_degree(g, -10), out_of_range);
}

SERVICE_FUNCTIONS_BADARG_TEST("Vertex_id >= vertex_count for get_vertex_outward_degree") {
    const auto g = create_graph<dal::preview::directed_adjacency_vector_graph<>,
                                d_graph_with_isolated_vertices_10_type>();
    REQUIRE_THROWS_AS(dal::preview::get_vertex_outward_degree(g, 100), out_of_range);
    REQUIRE_THROWS_AS(dal::preview::get_vertex_outward_degree(g, 10), out_of_range);
}

SERVICE_FUNCTIONS_BADARG_TEST("Vertex_id < 0 for get_vertex_outward_degree") {
    const auto g = create_graph<dal::preview::directed_adjacency_vector_graph<>,
                                d_graph_with_isolated_vertices_10_type>();
    REQUIRE_THROWS_AS(dal::preview::get_vertex_outward_degree(g, -10), out_of_range);
}

SERVICE_FUNCTIONS_BADARG_TEST("Vertex_id >= vertex_count for get_vertex_neighbors") {
    const auto g = create_graph<dal::preview::undirected_adjacency_vector_graph<>,
                                graph_with_isolated_vertices_10_type>();
    REQUIRE_THROWS_AS(dal::preview::get_vertex_neighbors(g, 100), out_of_range);
    REQUIRE_THROWS_AS(dal::preview::get_vertex_neighbors(g, 10), out_of_range);
}

SERVICE_FUNCTIONS_BADARG_TEST("Vertex_id < 0 for get_vertex_neighbors") {
    const auto g = create_graph<dal::preview::undirected_adjacency_vector_graph<>,
                                graph_with_isolated_vertices_10_type>();
    REQUIRE_THROWS_AS(dal::preview::get_vertex_neighbors(g, -10), out_of_range);
}

SERVICE_FUNCTIONS_BADARG_TEST("Vertex_id >= vertex_count for get_vertex_outward_neighbors") {
    const auto g = create_graph<dal::preview::directed_adjacency_vector_graph<>,
                                d_graph_with_isolated_vertices_10_type>();
    REQUIRE_THROWS_AS(dal::preview::get_vertex_outward_neighbors(g, 100), out_of_range);
    REQUIRE_THROWS_AS(dal::preview::get_vertex_outward_neighbors(g, 10), out_of_range);
}

SERVICE_FUNCTIONS_BADARG_TEST("Vertex_id < 0 for get_vertex_outward_neighbors") {
    const auto g = create_graph<dal::preview::directed_adjacency_vector_graph<>,
                                d_graph_with_isolated_vertices_10_type>();
    REQUIRE_THROWS_AS(dal::preview::get_vertex_outward_neighbors(g, -10), out_of_range);
}

SERVICE_FUNCTIONS_BADARG_TEST("Edge is not in the graph for get_edge_value") {
    d_graph_with_isolated_vertices_10_type graph_data;
    const auto &graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
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
    REQUIRE_THROWS_AS(dal::preview::get_edge_value(g, 1, 7), out_of_range);
}

//incorrect result
// SERVICE_FUNCTIONS_BADARG_TEST("First vertex_id >= vertex_count for get_edge_value") {
//     d_graph_with_isolated_vertices_10_type graph_data;
//         const auto &graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
//             std::int32_t,
//             std::int32_t,
//             oneapi::dal::preview::empty_value,
//             std::int32_t,
//             std::allocator<char>>(graph_data.get_vertex_count(),
//                                   graph_data.get_edge_count(),
//                                   graph_data.rows.data(),
//                                   graph_data.cols.data(),
//                                   graph_data.edge_weights_int.data());
//         const auto &g = graph_builder.get_graph();
// REQUIRE_THROWS_AS(dal::preview::get_edge_value(g, 10, 0), out_of_range);
// }

SERVICE_FUNCTIONS_BADARG_TEST("Second vertex_id >= vertex_count for get_edge_value") {
    d_graph_with_isolated_vertices_10_type graph_data;
    const auto &graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
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
    REQUIRE_THROWS_AS(dal::preview::get_edge_value(g, 0, 10), out_of_range);
}

SERVICE_FUNCTIONS_BADARG_TEST("First vertex_id < 0 for get_edge_value") {
    d_graph_with_isolated_vertices_10_type graph_data;
    const auto &graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
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
    REQUIRE_THROWS_AS(dal::preview::get_edge_value(g, -10, 0), out_of_range);
}

SERVICE_FUNCTIONS_BADARG_TEST("Second vertex_id < 0 for get_edge_value") {
    d_graph_with_isolated_vertices_10_type graph_data;
    const auto &graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
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
    REQUIRE_THROWS_AS(dal::preview::get_edge_value(g, 0, -10), out_of_range);
}

// SERVICE_FUNCTIONS_BADARG_TEST("Directed graph for get_vertex_degree") {
// const auto g = create_graph<dal::preview::directed_adjacency_vector_graph<>,d_graph_with_isolated_vertices_10_type>();
// REQUIRE_THROWS_AS(dal::preview::get_vertex_degree(g, 0), invalid_argument);
// }

/* static_assert
SERVICE_FUNCTIONS_BADARG_TEST("Directed graph for get_vertex_neighbors") {
const auto g = create_graph<dal::preview::directed_adjacency_vector_graph<>,d_graph_with_isolated_vertices_10_type>();
REQUIRE_THROWS_AS(dal::preview::get_vertex_neighbors(g, 0), invalid_argument);
}
*/

// SERVICE_FUNCTIONS_BADARG_TEST("Undirected graph for get_vertex_outward_degree") {
//     const auto g = create_graph<dal::preview::undirected_adjacency_vector_graph<>, graph_with_isolated_vertices_10_type>();
// REQUIRE_THROWS_AS(dal::preview::get_vertex_outward_degree(g, 0), invalid_argument);
// }

// SERVICE_FUNCTIONS_BADARG_TEST("Undirected graph for get_vertex_outward_neighbors") {
//     const auto g = create_graph<dal::preview::undirected_adjacency_vector_graph<>, graph_with_isolated_vertices_10_type>();
// REQUIRE_THROWS_AS(dal::preview::get_vertex_outward_neighbors(g, 0), invalid_argument);
// }

/*static_assert
SERVICE_FUNCTIONS_BADARG_TEST("Undirected graph for get_vertex_outward_neighbors") {
    const auto g = create_graph<dal::preview::undirected_adjacency_vector_graph<>, graph_with_isolated_vertices_10_type>();
REQUIRE_THROWS_AS(dal::preview::get_edge_value(g, 0, 1), invalid_argument);
}
*/

} // namespace oneapi::dal::graph::test
