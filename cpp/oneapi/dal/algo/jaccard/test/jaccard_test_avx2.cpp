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

#include <initializer_list>

#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel_avx2.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::jaccard::test {

class jaccard_test {
public:
    static constexpr std::int64_t nonzero_coeff_count = 8;
    static constexpr std::int64_t rows_count = 12;

    auto create_graph() {
        oneapi::dal::preview::undirected_adjacency_vector_graph<> my_graph;
        auto &graph_impl = oneapi::dal::detail::get_impl(my_graph);
        auto &vertex_allocator = graph_impl._vertex_allocator;
        auto &edge_allocator = graph_impl._edge_allocator;

        const std::int64_t vertex_count = 7;
        const std::int64_t edge_count = 8;
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

        std::int32_t *degrees = new (degrees_) std::int32_t[vertex_count]{ 1, 3, 4, 2, 3, 1, 2 };

        std::int32_t *cols =
            new (cols_) std::int32_t[cols_count]{ 1, 0, 2, 4, 1, 3, 4, 5, 2, 6, 1, 2, 6, 2, 3, 4 };
        std::int64_t *rows = new (rows_) std::int64_t[rows_count]{ 0, 1, 4, 8, 10, 13, 14, 16 };

        std::int32_t *rows_vertex =
            new (rows_vertex_) std::int32_t[rows_count]{ 0, 1, 4, 8, 10, 13, 14, 16 };

        graph_impl.set_topology(vertex_count, edge_count, rows, cols, degrees);
        graph_impl.get_topology()._rows_vertex =
            oneapi::dal::preview::detail::container<std::int32_t>::wrap(rows_vertex, rows_count);

        return my_graph;
    }

    template <typename VertexPairsDataType>
    void check_vertex_pairs_data(const VertexPairsDataType &vertex_pairs_data) {
        int correct_pair_count = 0;
        for (std::int64_t i = 0; i < nonzero_coeff_count; i++) {
            if (vertex_pairs_data[i] == vertex_pairs[i] &&
                vertex_pairs_data[i + rows_count] == vertex_pairs[i + rows_count])
                correct_pair_count++;
        }
        std::cout << std::endl;
        REQUIRE(correct_pair_count == nonzero_coeff_count);
    }

    template <typename JaccardCoeffsDataType>
    void check_jaccard_coeffs_data(const JaccardCoeffsDataType &jaccard_coeffs_data) {
        int correct_element_count = 0;
        for (std::int64_t i = 0; i < nonzero_coeff_count; i++) {
            if (Approx(jaccard_coeffs_data[i]) == jaccard_coeffs[i])
                correct_element_count++;
        }
        std::cout << std::endl;
        REQUIRE(correct_element_count == nonzero_coeff_count);
    }

private:
    static constexpr std::array<std::int64_t, 20> vertex_pairs = { 0, 0, 1, 1, 1, 2, 2, 2, 0, 0,
                                                                   0, 0, 0, 2, 1, 2, 3, 0, 1, 2 };
    static constexpr std::array<float, 8> jaccard_coeffs = { 1.0,  0.25, 1.0,      0.166667,
                                                             0.25, 0.25, 0.166667, 1.0 };
};

TEST_M(jaccard_test, "min_neigh_u > neigh_v[n_v - 1] && n_u >= 8 && n_v >= 8") {
    std::array<std::int32_t, 11> neigh_u = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    std::array<std::int32_t, 10> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                11,
                                                                10) == 0);
}

TEST_M(jaccard_test, "min_neigh_v > neigh_u[n_u - 1] && n_u >= 8 && n_v >= 8") {
    std::array<std::int32_t, 10> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::array<std::int32_t, 11> neigh_v = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                10,
                                                                11) == 0);
}

TEST_M(jaccard_test, "min_neigh_u > neigh_v[n_v - 1] && 4 <= n_u < 8 && 4 <= n_v < 8") {
    std::array<std::int32_t, 5> neigh_u = { 4, 5, 6, 7, 8 };
    std::array<std::int32_t, 4> neigh_v = { 0, 1, 2, 3 };
    REQUIRE(+oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                 neigh_v.data(),
                                                                 5,
                                                                 4) == 0);
}

TEST_M(jaccard_test, "min_neigh_v > neigh_u[n_u - 1] && 4 <= n_u < 8 && 4 <= n_v < 8") {
    std::array<std::int32_t, 4> neigh_u = { 0, 1, 2, 3 };
    std::array<std::int32_t, 5> neigh_v = { 4, 5, 6, 7, 8 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 4, 5) ==
        0);
}

TEST_M(jaccard_test, "min_neigh_u > neigh_v[n_v - 1] && n_u < 4 && n_v < 4") {
    std::array<std::int32_t, 2> neigh_u = { 3, 4 };
    std::array<std::int32_t, 3> neigh_v = { 0, 1, 2 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 2, 3) ==
        0);
}

TEST_M(jaccard_test, "min_neigh_v > neigh_u[n_u - 1] && n_u < 4 && n_v < 4") {
    std::array<std::int32_t, 3> neigh_u = { 0, 1, 2 };
    std::array<std::int32_t, 2> neigh_v = { 3, 4 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 3, 2) ==
        0);
}

TEST_M(jaccard_test,
       "min_neigh_u > max_neigh_v && min_neigh_u < neigh_v[n_v - 1] && n_u >= 8 && n_v >= 8") {
    std::array<std::int32_t, 8> neigh_u = { 12, 13, 14, 15, 16, 17, 18, 19 };
    std::array<std::int32_t, 12> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7, 13, 15, 19, 21 };
    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                8,
                                                                12) == 3);
}

TEST_M(jaccard_test,
       "min_neigh_v > max_neigh_u && min_neigh_v < neigh_u[n_u - 1] && n_u >= 8 && n_v >= 8") {
    std::array<std::int32_t, 12> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7, 13, 15, 19, 21 };
    std::array<std::int32_t, 8> neigh_v = { 12, 13, 14, 15, 16, 17, 18, 19 };

    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                12,
                                                                8) == 3);
}

TEST_M(
    jaccard_test,
    "min_neigh_u > max_neigh_v && min_neigh_u < neigh_v[n_v - 1] && 4 <= n_u < 8 && 4 <= n_v < 8 ") {
    std::array<std::int32_t, 4> neigh_u = { 13, 14, 15, 16 };
    std::array<std::int32_t, 5> neigh_v = { 0, 1, 2, 3, 17 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 4, 5) ==
        0);
}

TEST_M(
    jaccard_test,
    "min_neigh_v > max_neigh_u && min_neigh_v < neigh_u[n_u - 1] && 4 <= n_u < 8 && 4 <= n_v < 8") {
    std::array<std::int32_t, 5> neigh_u = { 0, 1, 2, 3, 17 };
    std::array<std::int32_t, 4> neigh_v = { 13, 14, 15, 16 };

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 5, 4) ==
        0);
}

TEST_M(jaccard_test, "Several intersections for blocks of 8 elements") {
    std::array<std::int32_t, 8> neigh_u = { 3, 4, 5, 6, 7, 8, 9, 10 };
    std::array<std::int32_t, 8> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7 };

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 8, 8) ==
        5);
}

TEST_M(jaccard_test, "Several intersections for blocks of 4 elements") {
    std::array<std::int32_t, 4> neigh_u = { 2, 3, 4, 5 };
    std::array<std::int32_t, 4> neigh_v = { 0, 1, 2, 3 };

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 4, 4) ==
        2);
}

TEST_M(jaccard_test, "Several intersections for < 4 elements") {
    std::array<std::int32_t, 2> neigh_u = { 3, 4 };
    std::array<std::int32_t, 3> neigh_v = { 0, 4, 5 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 2, 3) ==
        1);
}

TEST_M(jaccard_test, "Half-vectorized branch for block of 8 elements") {
    std::array<std::int32_t, 5> neigh_u = { 2, 4, 6, 8, 10 };
    std::array<std::int32_t, 8> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7 };

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 5, 8) ==
        3);

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_v.data(), neigh_u.data(), 8, 5) ==
        3);
}

TEST_M(jaccard_test, "Half-vectorized branch for block of 4 elements") {
    std::array<std::int32_t, 2> neigh_u = { 1, 3 };
    std::array<std::int32_t, 5> neigh_v = { 0, 1, 2, 3, 4 };

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 2, 5) ==
        2);
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_v.data(), neigh_u.data(), 5, 2) ==
        2);
}

TEST_M(jaccard_test, "Equal blocks of 8 elements") {
    std::array<std::int32_t, 8> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::array<std::int32_t, 8> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 8, 8) ==
        8);
}

TEST_M(jaccard_test, "Equal blocks of 4 elements") {
    std::array<std::int32_t, 4> neigh_u = { 0, 1, 2, 3 };
    std::array<std::int32_t, 4> neigh_v = { 0, 1, 2, 3 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 4, 4) ==
        4);
}

TEST_M(jaccard_test, "Isolated vertex") {
    std::array<std::int32_t, 8> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::array<std::int32_t, 5> neigh_v = { 2, 4, 6, 8, 10 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 5, 0) ==
        0);
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 0, 5) ==
        0);
}

TEST_M(jaccard_test, "Jaccard kernel") {
    const auto jaccard_desc = dal::preview::jaccard::descriptor<>().set_block({ 0, 3 }, { 0, 4 });
    const auto my_graph = this->create_graph();
    dal::preview::jaccard::caching_builder builder;
    const auto result_vertex_similarity =
        dal::preview::vertex_similarity(jaccard_desc, my_graph, builder);

    INFO("check nonzero_coeff_count")
    const std::int64_t nonzero_coeff_count = result_vertex_similarity.get_nonzero_coeff_count();
    REQUIRE(nonzero_coeff_count == this->nonzero_coeff_count);

    INFO("check rows_count")
    const auto rows_count = result_vertex_similarity.get_vertex_pairs().get_row_count();
    REQUIRE(rows_count == this->rows_count);

    INFO("check vertex_pairs_data")
    auto vertex_pairs_table = result_vertex_similarity.get_vertex_pairs();
    homogen_table &vertex_pairs = static_cast<homogen_table &>(vertex_pairs_table);
    const auto vertex_pairs_data = vertex_pairs.get_data<int>();
    this->check_vertex_pairs_data(vertex_pairs_data);

    INFO("check jaccard_coeffs_data")
    auto coeffs_table = result_vertex_similarity.get_coeffs();
    homogen_table &coeffs = static_cast<homogen_table &>(coeffs_table);
    const auto jaccard_coeffs_data = coeffs.get_data<float>();
    this->check_jaccard_coeffs_data(jaccard_coeffs_data);
}

} // namespace oneapi::dal::algo::jaccard::test
