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

#include "tbb/global_control.h"
#include "tbb/parallel_for.h"

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/jaccard.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace dal = oneapi::dal;

using namespace dal;

/// Computes Jaccard similarity coefficients for the graph. The upper triangular
/// matrix is processed only as it is symmetic for undirected graph.
///
/// @param [in]   g  The input graph
/// @param [in]   block_row_count    The size of block by rows
/// @param [in]   block_column_count The size of block by columns
template <class Graph>
void vertex_similarity_block_processing(const Graph &g,
                                        std::int32_t block_row_count,
                                        std::int32_t block_column_count);

int main(int argc, char **argv) {
    // load the graph
    const auto filename = get_data_path("graph.csv");

    using graph_t = preview::undirected_adjacency_vector_graph<>;
    const auto graph = read<graph_t>(csv::data_source{ filename });

    // set the block sizes for Jaccard similarity block processing
    const std::int32_t block_row_count = 2;
    const std::int32_t block_column_count = 5;

    // set the number of threads
    const std::int32_t tbb_threads_number = 4;
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, tbb_threads_number);

    // compute Jaccard similarity coefficients for the graph
    vertex_similarity_block_processing(graph, block_row_count, block_column_count);

    return 0;
}

template <class Graph>
void vertex_similarity_block_processing(const Graph &g,
                                        std::int32_t block_row_count,
                                        std::int32_t block_column_count) {
    // create caching builders for all threads
    std::vector<preview::jaccard::caching_builder> processing_blocks(
        tbb::this_task_arena::max_concurrency());

    // compute the number of vertices in graph
    const std::int32_t vertex_count = preview::get_vertex_count(g);

    // compute the number of rows
    std::int32_t row_count = vertex_count / block_row_count;
    if (vertex_count % block_row_count) {
        row_count++;
    }

    // parallel processing by rows
    tbb::parallel_for(
        tbb::blocked_range<std::int32_t>(0, row_count),
        [&](const tbb::blocked_range<std::int32_t> &r) {
            for (std::int32_t i = r.begin(); i != r.end(); ++i) {
                // compute the range of rows
                const std::int32_t row_range_begin = i * block_row_count;
                const std::int32_t row_range_end = (i + 1) * block_row_count;

                // start column ranges from diagonal
                const std::int32_t column_begin = 1 + row_range_begin;

                // compute the number of columns
                std::int32_t column_count = (vertex_count - column_begin) / block_column_count;
                if ((vertex_count - column_begin) % block_column_count) {
                    column_count++;
                }

                // parallel processing by columns
                tbb::parallel_for(
                    tbb::blocked_range<std::int32_t>(0, column_count),
                    [&](const tbb::blocked_range<std::int32_t> &inner_r) {
                        for (std::int32_t j = inner_r.begin(); j != inner_r.end(); ++j) {
                            // compute the range of columns
                            const std::int32_t column_range_begin =
                                column_begin + j * block_column_count;
                            const std::int32_t column_range_end =
                                column_begin + (j + 1) * block_column_count;

                            // set block ranges for the vertex similarity algorithm
                            const auto jaccard_desc = preview::jaccard::descriptor<>().set_block(
                                { row_range_begin, std::min(row_range_end, vertex_count) },
                                { column_range_begin, std::min(column_range_end, vertex_count) });

                            // compute Jaccard coefficients for the block
                            preview::vertex_similarity(
                                jaccard_desc,
                                g,
                                processing_blocks[tbb::this_task_arena::current_thread_index()]);

                            // do application specific postprocessing of the result here
                        }
                    },
                    tbb::simple_partitioner{});
            }
        },
        tbb::simple_partitioner{});
}
