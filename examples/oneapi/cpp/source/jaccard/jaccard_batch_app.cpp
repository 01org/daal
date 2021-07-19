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

#include <iostream>

#include "tbb/global_control.h"
#include "tbb/parallel_for.h"

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/jaccard.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include <chrono>
#include <vector>
#include "example_util/output_helpers_graph.hpp"

using namespace std::chrono;
using namespace std;
using namespace oneapi::dal::preview;

using namespace oneapi::dal;

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
    // read the graph
    if (argc < 2)
        return 0;
    std::string filename = argv[1];
    graph_csv_data_source ds(filename);
    load_graph::descriptor<> d;
    auto my_graph = load_graph::load(d, ds);

    int num_trials_custom = 3;
    int num_trials_lib = 1;

    int32_t row_size = 1;
    int32_t column_size = 16000;

    int32_t tbb_threads_number = 24;

    if (argc > 5) {
        row_size = stoi(argv[2]);
        column_size = stoi(argv[3]);
        num_trials_custom = stoi(argv[4]);
        tbb_threads_number = stoi(argv[5]);
    }
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, tbb_threads_number);
    //cout << "jaccard_non_existent with custom block_size = " << row_size <<"i" <<column_size << endl;
    vector<double> time;
    double median;

    for (int i = 0; i < num_trials_custom; i++) {
        auto start = high_resolution_clock::now();
        vertex_similarity_block_processing(my_graph, row_size, column_size);
        auto stop = high_resolution_clock::now();

        time.push_back(
            std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count());
        cout << i << " iter: " << time.back() << endl;
    }
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
                            auto result_vertex_similarity = preview::vertex_similarity(
                                jaccard_desc,
                                g,
                                processing_blocks[tbb::this_task_arena::current_thread_index()]);

                            //print_vertex_similarity_result(result_vertex_similarity);

                            // do application specific postprocessing of the result here
                        }
                    },
                    tbb::simple_partitioner{});
            }
        },
        tbb::simple_partitioner{});
}

/*
using namespace oneapi::dal;
using namespace oneapi::dal::preview;
using namespace std;
using namespace std::chrono;

template <class Graph>
void vertex_similarity_block_processing(const Graph &g,
                                        const int32_t block_row_count,
                                        const int32_t block_column_count);

int main(int argc, char **argv) {
 // read the graph
    if (argc < 2) return 0;
    std::string filename = argv[1];
    graph_csv_data_source ds(filename);
  load_graph::descriptor<> d;
  auto my_graph = load_graph::load(d, ds);



    int num_trials_custom = 1;
    int num_trials_lib = 1;

    int32_t row_size = 1;
    int32_t column_size = 1024;

    int32_t tbb_threads_number = 1;

    if (argc > 5) {
        row_size = stoi(argv[2]);
        column_size = stoi(argv[3]);
        num_trials_custom = stoi(argv[4]);
        tbb_threads_number = stoi(argv[5]);
    }
      tbb::global_control c(tbb::global_control::max_allowed_parallelism, tbb_threads_number);
    //cout << "jaccard_non_existent with custom block_size = " << row_size <<"i" <<column_size << endl;
    vector<double> time;
    double median;

    for(int i = 0; i < num_trials_custom; i++) {
        auto start = high_resolution_clock::now();
        vertex_similarity_block_processing(my_graph, row_size, column_size);
        auto stop = high_resolution_clock::now();

        time.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count());
        cout << i << " iter: " << time.back() << endl;
    }
  return 0;
}

void print_revert_vertex_similarity_result(const oneapi::dal::table &table1, const oneapi::dal::table &table2, const int64_t& nnz_count) {
  auto arr1 = oneapi::dal::row_accessor<const float>(table1).pull();
  const auto x1 = arr1.get_data();
  auto arr2 = oneapi::dal::row_accessor<const float>(table2).pull();
  const auto x2 = arr2.get_data();

  for (std::int64_t i = 0; i < nnz_count; i++) {
    if (x1[i] < x1[table1.get_column_count() + i]) {
      cout << x1[i] << " " << x1[table1.get_column_count() + i] << " " << x2[i] << endl;
    }
  }
}

template <class Graph>
void vertex_similarity_block_processing(const Graph &g,
                                        const int32_t block_row_count,
                                        const int32_t block_column_count) {
  // create caching builders for all threads
  std::vector<jaccard::caching_builder> processing_blocks(
      tbb::this_task_arena::max_concurrency());

  // compute the number of vertices in graph
  int32_t vertex_count = get_vertex_count(g);

  // compute the number of rows
  int32_t row_count = vertex_count / block_row_count;
  if (vertex_count % block_row_count) {
    row_count++;
  }

  // parallel processing by rows
  tbb::parallel_for(
      tbb::blocked_range<int32_t>(0, row_count),
      [&](const tbb::blocked_range<int32_t> &r) {
        for (int32_t i = r.begin(); i != r.end(); ++i) {
          // compute the range of rows
          int32_t row_range_begin = i * block_row_count;
          int32_t row_range_end   = (i + 1) * block_row_count;

          // start column ranges from diagonal
          int32_t column_begin = 1 + row_range_begin;

          // compute the number of columns
          int32_t column_count = (vertex_count - column_begin) / block_column_count;
          if ((vertex_count - column_begin) % block_column_count) {
            column_count++;
          }

          // parallel processing by columns
          tbb::parallel_for(
              tbb::blocked_range<int32_t>(0, column_count),
              [&](const tbb::blocked_range<int32_t> &inner_r) {
                for (int32_t j = inner_r.begin(); j != inner_r.end(); ++j) {
                  // compute the range of columns
                  int32_t column_range_begin = column_begin + j * block_column_count;
                  int32_t column_range_end   = column_begin + (j + 1) * block_column_count;

                  // set block ranges for the vertex similarity algorithm
                  const auto jaccard_desc =
                      jaccard::descriptor<>().set_block(
                          {row_range_begin,
                           std::min(row_range_end, vertex_count)},
                          {column_range_begin,
                           std::min(column_range_end, vertex_count)});

                  // compute Jaccard coefficients for the block
                  auto result_vertex_similarity = vertex_similarity(jaccard_desc, g,
                    processing_blocks[tbb::this_task_arena::current_thread_index()]);
                                    // extract the result
                  auto jaccard_coeffs = result_vertex_similarity.get_coeffs();
                  auto vertex_pairs = result_vertex_similarity.get_vertex_pairs();
                  auto nonzero_coeff_count = result_vertex_similarity.get_nonzero_coeff_count();
                  print_revert_vertex_similarity_result(vertex_pairs, jaccard_coeffs, nonzero_coeff_count);
                  // do application specific postprocessing of the result here

                  // do application specific postprocessing of the result here
                }
              },
              tbb::simple_partitioner{});
        }
      },
      tbb::simple_partitioner{});
}

*/
