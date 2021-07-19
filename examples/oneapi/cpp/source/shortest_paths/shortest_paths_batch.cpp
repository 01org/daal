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

#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/shortest_paths.hpp"
#include "oneapi/dal/graph/directed_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"



namespace dal = oneapi::dal;
using namespace dal::preview::shortest_paths;
using namespace std;
using namespace std::chrono;


/* gap vertices
4795720 Trial Time:          0.89146
21003853 Trial Time:          0.81985
417968 Trial Time:          0.81908
6496511 Trial Time:          0.86343
6648699 Trial Time:          0.74315
9811073 Trial Time:          0.90958
22247478 Trial Time:          0.79933
5720252 Trial Time:          0.84234
12366459 
20413729 
4217374 
2674749 
22085557 
19445040 
2360788 
19115968 

*/

void print_stats(vector<double>& trials) {
    double median, max, min, mean, std;
    sort(trials.begin(), trials.end());
    if (trials.size() % 2 == 0) median = trials[trials.size()/2]; else median = trials[trials.size()/2 + 1];
    mean = std::accumulate(trials.begin(), trials.end(), 0.0) / trials.size();
    std = std::sqrt(std::inner_product(trials.begin(), trials.end(), trials.begin(), 0.0,
                [](const double& x, const double& y) { return x + y; },
                [mean](const double& x, const double& y) { return (x - mean)*(y - mean); }) / (double)trials.size());
    cout << "Median: "<< median << endl;
    cout << "Max: " << trials.back() << endl;
    cout << "Min: " << trials[0] << endl;
    cout << "Mean: " << mean << endl;
    cout << "Std: " << std << endl;
}

int main(int argc, char** argv) {
    const auto filename = get_data_path(argv[1]);


    // read the graph
    const dal::preview::graph_csv_data_source ds(filename);

    using vertex_type = int32_t;
    using weight_type = double;
    using my_graph_type = dal::preview::directed_adjacency_vector_graph<vertex_type, weight_type>;

    const dal::preview::load_graph::
        descriptor<dal::preview::weighted_edge_list<vertex_type, weight_type>, my_graph_type>
            d;
    const auto my_graph = dal::preview::load_graph::load(d, ds);

    std::allocator<char> alloc;
    double delta = 50000;

    const int num_trials = 64;
    
    vector<double> trials(num_trials, 0);
    std::vector<int64_t> sources = {4795720, 21003853, 417968, 6496511, 6648699, 9811073, 22247478, 5720252, 12366459, 20413729, 4217374,
2674749, 22085557, 19445040, 2360788, 19115968, 7758767, 13468234, 30367, 18599547, 7526108,
16836280, 12742067, 7697995, 5876443, 9616340, 2497673, 10052290, 12493057, 1670855, 2760679,
2460941, 8489650, 5005225, 8744645, 8512023, 21912165, 1105390, 15432163, 1600177, 19079469,
16516637, 20202566, 21372803, 2898009, 8491277, 18798317, 23757560, 17161819, 23180739, 10997085,
3730630, 1079068, 15426822, 12190925, 1155218, 10693488, 14434835, 19963339, 3486185, 18383269,
20269908, 12370764, 7843140, 18652322, 14392639, 19466898, 4391776, 11614381, 4666931, 1574528,
15876126, 21175857, 21777673, 7573082, 5272501, 17602056, 12859233, 2129573, 18557938, 4306463,
731592, 12833041, 20608043, 11310067, 831167, 2011146, 13163387, 18954495, 13703899, 5597676,
19903307, 18049236, 7688344, 7476776, 4045808, 2121979, 6484515, 7660060, 3193977};
int k = 0;
    cout << argv[1] << "\n\n";
    cout <<  "Only Distances \n";
    for(auto& trial: trials) {
        auto start = high_resolution_clock::now();
            // set algorithm parameters
            const auto shortest_paths_desc =
                descriptor<float, method::delta_stepping, task::one_to_all, std::allocator<char>>(
                    sources[k],
                    delta,
                    optional_results::distances,
                    alloc);
            // compute shortest paths
            //const auto result_shortest_paths = dal::preview::traverse(shortest_paths_desc, my_graph);
        auto stop = high_resolution_clock::now();
        trial = duration_cast<duration<double>>(stop - start).count(); 
        cout << "Source: " << sources[k] <<  ": "<< trial << "\n";   
        k++;
    }
    cout << endl;
    print_stats(trials);
 k = 0;
        cout <<  "\n Preds + Distances \n";
    for(auto& trial: trials) {
        auto start = high_resolution_clock::now();
            // set algorithm parameters
            const auto shortest_paths_desc =
                descriptor<float, method::delta_stepping, task::one_to_all, std::allocator<char>>(
                    sources[k],
                    delta,
                    optional_results::distances | optional_results::predecessors ,
                    alloc);
            // compute shortest paths
            const auto result_shortest_paths = dal::preview::traverse(shortest_paths_desc, my_graph);
        auto stop = high_resolution_clock::now();
        trial = duration_cast<duration<double>>(stop - start).count(); 
        cout << "Source: " << sources[k] <<  ": "<< trial << "\n";   
        k++;
    }
    cout << endl;
    print_stats(trials);
 k = 0;
            cout <<  "\n Preds \n";
    for(auto& trial: trials) {
        auto start = high_resolution_clock::now();
            // set algorithm parameters
            const auto shortest_paths_desc =
                descriptor<float, method::delta_stepping, task::one_to_all, std::allocator<char>>(
                    sources[k],
                    delta,
                    optional_results::predecessors ,
                    alloc);
            // compute shortest paths
            const auto result_shortest_paths = dal::preview::traverse(shortest_paths_desc, my_graph);
        auto stop = high_resolution_clock::now();
        trial = duration_cast<duration<double>>(stop - start).count(); 
        cout << "Source: " << sources[k] <<  ": "<< trial << "\n";   
        k++;
    }
    cout << endl;
    print_stats(trials);


    return 0;
}
