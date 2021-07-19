
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

#pragma once

#include <atomic>

#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

template <typename Vertex, typename EdgeValue, typename BinsVector>
inline void update_bins(const Vertex& v,
                        const EdgeValue& new_dist,
                        const EdgeValue& delta,
                        BinsVector& local_bins) {
    ONEDAL_ASSERT(new_dist > 0);
    ONEDAL_ASSERT(delta > 0);
    ONEDAL_ASSERT(new_dist / delta <= std::numeric_limits<EdgeValue>::max());
    ONEDAL_ASSERT(new_dist / delta <=
                  static_cast<EdgeValue>(std::numeric_limits<std::int64_t>::max()));
    const std::int64_t dest_bin = static_cast<std::int64_t>(new_dist / delta);
    ONEDAL_ASSERT(dest_bin >= 0);
    if (dest_bin >= local_bins.size()) {
        local_bins.resize(dest_bin + 1);
    }
    local_bins[dest_bin].push_back(v);
}

template <typename Topology, typename EdgeValue, typename BinsVector>
inline void relax_edges_seq(const Topology& t,
                            const EdgeValue* vals,
                            typename Topology::vertex_type u,
                            EdgeValue delta,
                            EdgeValue* dist,
                            BinsVector& local_bins) {
    for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
        const auto v = t._cols_ptr[v_];
        const auto v_w = vals[v_];
        EdgeValue old_dist = dist[v];
        const EdgeValue new_dist = dist[u] + v_w;
        if (new_dist < old_dist) {
            dist[v] = new_dist;
            update_bins(v, new_dist, delta, local_bins);
        }
    }
}

template <typename Topology, typename EdgeValue, typename BinsVector>
inline void relax_edges(const Topology& t,
                        const EdgeValue* vals,
                        typename Topology::vertex_type u,
                        EdgeValue delta,
                        std::atomic<EdgeValue>* dist,
                        BinsVector& local_bins) {
    for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
        const auto v = t._cols_ptr[v_];
        const auto v_w = vals[v_];
        EdgeValue old_dist = dist[v].load();
        const EdgeValue new_dist = dist[u].load() + v_w;
        while (new_dist < old_dist) {
            if (dist[v].compare_exchange_strong(old_dist, new_dist)) {
                update_bins(v, new_dist, delta, local_bins);
            }
            old_dist = dist[v].load();
        }
    }
}

template <typename EV, typename VT>
struct dist_pred {
    dist_pred(const EV& dist_, const VT& pred_) : dist(dist_), pred(pred_) {}
    EV dist = 0;
    VT pred = 0;
};

template <class T1, class T2>
bool operator==(const dist_pred<T1, T2>& lhs, const dist_pred<T1, T2>& rhs) {
    return lhs.first == rhs.first && lhs.second == rhs.second;
}

template <class T1, class T2>
bool operator!=(const dist_pred<T1, T2>& lhs, const dist_pred<T1, T2>& rhs) {
    return !(lhs == rhs);
}

template <typename Topology, typename EdgeValue, typename DP, typename BinsVector>
inline void relax_edges_with_pred(const Topology& t,
                                  const EdgeValue* vals,
                                  typename Topology::vertex_type u,
                                  EdgeValue delta,
                                  DP* dp,
                                  BinsVector& local_bins) {
    for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
        const auto v = t._cols_ptr[v_];
        const auto v_w = vals[v_];
        auto old_dp = dp[v];
        const EdgeValue new_dist = dp[u].dist + v_w;
        if (new_dist < old_dp.dist) {
            dp[v] = dist_pred(new_dist, u);
            update_bins(v, new_dist, delta, local_bins);
        }
    }
}

template <typename BinsVector>
inline bool find_next_bin_index_seq(std::int64_t& curr_bin_index, const BinsVector& local_bins) {
    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    bool is_queue_empty = true;
    for (std::int64_t i = curr_bin_index; i < local_bins[0].size(); i++) {
        if (!local_bins[0][i].empty()) {
            if (i < max_bin_count) {
                curr_bin_index = i;
            }
            else {
                curr_bin_index = max_bin_count;
            }
            is_queue_empty = false;
            break;
        }
    }
    return is_queue_empty;
}

template <typename BinsVector>
inline bool find_next_bin_index(std::int64_t& curr_bin_index, const BinsVector& local_bins) {
    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    bool is_queue_empty = true;

    auto total = oneapi::dal::detail::parallel_reduce_int32_int64_t(
        (std::int64_t)local_bins.size(),
        (std::int64_t)max_bin_count,
        [&](std::int64_t begin, std::int64_t end, std::int64_t thread_min_index) -> std::int64_t {
            for (std::int64_t id = begin; id < end; ++id) {
                for (std::int64_t i = curr_bin_index; i < local_bins[id].size(); i++) {
                    if (!local_bins[id][i].empty()) {
                        thread_min_index = std::min(max_bin_count, i);
                        break;
                    }
                }
            }
            return thread_min_index;
        },
        [&](std::int64_t x, std::int64_t y) -> std::int64_t {
            return std::min(x, y);
        });

    if (total < max_bin_count) {
        curr_bin_index = total;
        is_queue_empty = false;
    }

    return is_queue_empty;
}

template <typename SharedBinContainer, typename BinsVector>
inline std::int64_t reduce_to_common_bin_seq(const std::int64_t& curr_bin_index,
                                             BinsVector& local_bins,
                                             SharedBinContainer& shared_bin) {
    std::int64_t curr_shared_bin_tail = 0;

    if (curr_bin_index < local_bins[0].size()) {
        std::int64_t copy_start = curr_shared_bin_tail;
        curr_shared_bin_tail += local_bins[0][curr_bin_index].size();
        copy(local_bins[0][curr_bin_index].begin(),
             local_bins[0][curr_bin_index].end(),
             shared_bin.get_mutable_data() + copy_start);
        local_bins[0][curr_bin_index].resize(0);
    }
    return curr_shared_bin_tail;
}

template <typename SharedBinContainer, typename BinsVector>
inline std::int64_t reduce_to_common_bin(const std::int64_t& curr_bin_index,
                                         BinsVector& local_bins,
                                         SharedBinContainer& shared_bin) {
    std::atomic<std::int64_t> curr_shared_bin_tail = 0;
    dal::detail::threader_for(local_bins.size(), local_bins.size(), [&](std::int64_t i) {
        int thread_id = dal::detail::threader_get_current_thread_index();
        if (curr_bin_index < local_bins[thread_id].size()) {
            std::int64_t copy_start =
                curr_shared_bin_tail.fetch_add(local_bins[thread_id][curr_bin_index].size());
            copy(local_bins[thread_id][curr_bin_index].begin(),
                 local_bins[thread_id][curr_bin_index].end(),
                 shared_bin.get_mutable_data() + copy_start);
            local_bins[thread_id][curr_bin_index].resize(0);
        }
    });
    return curr_shared_bin_tail.load();
}

template <typename Cpu, typename EdgeValue>
struct delta_stepping {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr) {
        using value_type = EdgeValue;
        using vertex_type = std::int32_t;
        using atomic_value_allocator_type = inner_alloc<std::atomic<value_type>>;
        using vertex_allocator_type = inner_alloc<vertex_type>;

        vertex_allocator_type vertex_allocator(alloc_ptr);
        atomic_value_allocator_type atomic_value_allocator(alloc_ptr);

        const auto source = desc.get_source();

        const value_type delta = desc.get_delta();

        const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
        const std::int64_t max_elements_in_bin = 1000;
        const auto vertex_count = t.get_vertex_count();
        const value_type max_dist = std::numeric_limits<value_type>::max();

        std::atomic<value_type>* dist = allocate(atomic_value_allocator, vertex_count);
        dist = new (dist) std::atomic<value_type>[vertex_count]();
        dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
            dist[i] = max_dist;
        });
        dist[source] = 0;

        vector_container<vertex_type, vertex_allocator_type> shared_bin(t.get_edge_count(),
                                                                        vertex_allocator);

        shared_bin[0] = source;
        std::int64_t curr_bin_index = 0;
        std::int64_t curr_shared_bin_tail = 1;
        bool empty_queue = false;
        std::int64_t thread_cnt = dal::detail::threader_get_max_threads();

        using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
        using v1a_t = inner_alloc<v1v_t>;

        using v2v_t = vector_container<v1v_t, v1a_t>;
        using v2a_t = inner_alloc<v2v_t>;
        v2a_t v2a(alloc_ptr);

        using v3v_t = vector_container<v2v_t, v2a_t>;

        v3v_t local_bins(thread_cnt, v2a);

        local_bins[0].reserve(t.get_vertex_degree(source));

        std::int64_t iter = 0;

        while (curr_bin_index != max_bin_count && iter != max_bin_count && !empty_queue) {
            dal::detail::threader_for(
                curr_shared_bin_tail,
                curr_shared_bin_tail,
                [&](std::int64_t i) {
                    vertex_type u = shared_bin[i];
                    if (dist[u].load() >= delta * static_cast<value_type>(curr_bin_index)) {
                        relax_edges(t,
                                    vals,
                                    u,
                                    delta,
                                    dist,
                                    local_bins[dal::detail::threader_get_current_thread_index()]);
                    }
                });

            dal::detail::threader_for(thread_cnt, thread_cnt, [&](std::int64_t i) {
                int thread_id = dal::detail::threader_get_current_thread_index();
                while (curr_bin_index < local_bins[thread_id].size() &&
                       !local_bins[thread_id][curr_bin_index].empty() &&
                       local_bins[thread_id][curr_bin_index].size() < max_elements_in_bin) {
                    vector_container<vertex_type> curr_bin_copy(
                        local_bins[thread_id][curr_bin_index].size());
                    copy(local_bins[thread_id][curr_bin_index].begin(),
                         local_bins[thread_id][curr_bin_index].end(),
                         curr_bin_copy.begin());

                    local_bins[thread_id][curr_bin_index].resize(0);
                    for (std::int64_t j = 0; j < curr_bin_copy.size(); ++j)
                        relax_edges(t, vals, curr_bin_copy[j], delta, dist, local_bins[thread_id]);
                }
            });

            empty_queue = find_next_bin_index(curr_bin_index, local_bins);

            curr_shared_bin_tail = reduce_to_common_bin(curr_bin_index, local_bins, shared_bin);

            iter++;
        }

        auto dist_arr = array<value_type>::empty(vertex_count);
        value_type* dist_ = dist_arr.get_mutable_data();
        dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
            dist_[i] = dist[i].load();
        });

        deallocate(atomic_value_allocator, dist, vertex_count);
        return traverse_result<task::one_to_all>().set_distances(
            dal::detail::homogen_table_builder{}.reset(dist_arr, t.get_vertex_count(), 1).build());
    }
};

/*
template <typename Cpu, typename EdgeValue>
struct delta_stepping {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr) {
        {
            using value_type = EdgeValue;
            using vertex_type = std::int32_t;
            using value_allocator_type = inner_alloc<value_type>;
            using vertex_allocator_type = inner_alloc<vertex_type>;

            vertex_allocator_type vertex_allocator(alloc_ptr);
            value_allocator_type value_allocator(alloc_ptr);

            const auto source = dal::detail::integral_cast<std::int32_t>(desc.get_source());
            const value_type delta = desc.get_delta();

            const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
            const std::int64_t max_elements_in_bin = 1000;
            const auto vertex_count = t.get_vertex_count();
            const value_type max_dist = std::numeric_limits<value_type>::max();
            std::atomic<value_type>* dist = allocate(atomic_value_allocator, vertex_count);
        dist = new (dist) std::atomic<value_type>[vertex_count]();
        dal::detail::threader_for(vertex_count, vertex_count, [&](std::int64_t i) {
            dist[i] = max_dist;
        });
            dist[source].store(0);

            value_type* dist = allocate(value_allocator, vertex_count);
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                dist[i] = max_dist;
            }

            dist[source] = 0;

            vector_container<vertex_type, vertex_allocator_type> shared_bin(t.get_edge_count(),
                                                                            vertex_allocator);

            shared_bin[0] = source;
            std::int64_t curr_bin_index = 0;
            std::int64_t curr_shared_bin_tail = 1;
            bool empty_queue = false;

            using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
            using v1a_t = inner_alloc<v1v_t>;
            using v2v_t = vector_container<v1v_t, v1a_t>;
            using v2a_t = inner_alloc<v2v_t>;
            using v3v_t = vector_container<v2v_t, v2a_t>;
            v2a_t v2a(alloc_ptr);
            v3v_t local_bins(1, v2a);

            std::int64_t iter = 0;

            while (curr_bin_index != max_bin_count && iter != max_bin_count && !empty_queue) {
                for (std::int64_t i = 0; i < curr_shared_bin_tail; ++i) {
                    vertex_type u = shared_bin[i];
                    if (dist[u] >= delta * static_cast<value_type>(curr_bin_index)) {
                        relax_edges(t, vals, u, delta, dist, local_bins[0]);
                    }
                }

                while (curr_bin_index < local_bins[0].size() &&
                       !local_bins[0][curr_bin_index].empty() &&
                       local_bins[0][curr_bin_index].size() < max_elements_in_bin) {
                    vector_container<vertex_type> curr_bin_copy(
                        local_bins[0][curr_bin_index].size());
                    copy(local_bins[0][curr_bin_index].begin(),
                         local_bins[0][curr_bin_index].end(),
                         curr_bin_copy.begin());

                    local_bins[0][curr_bin_index].resize(0);
                    for (std::int64_t j = 0; j < curr_bin_copy.size(); ++j) {
                        relax_edges(t, vals, curr_bin_copy[j], delta, dist, local_bins[0]);
                    }
                }

                empty_queue = find_next_bin_index_seq(curr_bin_index, local_bins);

                curr_shared_bin_tail =
                    reduce_to_common_bin_seq(curr_bin_index, local_bins, shared_bin);

                iter++;
            }

            auto dist_arr = array<value_type>::empty(vertex_count);
            value_type* dist_ = dist_arr.get_mutable_data();
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                dist_[i] = dist[i];
            }

            deallocate(value_allocator, dist, vertex_count);
            return traverse_result<task::one_to_all>().set_distances(
                dal::detail::homogen_table_builder{}
                    .reset(dist_arr, t.get_vertex_count(), 1)
                    .build());
        }
    }
};*/

template <typename Cpu, typename EdgeValue>
struct delta_stepping_with_pred {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr) {
        using value_type = EdgeValue;
        using vertex_type = std::int32_t;
        using vp_type = dist_pred<value_type, vertex_type>;
        using vp_allocator_type = inner_alloc<vp_type>;
        using vertex_allocator_type = inner_alloc<vertex_type>;

        vertex_allocator_type vertex_allocator(alloc_ptr);
        vp_allocator_type vp_allocator(alloc_ptr);

        const auto source = dal::detail::integral_cast<std::int32_t>(desc.get_source());

        const value_type delta = desc.get_delta();

        const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
        const std::int64_t max_elements_in_bin = 1000;
        const auto vertex_count = t.get_vertex_count();
        const value_type max_dist = std::numeric_limits<value_type>::max();

        vp_type* dp = allocate(vp_allocator, vertex_count);
        for (std::int64_t i = 0; i < vertex_count; ++i) {
            new (dp + i) dist_pred<value_type, vertex_type>(max_dist, -1);
        }

        dp[source] = dist_pred<value_type, vertex_type>(0, -1);

        vector_container<vertex_type, vertex_allocator_type> shared_bin(t.get_edge_count(),
                                                                        vertex_allocator);

        shared_bin[0] = source;
        std::int64_t curr_bin_index = 0;
        std::int64_t curr_shared_bin_tail = 1;
        bool empty_queue = false;

        using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
        using v1a_t = inner_alloc<v1v_t>;

        using v2v_t = vector_container<v1v_t, v1a_t>;
        using v2a_t = inner_alloc<v2v_t>;
        using v3v_t = vector_container<v2v_t, v2a_t>;

        v2a_t v2a(alloc_ptr);
        v3v_t local_bins(1, v2a);

        std::int64_t iter = 0;

        while (curr_bin_index != max_bin_count && iter != max_bin_count && !empty_queue) {
            for (std::int64_t i = 0; i < curr_shared_bin_tail; ++i) {
                vertex_type u = shared_bin[i];
                if (dp[u].dist >= delta * static_cast<value_type>(curr_bin_index)) {
                    relax_edges_with_pred(t, vals, u, delta, dp, local_bins[0]);
                }
            }

            while (curr_bin_index < local_bins[0].size() &&
                   !local_bins[0][curr_bin_index].empty() &&
                   local_bins[0][curr_bin_index].size() < max_elements_in_bin) {
                vector_container<vertex_type> curr_bin_copy(local_bins[0][curr_bin_index].size());
                copy(local_bins[0][curr_bin_index].begin(),
                     local_bins[0][curr_bin_index].end(),
                     curr_bin_copy.begin());

                local_bins[0][curr_bin_index].resize(0);
                for (std::int64_t j = 0; j < curr_bin_copy.size(); ++j) {
                    relax_edges_with_pred(t, vals, curr_bin_copy[j], delta, dp, local_bins[0]);
                }
            }

            empty_queue = find_next_bin_index_seq(curr_bin_index, local_bins);

            curr_shared_bin_tail = reduce_to_common_bin_seq(curr_bin_index, local_bins, shared_bin);

            iter++;
        }

        if (desc.get_optional_results() & optional_results::distances) {
            auto dist_arr = array<value_type>::empty(vertex_count);
            auto pred_arr = array<vertex_type>::empty(vertex_count);
            value_type* dist_ = dist_arr.get_mutable_data();
            vertex_type* pred_ = pred_arr.get_mutable_data();
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                const auto dp_i = dp[i];
                dist_[i] = dp_i.dist;
                pred_[i] = dp_i.pred;
            }

            deallocate(vp_allocator, dp, vertex_count);
            return traverse_result<task::one_to_all>()
                .set_distances(dal::detail::homogen_table_builder{}
                                   .reset(dist_arr, t.get_vertex_count(), 1)
                                   .build())
                .set_predecessors(dal::detail::homogen_table_builder{}
                                      .reset(pred_arr, t.get_vertex_count(), 1)
                                      .build());
        }
        else {
            auto pred_arr = array<vertex_type>::empty(vertex_count);
            vertex_type* pred_ = pred_arr.get_mutable_data();
            for (std::int64_t i = 0; i < vertex_count; ++i) {
                const auto dp_i = dp[i];
                pred_[i] = dp_i.pred;
            }

            deallocate(vp_allocator, dp, vertex_count);
            return traverse_result<task::one_to_all>().set_predecessors(
                dal::detail::homogen_table_builder{}
                    .reset(pred_arr, t.get_vertex_count(), 1)
                    .build());
        }
    }
};

} // namespace oneapi::dal::preview::shortest_paths::backend
