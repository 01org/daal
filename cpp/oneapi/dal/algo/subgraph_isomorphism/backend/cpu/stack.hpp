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

#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph_status.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/inner_alloc.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

class stack {
public:
    stack(inner_alloc allocator);
    stack(std::int64_t max_size, inner_alloc allocator);
    virtual ~stack();
    stack(stack&& _stack);
    stack& operator=(stack&& _stack);
    state* operator[](std::int64_t index);

    graph_status push(state* new_state);
    state* pop();
    std::int64_t size() const;
    void clear(bool direct = true);
    void clear_state(std::int64_t index);
    void add(stack& _stack);

private:
    inner_alloc allocator_;
    std::int64_t max_stack_size;
    std::int64_t stack_size;
    state** data;

    graph_status increase_stack_size();
    void delete_data();
};

class vertex_stack {
public:
    vertex_stack(inner_alloc allocator);
    vertex_stack(const std::uint64_t max_states_size, inner_alloc allocator);
    vertex_stack(const std::uint64_t max_states_size,
                 const std::uint64_t* pdata,
                 inner_alloc allocator);
    virtual ~vertex_stack();

    graph_status push(const std::uint64_t vertex_id);
    std::int64_t pop();
    bool delete_vertex();
    std::uint64_t size() const;
    std::uint64_t max_size() const;

    inner_alloc allocator_;
    std::uint64_t stack_size;
    std::uint64_t* stack_data;
    std::uint64_t* ptop;
    graph_status increase_stack_size();
    bool use_external_memory;
};

class dfs_stack {
public:
    dfs_stack(inner_alloc allocator);
    dfs_stack(const std::uint64_t levels, inner_alloc allocator);
    dfs_stack(const std::uint64_t levels,
              const std::uint64_t max_states_size,
              inner_alloc allocator);
    dfs_stack(const std::uint64_t levels,
              const std::uint64_t* max_states_size_per_level,
              inner_alloc allocator);
    virtual ~dfs_stack();
    void init(const std::uint64_t levels);
    void init(const std::uint64_t levels, const std::uint64_t max_states_size);
    void init(const std::uint64_t levels, const std::uint64_t* max_states_size_per_level);

    void push_into_current_level(const std::uint64_t vertex_id);
    void push_into_next_level(const std::uint64_t vertex_id);
    void update();

    std::uint64_t states_in_stack() const;
    std::uint64_t size(const std::uint64_t level) const;
    std::uint64_t max_level_width(const std::uint64_t level) const;

    std::uint64_t get_current_level() const;
    std::uint64_t get_current_level_index() const;
    std::uint64_t get_current_level_fill_size() const;
    std::uint64_t top(const std::uint64_t level) const;

    state get_current_state() const;
    void fill_solution(std::int64_t* solution_core, const std::uint64_t last_vertex_id) const;

    void delete_current_state();

protected:
    inner_alloc allocator_;
    std::uint64_t max_level_size;
    vertex_stack* data_by_levels;

    std::uint64_t current_level;

    void increase_core_level();
    void decrease_core_level();

private:
    void delete_data();
};

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
