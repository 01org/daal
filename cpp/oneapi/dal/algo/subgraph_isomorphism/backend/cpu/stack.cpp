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

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/stack.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

namespace dal = oneapi::dal;

stack::stack(inner_alloc allocator) : allocator_(allocator) {
    max_stack_size = 100;
    stack_size = 0;
    data = allocator_.allocate<state*>(max_stack_size);
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        data[i] = nullptr;
    }
}

stack::stack(std::int64_t max_size, inner_alloc allocator) : allocator_(allocator) {
    max_stack_size = max_size;
    stack_size = 0;
    data = allocator_.allocate<state*>(max_stack_size);
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        data[i] = nullptr;
    }
}

void stack::delete_data() {
    if (data != nullptr) {
        for (std::int64_t i = 0; i < max_stack_size; i++) {
            if (data[i] != nullptr) {
                data[i]->~state();
                allocator_.deallocate<state>(data[i], 0);
                data[i] = nullptr;
            }
        }
        allocator_.deallocate<state*>(data, max_stack_size);
        data = nullptr;
    }
}

void stack::clear(bool direct) {
    for (std::int64_t i = 0; i < stack_size * direct; i++) {
        if (data[i] != nullptr) {
            data[i]->~state();
            allocator_.deallocate<state>(data[i], 0);
            data[i] = nullptr;
        }
    }
    stack_size = 0;
}

void stack::clear_state(std::int64_t index) {
    data[index]->~state();
    allocator_.deallocate<state>(data[index], 0);
    data[index] = nullptr;
}

void stack::add(stack& _stack) {
    std::int64_t current_size = _stack.size();
    for (std::int64_t i = 0; i < current_size; i++) {
        push(_stack.pop());
    }
    _stack.clear();
}

stack::stack(stack&& _stack) : allocator_(_stack.allocator_), data(_stack.data) {
    max_stack_size = _stack.max_stack_size;
    stack_size = _stack.stack_size;
    _stack.data = nullptr;
    _stack.max_stack_size = 0;
    _stack.stack_size = 0;
}

stack& stack::operator=(stack&& _stack) {
    if (&_stack == this) {
        return *this;
    }
    delete_data();
    max_stack_size = _stack.max_stack_size;
    stack_size = _stack.stack_size;
    data = _stack.data;
    _stack.data = nullptr;
    _stack.max_stack_size = 0;
    _stack.stack_size = 0;
    return *this;
}

stack::~stack() {
    if (data != nullptr) {
        for (std::int64_t i = 0; i < max_stack_size; i++) {
            if (data[i] != nullptr) {
                data[i]->~state();
                allocator_.deallocate<state>(data[i], 0);
                data[i] = nullptr;
            }
        }
        allocator_.deallocate<state*>(data, max_stack_size);
        data = nullptr;
    }
    stack_size = 0;
}

graph_status stack::push(state* new_state) {
    if (new_state != nullptr) {
        if (max_stack_size == 0 || stack_size >= max_stack_size) {
            graph_status increase_status = increase_stack_size();
            if (increase_status != ok) {
                return increase_status;
            }
        }
        data[stack_size] = new_state;
        stack_size++;
    }
    return ok;
}

state* stack::pop() {
    state* pstate = nullptr;
    if (stack_size > 0) {
        stack_size--;
        pstate = data[stack_size];
        data[stack_size] = nullptr;
    }
    return pstate;
}

state* stack::operator[](std::int64_t index) {
    return data[index];
}

std::int64_t stack::size() const {
    return stack_size;
}

graph_status stack::increase_stack_size() {
    const auto new_max_stack_size = (max_stack_size > 0) ? 2 * max_stack_size : 100;
    state** tmp_data = allocator_.allocate<state*>(new_max_stack_size);
    if (tmp_data == nullptr) {
        throw oneapi::dal::host_bad_alloc();
    }
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        tmp_data[i] = data[i];
        data[i] = nullptr;
    }
    for (std::int64_t i = max_stack_size; i < new_max_stack_size; i++) {
        tmp_data[i] = nullptr;
    }
    allocator_.deallocate<state*>(data, max_stack_size);
    max_stack_size = new_max_stack_size;
    data = tmp_data;
    tmp_data = nullptr;
    return ok;
}

vertex_stack::vertex_stack(inner_alloc allocator) : allocator_(allocator), bottom_(nullptr) {
    stack_size = 0;
    stack_data = nullptr;
    ptop = nullptr;
    use_external_memory = false;
}

vertex_stack::vertex_stack(const std::uint64_t max_states_size, inner_alloc allocator)
        : allocator_(allocator) {
    use_external_memory = false;
    stack_size = max_states_size;
    stack_data = allocator_.allocate<std::uint64_t>(stack_size);
    ptop = stack_data;
    bottom_ = stack_data;
}

vertex_stack::vertex_stack(const std::uint64_t max_states_size,
                           const std::uint64_t* pdata,
                           inner_alloc allocator)
        : allocator_(allocator) {
    use_external_memory = true;
    stack_size = max_states_size;
    //stack_data = pdata;
    //ptop = stack_data;
}

vertex_stack::~vertex_stack() {
    allocator_.deallocate<std::uint64_t>(stack_data, stack_size);
    stack_data = nullptr;
    ptop = nullptr;
    stack_size = 0;
}

graph_status vertex_stack::push(const std::uint64_t vertex_id) {
    ONEDAL_ASSERT(ptop != nullptr);
    ONEDAL_ASSERT(stack_data != nullptr);
    if (ptop - stack_data >= stack_size) {
        if (increase_stack_size() != ok) {
            throw dal::host_bad_alloc();
        }
    }
    ONEDAL_ASSERT(ptop != nullptr);
    ONEDAL_ASSERT(ptop >= bottom_);
    ONEDAL_ASSERT(ptop <= stack_data + stack_size);
    *ptop = vertex_id;
    ptop++;
    return ok;
}

std::int64_t vertex_stack::pop() {
    if (ptop != nullptr && ptop != bottom_) {
        ONEDAL_ASSERT(ptop >= bottom_);
        ONEDAL_ASSERT(ptop <= stack_data + stack_size);

        ptop--;
        return *ptop;
    }
    return -1;
}

bool vertex_stack::delete_vertex() {
    ONEDAL_ASSERT(ptop != nullptr);
    ONEDAL_ASSERT(ptop >= bottom_);
    ONEDAL_ASSERT(ptop <= stack_data + stack_size);

    ptop -= (ptop != nullptr) && (ptop != bottom_);
    return !(ptop - bottom_);
}

std::uint64_t vertex_stack::size() const {
    return ptop - bottom_;
}

std::uint64_t vertex_stack::max_size() const {
    return stack_size;
}

graph_status vertex_stack::increase_stack_size() {
    std::uint64_t* tmp_data = allocator_.allocate<std::uint64_t>(2 * stack_size);
    if (tmp_data == nullptr) {
        return bad_allocation;
    }
    const auto skip_count = bottom_ - stack_data;
    for (std::uint64_t i = 0; i < stack_size - skip_count; i++) {
        tmp_data[i] = stack_data[i + skip_count];
        //tmp_data[i + stack_size] = null_node;
    }
    allocator_.deallocate<std::uint64_t>(stack_data, stack_size);
    stack_size *= 2;
    ptop = size() + tmp_data;
    bottom_ = tmp_data;
    stack_data = tmp_data;
    tmp_data = nullptr;

    ONEDAL_ASSERT(ptop != nullptr);
    ONEDAL_ASSERT(ptop >= bottom_);
    ONEDAL_ASSERT(ptop <= stack_data + stack_size);

    return ok;
}

bool global_stack::push(dfs_stack& s) {
    for (auto level = s.get_current_level_index(); level > 0; --level) {
        if (s.data_by_levels[level].size() > 1) {
            internal_push(s, level);
            return true;
        }
    }

    if (s.data_by_levels[0].size() > 1) {
        internal_push(s, 0);
        return true;
    }

    return false;
}

void global_stack::pop(dfs_stack& s) {
    ONEDAL_ASSERT(s.empty());
    lock_type lock(mutex_);
    if (!data_.empty()) {
        const auto& v = data_.top();
        ONEDAL_ASSERT(v.size() <= s.max_level_size);
        for (std::uint64_t i = 0; i < v.size(); ++i) {
            s.push_into_current_level(v[i]);
            if (i != v.size() - 1) {
                s.increase_core_level();
            }
        }
        data_.pop();
    }
}

void global_stack::internal_push(dfs_stack& s, std::uint64_t level) {
    // Temporary
    static std::mutex m;

    // Collect state and push back
    {
        decltype(data_)::value_type v(level + 1);

        for (std::uint64_t i = 0; i < level; ++i) {
            ONEDAL_ASSERT(i < s.max_level_size);
            ONEDAL_ASSERT(s.data_by_levels[i].ptop != nullptr);
            ONEDAL_ASSERT(s.data_by_levels[i].ptop != s.data_by_levels[i].bottom_);
            ONEDAL_ASSERT(s.data_by_levels[i].ptop >= s.data_by_levels[i].bottom_);
            ONEDAL_ASSERT(s.data_by_levels[i].ptop <=
                          s.data_by_levels[i].stack_data + s.data_by_levels[i].stack_size);
            v[i] = s.data_by_levels[i].ptop[-1];
        }

        // Temporary
        if (!(s.data_by_levels[level].ptop <=
              s.data_by_levels[level].stack_data + s.data_by_levels[level].stack_size)) {
            const std::scoped_lock<std::mutex> lock(m);
            std::cout << "[global_stack::internal_push] this = " << (const void*)this << std::endl;
            std::cout << "[global_stack::internal_push] level = " << level << std::endl;
            std::cout << "[global_stack::internal_push] s.max_level_size = " << s.max_level_size
                      << std::endl;
            std::cout << "[global_stack::internal_push] s.data_by_levels[level].ptop = "
                      << (const void*)s.data_by_levels[level].ptop << std::endl;
            std::cout << "[global_stack::internal_push] s.data_by_levels[level].bottom_ = "
                      << (const void*)s.data_by_levels[level].bottom_ << std::endl;
            std::cout << "[global_stack::internal_push] s.data_by_levels[level].stack_data = "
                      << (const void*)s.data_by_levels[level].stack_data << std::endl;

            std::cout << "[global_stack::internal_push] s.data_by_levels[level].stack_size = "
                      << s.data_by_levels[level].stack_size << std::endl;
        }
        ONEDAL_ASSERT(level < s.max_level_size);
        ONEDAL_ASSERT(s.data_by_levels[level].ptop != nullptr);
        ONEDAL_ASSERT(s.data_by_levels[level].bottom_ != nullptr);
        ONEDAL_ASSERT(s.data_by_levels[level].ptop != s.data_by_levels[level].bottom_);
        ONEDAL_ASSERT(s.data_by_levels[level].ptop >= s.data_by_levels[level].bottom_);
        ONEDAL_ASSERT(s.data_by_levels[level].ptop <=
                      s.data_by_levels[level].stack_data + s.data_by_levels[level].stack_size);
        v[level] = *(s.data_by_levels[level].bottom_);

        lock_type lock(mutex_);
        data_.push(v);
    }

    // Remove state
    ++(s.data_by_levels[level].bottom_);
}

dfs_stack::dfs_stack(inner_alloc allocator) : allocator_(allocator) {
    max_level_size = 0;
    data_by_levels = nullptr;

    current_level = 0;
}

dfs_stack::dfs_stack(const std::uint64_t levels, inner_alloc allocator) : allocator_(allocator) {
    init(levels);
}

dfs_stack::dfs_stack(const std::uint64_t levels,
                     const std::uint64_t max_states_size,
                     inner_alloc allocator)
        : allocator_(allocator) {
    init(levels, max_states_size);
}

dfs_stack::dfs_stack(const std::uint64_t levels,
                     const std::uint64_t* max_states_size_per_level,
                     inner_alloc allocator)
        : allocator_(allocator) {
    init(levels, max_states_size_per_level);
}

void dfs_stack::init(const std::uint64_t levels) {
    max_level_size = levels;
    current_level = 0;

    data_by_levels =
        static_cast<vertex_stack*>(operator new[](sizeof(vertex_stack) * max_level_size));
}

void dfs_stack::init(const std::uint64_t levels, const std::uint64_t max_states_size) {
    init(levels);

    for (std::uint64_t i = 0; i < max_level_size; ++i) {
        new (data_by_levels + i) vertex_stack(max_states_size, allocator_);
    }
}

void dfs_stack::init(const std::uint64_t levels, const std::uint64_t* max_states_size_per_level) {
    init(levels);

    for (std::uint64_t i = 0; i < max_level_size; ++i) {
        new (data_by_levels + i) vertex_stack(max_states_size_per_level[i], allocator_);
    }
}

void dfs_stack::delete_data() {
    for (std::uint64_t i = 0; i < max_level_size; i++) {
        data_by_levels[i].~vertex_stack();
    }
    operator delete[](data_by_levels);
    data_by_levels = nullptr;

    max_level_size = 0;
    current_level = 0;
}

dfs_stack::~dfs_stack() {
    delete_data();
}

std::uint64_t dfs_stack::get_current_level() const {
    return current_level + 1;
}

std::uint64_t dfs_stack::get_current_level_fill_size() const {
    return data_by_levels[current_level].size();
}

void dfs_stack::push_into_current_level(const std::uint64_t vertex_id) {
    data_by_levels[current_level].push(vertex_id);
}

std::uint64_t dfs_stack::get_current_level_index() const {
    return current_level;
}

void dfs_stack::push_into_next_level(const std::uint64_t vertex_id) {
    //current_level++;
    data_by_levels[current_level + 1].push(vertex_id);
}

void dfs_stack::increase_core_level() {
    current_level++;
}

void dfs_stack::decrease_core_level() {
    current_level--;
}

void dfs_stack::update() {
    std::uint64_t new_level = current_level + 1;
    if (new_level < max_level_size && size(new_level) > 0) {
        current_level++;
    }
    else {
        delete_current_state();
    }
}

state dfs_stack::get_current_state() const {
    state result(current_level + 1, allocator_);
    for (std::int64_t i = 0; i < result.core_length; i++) {
        result.core[i] = *(data_by_levels[i].ptop - 1);
    }
    return result;
}

void dfs_stack::fill_solution(std::int64_t* solution_core,
                              const std::uint64_t last_vertex_id) const {
    for (std::uint64_t i = 0; i <= current_level; i++) {
        solution_core[i] = *(data_by_levels[i].ptop - 1);
    }
    solution_core[current_level + 1] = last_vertex_id;
}

std::uint64_t dfs_stack::top(const std::uint64_t level) const {
    //if (data_by_levels[level].size()) {
    return *(data_by_levels[level].ptop - 1);
    //}
    //return null_node;
}

void dfs_stack::delete_current_state() {
    while (data_by_levels[current_level].delete_vertex() && current_level > 0) {
        current_level--;
    }
}

std::uint64_t dfs_stack::states_in_stack() const {
    std::int64_t size = 0;
    for (std::uint64_t i = 0; i <= current_level; i++) {
        size += data_by_levels[i].size();
    }
    return size - current_level;
}

std::uint64_t dfs_stack::size(const std::uint64_t level) const {
    return data_by_levels[level].size();
}

std::uint64_t dfs_stack::max_level_width(const std::uint64_t level) const {
    return data_by_levels[level].max_size();
}

bool dfs_stack::empty() const {
    return (current_level == 0) && ((max_level_size == 0) || (data_by_levels[0].size() == 0));
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
