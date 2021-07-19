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

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/decision_forest/train_types.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_misc_structs.hpp"

#include <CL/sycl/ONEAPI/experimental/builtins.hpp>

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace de = dal::detail;
namespace be = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = cl::sycl::usm::alloc;

template <typename Float, typename Index, typename Task>
struct impurity_data;

template <typename Float, typename Index>
struct impurity_data<Float, Index, task::classification> {
    using task_t = task::classification;
    using context_t = train_context<Float, Index, task_t>;
    using impl_const_t = impl_const<Index, task_t>;
    using imp_data_t = impurity_data<Float, Index, task_t>;

    impurity_data() = default;
    ~impurity_data() = default;
    impurity_data(const cl::sycl::queue& q, Index node_count, Index class_count)
            : imp_list_(
                  pr::ndarray<Float, 1>::empty(q,
                                               { node_count * impl_const_t::node_imp_prop_count_ },
                                               alloc::device)),
              class_hist_list_(
                  pr::ndarray<Index, 1>::empty(q, { node_count * class_count }, alloc::device)) {}

    impurity_data(const cl::sycl::queue& q, const context_t& ctx, Index node_count)
            : imp_list_(
                  pr::ndarray<Float, 1>::empty(q,
                                               { node_count * impl_const_t::node_imp_prop_count_ },
                                               alloc::device)),
              class_hist_list_(pr::ndarray<Index, 1>::empty(q,
                                                            { node_count * ctx.class_count_ },
                                                            alloc::device)) {}

    imp_data_t to_host(cl::sycl::queue& q, const dal::backend::event_vector& deps = {}) const {
        imp_data_t imp_data_host;
        imp_data_host.imp_list_ = imp_list_.to_host(q, deps);
        imp_data_host.class_hist_list_ = class_hist_list_.to_host(q);
        return imp_data_host;
    }

    dal::backend::primitives::ndarray<Float, 1> imp_list_;
    dal::backend::primitives::ndarray<Index, 1> class_hist_list_;
};

template <typename Float, typename Index>
struct impurity_data<Float, Index, task::regression> {
    using task_t = task::regression;
    using context_t = train_context<Float, Index, task_t>;
    using impl_const_t = impl_const<Index, task::regression>;
    using imp_data_t = impurity_data<Float, Index, task_t>;

    impurity_data() = default;
    ~impurity_data() = default;
    impurity_data(const cl::sycl::queue& q, Index node_count)
            : imp_list_(
                  pr::ndarray<Float, 1>::empty(q,
                                               { node_count * impl_const_t::node_imp_prop_count_ },
                                               alloc::device)) {}

    impurity_data(const cl::sycl::queue& q, const context_t& ctx, Index node_count)
            : imp_list_(
                  pr::ndarray<Float, 1>::empty(q,
                                               { node_count * impl_const_t::node_imp_prop_count_ },
                                               alloc::device)) {}

    imp_data_t to_host(cl::sycl::queue& q, const dal::backend::event_vector& deps = {}) const {
        imp_data_t imp_data_host;
        imp_data_host.imp_list_ = imp_list_.to_host(q, deps);
        return imp_data_host;
    }

    dal::backend::primitives::ndarray<Float, 1> imp_list_;
};

// holders for impurity data pointers parametrized by task
template <typename Float, typename Index, typename Task = task::by_default>
struct imp_data_list_ptr;

template <typename Float, typename Index>
struct imp_data_list_ptr<Float, Index, task::classification> {
    imp_data_list_ptr(const impurity_data<Float, Index, task::classification>& imp_data)
            : imp_list_ptr_(imp_data.imp_list_.get_data()),
              class_hist_list_ptr_(imp_data.class_hist_list_.get_data()) {}
    const Float* imp_list_ptr_;
    const Index* class_hist_list_ptr_;

    const Index* get_class_hist_list_ptr_or_null() {
        return class_hist_list_ptr_;
    }
};

template <typename Float, typename Index>
struct imp_data_list_ptr<Float, Index, task::regression> {
    imp_data_list_ptr(const impurity_data<Float, Index, task::regression>& imp_data)
            : imp_list_ptr_(imp_data.imp_list_.get_data()) {}
    const Float* imp_list_ptr_;

    const Index* get_class_hist_list_ptr_or_null() {
        return nullptr;
    }
};

template <typename Float, typename Index, typename Task = task::by_default>
struct imp_data_list_ptr_mutable;

template <typename Float, typename Index>
struct imp_data_list_ptr_mutable<Float, Index, task::classification> {
    imp_data_list_ptr_mutable(impurity_data<Float, Index, task::classification>& imp_data)
            : imp_list_ptr_(imp_data.imp_list_.get_mutable_data()),
              class_hist_list_ptr_(imp_data.class_hist_list_.get_mutable_data()) {}

    Float* imp_list_ptr_;
    Index* class_hist_list_ptr_;

    Index* get_class_hist_list_ptr_or_null() {
        return class_hist_list_ptr_;
    }
};

template <typename Float, typename Index>
struct imp_data_list_ptr_mutable<Float, Index, task::regression> {
    imp_data_list_ptr_mutable(impurity_data<Float, Index, task::regression>& imp_data)
            : imp_list_ptr_(imp_data.imp_list_.get_mutable_data()) {}

    Float* imp_list_ptr_;

    Index* get_class_hist_list_ptr_or_null() {
        return nullptr;
    }
};

template <typename Float, typename Index, typename Task>
struct impurity_data_manager;

template <typename Float, typename Index>
struct impurity_data_manager<Float, Index, task::classification> {
    using task_t = task::classification;
    using imp_data_t = impurity_data<Float, Index, task_t>;
    using context_t = train_context<Float, Index, task_t>;

    impurity_data_manager(const cl::sycl::queue& q, const context_t& ctx)
            : queue_(q),
              class_count_(ctx.class_count_) {}
    ~impurity_data_manager() = default;

    void init_new_level(Index node_count) {
        ONEDAL_ASSERT(node_count);
        level_node_imp_list_.push_back(imp_data_t{ queue_, node_count, class_count_ });
    }

    const imp_data_t& get_data(Index level) {
        ONEDAL_ASSERT(static_cast<std::uint64_t>(level) < level_node_imp_list_.size());
        return level_node_imp_list_[level];
    }

    imp_data_t& get_mutable_data(Index level) {
        ONEDAL_ASSERT(static_cast<std::uint64_t>(level) < level_node_imp_list_.size());
        return level_node_imp_list_[level];
    }

    const cl::sycl::queue& queue_;
    Index class_count_ = 0;
    std::vector<imp_data_t> level_node_imp_list_;
};

template <typename Float, typename Index>
struct impurity_data_manager<Float, Index, task::regression> {
    using task_t = task::regression;
    using imp_data_t = impurity_data<Float, Index, task_t>;
    using context_t = train_context<Float, Index, task_t>;

    impurity_data_manager(const cl::sycl::queue& q, const context_t& ctx) : queue_(q) {}
    ~impurity_data_manager() = default;

    void init_new_level(Index node_count) {
        ONEDAL_ASSERT(node_count);
        level_node_imp_list_.push_back(imp_data_t{ queue_, node_count });
    }

    const imp_data_t& get_data(Index level) {
        ONEDAL_ASSERT(static_cast<std::uint64_t>(level) < level_node_imp_list_.size());
        return level_node_imp_list_[level];
    }

    imp_data_t& get_mutable_data(Index level) {
        ONEDAL_ASSERT(static_cast<std::uint64_t>(level) < level_node_imp_list_.size());
        return level_node_imp_list_[level];
    }

    const cl::sycl::queue& queue_;
    std::vector<imp_data_t> level_node_imp_list_;
};

} // namespace oneapi::dal::decision_forest::backend

#endif
