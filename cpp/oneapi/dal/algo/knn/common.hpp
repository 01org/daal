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

#include "oneapi/dal/algo/knn/detail/distance.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/common.hpp"

namespace oneapi::dal::knn {

namespace v1 {
/// Weight function used in prediction
enum class voting_mode {
    /// Uniform weights for neighbors for prediction voting.
    uniform,
    /// Weight neighbors by the inverse of their distance.
    distance
};
} // namespace v1

using v1::voting_mode;

namespace task {
namespace v1 {
/// Tag-type that parameterizes entities used for solving
/// :capterm:`classification problem <classification>`.
struct classification {};

/// Tag-type that parameterizes entities used for solving
/// the :capterm:`search problem <search>`.
struct search {};

/// Alias tag-type for classification task.
using by_default = classification;
} // namespace v1

using v1::classification;
using v1::search;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
/// Tag-type that denotes :ref:`k-d tree <knn_t_math_kd_tree>` computational method.
struct kd_tree {};

/// Tag-type that denotes :ref:`brute-force <knn_t_math_brute_force>` computational
/// method.
struct brute_force {};

/// Alias tag-type for :ref:`brute-force <knn_t_math_brute_force>` computational
/// method.
using by_default = brute_force;
} // namespace v1

using v1::kd_tree;
using v1::brute_force;
using v1::by_default;

} // namespace method

/// Represents result option flag
/// Behaves like a regular :expr`enum`.
class result_option_id : public result_option_id_base {
public:
    result_option_id() : result_option_id_base{} {}
    result_option_id(const result_option_id_base& base) : result_option_id_base{ base } {}
};

namespace detail {

ONEDAL_EXPORT result_option_id get_indices_id();
ONEDAL_EXPORT result_option_id get_distances_id();
ONEDAL_EXPORT result_option_id get_responses_id();

} // namespace detail

/// Result options are used to define
/// what should algorithm return
namespace result_options {

const result_option_id indices = detail::get_indices_id();
const result_option_id distances = detail::get_distances_id();
const result_option_id responses = detail::get_responses_id();

} // namespace result_options

namespace detail {
namespace v1 {
struct descriptor_tag {};
template <typename Task>
class descriptor_impl;

template <typename Task>
class model_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v =
    dal::detail::is_one_of_v<Method, method::kd_tree, method::brute_force>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::classification, task::search>;

template <typename Distance>
constexpr bool is_valid_distance_v =
    dal::detail::is_tag_one_of_v<Distance,
                                 minkowski_distance::detail::descriptor_tag,
                                 chebyshev_distance::detail::descriptor_tag,
                                 cosine_distance::detail::descriptor_tag>;

template <typename T>
using enable_if_search_t = std::enable_if_t<std::is_same_v<std::decay_t<T>, task::search>>;

template <typename T>
using enable_if_classification_t =
    std::enable_if_t<std::is_same_v<std::decay_t<T>, task::classification>>;

template <typename T>
using enable_if_brute_force_t =
    std::enable_if_t<std::is_same_v<std::decay_t<T>, method::brute_force>>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);
    friend detail::distance_accessor;

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;
    using distance_t = minkowski_distance::descriptor<float_t>;

    descriptor_base();

    std::int64_t get_class_count() const;
    std::int64_t get_neighbor_count() const;
    voting_mode get_voting_mode() const;
    result_option_id get_result_options() const;

protected:
    explicit descriptor_base(const detail::distance_ptr& distance);

    void set_class_count_impl(std::int64_t value);
    void set_neighbor_count_impl(std::int64_t value);
    void set_voting_mode_impl(voting_mode value);
    void set_distance_impl(const detail::distance_ptr& distance);
    const detail::distance_ptr& get_distance_impl() const;
    void set_result_options_impl(const result_option_id& value);

private:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor_tag;
using v1::descriptor_impl;
using v1::model_impl;
using v1::descriptor_base;

using v1::is_valid_float_v;
using v1::is_valid_method_v;
using v1::is_valid_task_v;
using v1::is_valid_distance_v;
using v1::enable_if_search_t;
using v1::enable_if_classification_t;
using v1::enable_if_brute_force_t;

} // namespace detail

namespace v1 {

/// @tparam Float       The floating-point type that the algorithm uses for
///                     intermediate computations. Can be :expr:`float` or
///                     :expr:`double`.
/// @tparam Method      Tag-type that specifies an implementation of algorithm. Can
///                     be :expr:`method::brute_force` or :expr:`method::kd_tree`.
/// @tparam Task        Tag-type that specifies type of the problem to solve. Can
///                     be :expr:`task::classification`.
/// @tparam Distance    The descriptor of the distance used for computations. Can be
///                     :expr:`minkowski_distance::descriptor` or
///                     :expr:`chebyshev_distance::descriptor`
template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default,
          typename Distance = oneapi::dal::minkowski_distance::descriptor<Float>>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);
    static_assert(detail::is_valid_distance_v<Distance>,
                  "Custom distances for kNN is not supported. "
                  "Use one of the predefined distances.");

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;
    using distance_t = Distance;

    /// Creates a new instance of the class with the given :literal:`class_count`
    /// and :literal:`neighbor_count` property values
    explicit descriptor(std::int64_t class_count, std::int64_t neighbor_count)
            : base_t(std::make_shared<detail::distance<distance_t>>(distance_t{})) {
        set_class_count(class_count);
        set_neighbor_count(neighbor_count);
    }

    /// Creates a new instance of the class with the given :literal:`class_count`,
    /// :literal:`neighbor_count` and :literal:`distance` property values.
    /// Used with :expr:`method::brute_force` only.
    template <typename M = Method, typename = detail::enable_if_brute_force_t<M>>
    explicit descriptor(std::int64_t class_count,
                        std::int64_t neighbor_count,
                        const distance_t& distance)
            : base_t(std::make_shared<detail::distance<distance_t>>(distance)) {
        set_class_count(class_count);
        set_neighbor_count(neighbor_count);
    }

    /// Creates a new instance of the class with the given :literal:`neighbor_count`
    /// property value.
    /// Used with :expr:`task::search` only.
    template <typename T = Task, typename = detail::enable_if_search_t<T>>
    explicit descriptor(std::int64_t neighbor_count) : base_t() {
        set_neighbor_count(neighbor_count);
    }

    /// Creates a new instance of the class with the given :literal:`neighbor_count`
    /// and :literal:`distance` property values.
    /// Used with :expr:`task::search` only.
    template <typename T = Task, typename = detail::enable_if_search_t<T>>
    explicit descriptor(std::int64_t neighbor_count, const distance_t& distance)
            : base_t(std::make_shared<detail::distance<distance_t>>(distance)) {
        set_neighbor_count(neighbor_count);
    }

    /// The number of classes c
    /// @invariant :expr:`class_count > 1`
    std::int64_t get_class_count() const {
        return base_t::get_class_count();
    }

    auto& set_class_count(std::int64_t value) {
        base_t::set_class_count_impl(value);
        return *this;
    }

    /// The number of neighbors k
    /// @invariant :expr:`neighbor_count > 0`
    std::int64_t get_neighbor_count() const {
        return base_t::get_neighbor_count();
    }

    auto& set_neighbor_count(std::int64_t value) {
        base_t::set_neighbor_count_impl(value);
        return *this;
    }

    /// The voting mode.
    voting_mode get_voting_mode() const {
        return base_t::get_voting_mode();
    }

    auto& set_voting_mode(voting_mode value) {
        base_t::set_voting_mode_impl(value);
        return *this;
    }

    /// Choose distance type for calculations. Used with :expr:`method::brute_force` only.
    template <typename M = Method, typename = detail::enable_if_brute_force_t<M>>
    const distance_t& get_distance() const {
        using dist_t = detail::distance<distance_t>;
        const auto dist = std::static_pointer_cast<dist_t>(base_t::get_distance_impl());
        return dist;
    }

    template <typename M = Method, typename = detail::enable_if_brute_force_t<M>>
    auto& set_distance(const distance_t& dist) {
        base_t::set_distance_impl(std::make_shared<detail::distance<distance_t>>(dist));
        return *this;
    }

    /// Choose which results should be computed and returned.
    result_option_id get_result_options() const {
        return base_t::get_result_options();
    }

    auto& set_result_options(const result_option_id& value) {
        base_t::set_result_options_impl(value);
        return *this;
    }
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::classification`.
template <typename Task = task::by_default>
class model : public base {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;
    friend dal::detail::serialization_accessor;

public:
    /// Creates a new instance of the class with the default property values.
    model();

private:
    void serialize(dal::detail::output_archive& ar) const;
    void deserialize(dal::detail::input_archive& ar);

    explicit model(const std::shared_ptr<detail::model_impl<Task>>& impl);
    dal::detail::pimpl<detail::model_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor;
using v1::model;

} // namespace oneapi::dal::knn
