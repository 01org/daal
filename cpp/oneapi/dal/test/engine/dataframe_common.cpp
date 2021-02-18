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

#include "oneapi/dal/test/engine/dataframe_common.hpp"

#include <list>
#include <random>
#include <unordered_map>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::test::engine {

class dataframe_builder_cache_entry {
public:
    dataframe_builder_cache_entry(const dataframe& df, std::size_t generation)
            : df_(df),
              generation_(generation) {}

    const dataframe& get_df() const {
        return df_;
    }

    std::size_t get_generation() const {
        return generation_;
    }

    void decrement_generation(std::size_t decrement) {
        ONEDAL_ASSERT(generation_ >= decrement);
        generation_ -= decrement;
    }

private:
    dataframe df_;
    std::size_t generation_;
};

class dataframe_builder_cache {
public:
    using cache_entry = dataframe_builder_cache_entry;

    static dataframe_builder_cache& get_instance() {
        static dataframe_builder_cache cache;
        return cache;
    }

    std::tuple<dataframe, bool> lookup(const dataframe_builder_program& program) {
        const auto it = map_.find(program.get_code());
        if (it == map_.end()) {
            const auto df = program.execute();
            store(program.get_code(), df);
            return { df, false };
        }
        return { it->second.get_df(), true };
    }

    double get_occupied_size_mb() const {
        return cache_size_ / 1024.0 / 1024.0;
    }

private:
    dataframe_builder_cache() = default;

    void store(const std::string& key, const dataframe& df) {
        try_remove_stale_elements(df.get_size());

        // We store new entry even if its size does not fit the cache
        // until new entry comes
        map_.emplace(key, cache_entry{ df, size_stack.size() });
        size_stack.push_back(df.get_size());
        cache_size_ += df.get_size();

#ifdef ONEDAL_DEBUG
        check_cache_invariants();
#endif
    }

    void try_remove_stale_elements(std::size_t new_entry_size) {
        ONEDAL_ASSERT(size_stack.size() == map_.size());

        // Do nothing if cache is empty
        if (size_stack.size() == 0) {
            return;
        }

        // Do nothing if cache allows storing more elements
        if (cache_size_ + new_entry_size <= max_cache_size_) {
            return;
        }

        // If new entry is too large, remove all entries
        if (new_entry_size >= max_cache_size_) {
            return clear();
        }

        // If new_entry_size leads to overflow, remove all entries
        if (cache_size_ + new_entry_size < cache_size_) {
            return clear();
        }

        // Compute generation cut point -- all entries below
        // the cut point are considered stale and shall be removed
        std::size_t cut_generation = 0;
        std::size_t cache_size_after_removal = cache_size_;
        for (std::size_t entry_size : size_stack) {
            cache_size_after_removal -= entry_size;
            if (cache_size_after_removal + new_entry_size < max_cache_size_) {
                break;
            }
            cut_generation++;
        }
        ONEDAL_ASSERT(cache_size_after_removal < cache_size_);
        ONEDAL_ASSERT(cut_generation < size_stack.size());

        // Remove stale entry sizes from stack
        for (std::size_t i = 0; i <= cut_generation; i++) {
            size_stack.pop_front();
        }

        // Remove stale entries from map
        for (auto it = map_.begin(); it != map_.end();) {
            const std::size_t gen = it->second.get_generation();
            if (gen <= cut_generation) {
                it = map_.erase(it);
            }
            else {
                it->second.decrement_generation(cut_generation + 1);
                ++it;
            }
        }

        cache_size_ = cache_size_after_removal;
    }

    void clear() {
        map_.clear();
        size_stack.clear();
        cache_size_ = 0;
    }

#ifdef ONEDAL_DEBUG
    void check_cache_invariants() const {
        ONEDAL_ASSERT(size_stack.size() == map_.size());

        std::vector<size_t> size_stack_vec;
        size_stack_vec.reserve(size_stack.size());
        for (std::size_t size : size_stack) {
            size_stack_vec.push_back(size);
        }

        std::size_t check_cache_size = 0;
        for (const auto& pair : map_) {
            const std::size_t gen = pair.second.get_generation();
            ONEDAL_ASSERT(gen < size_stack_vec.size());

            std::size_t df_size = pair.second.get_df().get_size();
            std::size_t gen_size = size_stack_vec[gen];
            ONEDAL_ASSERT(df_size == gen_size);

            check_cache_size += df_size;
        }
        ONEDAL_ASSERT(check_cache_size == cache_size_);
    }
#endif

    std::size_t cache_size_ = 0;
    std::list<size_t> size_stack;
    std::unordered_map<std::string, cache_entry> map_;

    // Default max cache size is 500Mb
    std::size_t max_cache_size_ = 500 * 1024 * 1024;
};

static dataframe_builder_cache& get_dataframe_builder_cache() {
    return dataframe_builder_cache::get_instance();
}

class dataframe_builder_action_allocate : public dataframe_builder_action {
public:
    explicit dataframe_builder_action_allocate(std::int64_t row_count, std::int64_t column_count)
            : row_count_(row_count),
              column_count_(column_count) {
        if (row_count == 0 || column_count == 0) {
            throw invalid_argument{ fmt::format(
                "Invalid dataframe shape, row and column count must be positive, "
                "but got row_count = {}, column_count = {}",
                row_count,
                column_count) };
        }
    }

    std::string get_opcode() const override {
        return fmt::format("allocate({},{})", row_count_, column_count_);
    }

    dataframe_impl* execute(dataframe_impl* df) const override {
        delete df;
        const auto arr = array<float>::empty(row_count_ * column_count_);
        return new dataframe_impl{ arr, row_count_, column_count_ };
    }

private:
    std::int64_t row_count_;
    std::int64_t column_count_;
};

class dataframe_builder_action_read_external_dataset : public dataframe_builder_action {
private:
    enum dataset_extensions { csv };

public:
    explicit dataframe_builder_action_read_external_dataset(const std::string& dataset)
            : dataset_(dataset) {
        dataset_ = trim_path_left(dataset_);
        dataset_ = trim_path_right(dataset_);
    }

    std::string get_opcode() const override {
        return fmt::format("read_external_dataset({})", dataset_);
    }

    dataframe_impl* execute(dataframe_impl* df) const override {
        delete df;

        std::string datasets_root = get_dataset_root();

        if (get_extension(dataset_) == dataset_extensions::csv) {
            const auto data_table =
                dal::read<dal::table>(dal::csv::data_source{ datasets_root + '/' + dataset_ });

            return new dataframe_impl{ dal::row_accessor<const float>{ data_table }.pull(),
                                       data_table.get_row_count(),
                                       data_table.get_column_count() };
        }
        else {
            throw internal_error{ fmt::format("{} dataset extension was not handled",
                                              get_extension(dataset_)) };
        }
    }

private:
    std::string trim_path_left(const std::string& path) const {
        std::string tmp(path);
        if (tmp.size() > 0 && tmp[0] == '/') {
            tmp.erase(0, 1);
        }
        return tmp;
    }

    std::string trim_path_right(const std::string& path) const {
        std::string tmp(path);
        if (tmp.size() > 0 && tmp[tmp.size() - 1] == '/') {
            tmp.erase(tmp.size() - 1, 1);
        }
        return tmp;
    }

    std::string get_dataset_root() const {
        const char* dataset_root_ptr = std::getenv("DAAL_DATASETS");
        if (dataset_root_ptr == nullptr) {
            throw invalid_argument{ "DAAL_DATASETS environment variable is unset" };
        }
        return trim_path_right(dataset_root_ptr);
    }

    dataset_extensions get_extension(const std::string& path) const {
        std::int64_t dot = static_cast<std::int64_t>(path.size()) - 1;
        while (dot >= 0 && path[dot] != '.') {
            --dot;
        }

        if (dot >= 0) {
            const std::string extension =
                path.substr(dot, static_cast<std::int64_t>(path.size()) - dot);
            if (extension == ".csv") {
                return dataset_extensions::csv;
            }
            throw unimplemented{ fmt::format("{} dataset extension is not supported", extension) };
        }
        throw invalid_argument{ "Dataset doesn't have any extension" };
    }

    std::string dataset_;
};

class dataframe_builder_action_fill_uniform : public dataframe_builder_action {
public:
    explicit dataframe_builder_action_fill_uniform(double a, double b, std::int64_t seed)
            : a_(a),
              b_(b),
              seed_(seed) {
        if (a >= b) {
            throw invalid_argument{ fmt::format("Invalid uniform distribution interval, "
                                                "expected b > a, but got a = {}, b = {}",
                                                a,
                                                b) };
        }
    }

    std::string get_opcode() const override {
        return fmt::format("fill_uniform({},{},{})", a_, b_, seed_);
    }

    dataframe_impl* execute(dataframe_impl* df) const override {
        if (!df) {
            throw invalid_argument{ "Action fill_uniform got null dataframe" };
        }

        float* data = df->get_array().need_mutable_data().get_mutable_data();

        // TODO: Migrate to MKL's random generators
        std::mt19937 rng(seed_);

        std::uniform_real_distribution<float> distr(a_, b_);
        for (std::int64_t i = 0; i < df->get_count(); i++) {
            data[i] = distr(rng);
        }

        return df;
    }

private:
    double a_ = 0.0;
    double b_ = 1.0;
    std::int64_t seed_ = 7777;
};

class dataframe_builder_action_fill_cluster : public dataframe_builder_action {
public:
    explicit dataframe_builder_action_fill_cluster(double a, double b, std::int64_t seed)
            : a_(a),
              b_(b),
              seed_(seed) {
        if (b < 0) {
            throw invalid_argument{ fmt::format("Invalid normal distribution interval, "
                                                "expected b >= 0, but got b = {}",
                                                b) };
        }
    }

    std::string get_opcode() const override {
        return fmt::format("fill_cluster({},{},{})", a_, b_, seed_);
    }

    dataframe_impl* execute(dataframe_impl* df) const override {
        if (!df) {
            throw invalid_argument{ "Action fill_normal got null dataframe" };
        }

        float* data = df->get_array().need_mutable_data().get_mutable_data();

        // TODO: Migrate to MKL's random generators
        std::mt19937 rng(seed_);

        std::normal_distribution<float> distr(a_, b_);
        for (std::int64_t i = 0; i < df->get_count(); i++) {
            data[i] = distr(rng);
        }

        return df;
    }

private:
    double a_ = 5.0;
    double b_ = 2.0;
    std::int64_t seed_ = 7777;
};

dataframe dataframe_builder_program::execute() const {
    dataframe_impl* impl = nullptr;
    for (const auto& action : actions_) {
        impl = action->execute(impl);
    }
    return dataframe{ impl };
}

dataframe_builder_impl::dataframe_builder_impl(std::int64_t row_count, std::int64_t column_count) {
    program_.add<dataframe_builder_action_allocate>(row_count, column_count);
}

dataframe_builder_impl::dataframe_builder_impl(const std::string& dataset) {
    program_.add<dataframe_builder_action_read_external_dataset>(dataset);
}

dataframe_builder& dataframe_builder::fill_uniform(double a, double b, std::int64_t seed) {
    impl_->get_program().add<dataframe_builder_action_fill_uniform>(a, b, seed);
    return *this;
}

dataframe_builder& dataframe_builder::fill_cluster(double a, double b, std::int64_t seed) {
    impl_->get_program().add<dataframe_builder_action_fill_cluster>(a, b, seed);
    return *this;
}

dataframe dataframe_builder::build() const {
    const auto& program = impl_->get_program();
    const auto df_hit = get_dataframe_builder_cache().lookup(program);
#ifdef ONEDAL_DEBUG_DATAFRAMES_CACHE
    const std::string hit_or_miss = std::get<1>(df_hit) ? "hit" : "miss";
    fmt::print("{}\t{}\t{:.2f}Mb\n",
               hit_or_miss,
               program.get_code(),
               get_dataframe_builder_cache().get_occupied_size_mb());
#endif
    return std::get<0>(df_hit);
}

template <typename Float>
static homogen_table wrap_to_homogen_table(host_test_policy& policy,
                                           const array<Float>& data,
                                           std::int64_t row_count,
                                           std::int64_t column_count) {
    return dal::detail::homogen_table_builder{}.reset(data, row_count, column_count).build();
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Float>
static homogen_table wrap_to_homogen_table(device_test_policy& policy,
                                           const array<Float>& data,
                                           std::int64_t row_count,
                                           std::int64_t column_count) {
    return dal::detail::homogen_table_builder{}
        .allocate(policy.get_queue(), row_count, column_count)
        .copy_data(policy.get_queue(), data.get_data(), row_count, column_count)
        .build();
}
#endif

static array<double> convert_to_f64(const array<float>& data) {
    auto data_f64 = array<double>::empty(data.get_count());

    const float* data_ptr = data.get_data();
    double* data_f64_ptr = data_f64.get_mutable_data();

    for (std::int64_t i = 0; i < data.get_count(); i++) {
        data_f64_ptr[i] = data_ptr[i];
    }

    return data_f64;
}

template <typename Policy>
static homogen_table wrap_to_homogen_table(Policy& policy,
                                           const array<float>& data,
                                           const table_id& id,
                                           std::int64_t row_count,
                                           std::int64_t column_count) {
    if (id.get_float_type() == table_float_type::f32) {
        return wrap_to_homogen_table(policy, data, row_count, column_count);
    }
    else if (id.get_float_type() == table_float_type::f64) {
        auto data_f64 = convert_to_f64(data);
        return wrap_to_homogen_table(policy, data_f64, row_count, column_count);
    }
    else {
        throw unimplemented{ "Only f32 and f64 floating point types are supported" };
    }
}

template <typename Policy>
static homogen_table build_homogen_table(Policy& policy,
                                         const array<float>& data,
                                         const table_id& id,
                                         std::int64_t row_count,
                                         std::int64_t column_count,
                                         std::int64_t first_column_idx,
                                         std::int64_t last_column_idx) {
    const std::int64_t actual_column_count = column_count;
    const std::int64_t required_column_count = last_column_idx - first_column_idx;

    ONEDAL_ASSERT(required_column_count >= 0);

    if (first_column_idx == 0 && actual_column_count == required_column_count) {
        return wrap_to_homogen_table(policy, data, id, row_count, required_column_count);
    }
    else {
        auto dst = array<float>::empty(row_count * required_column_count);
        float* dst_ptr = dst.get_mutable_data();
        const float* src_ptr = data.get_data();

        for (std::int64_t i = 0; i < row_count; ++i) {
            for (std::int64_t j = 0; j < required_column_count; ++j) {
                dst_ptr[i * required_column_count + j] =
                    src_ptr[i * actual_column_count + first_column_idx + j];
            }
        }

        return wrap_to_homogen_table(policy, dst, id, row_count, required_column_count);
    }
}

template <typename Policy>
static table build_table(Policy& policy, const dataframe& df, const table_id& id, range r) {
    std::int64_t first_column_idx = r.start_idx;
    std::int64_t last_column_idx = r.end_idx;

    if (last_column_idx <= 0) {
        last_column_idx += df.get_column_count();
    }

    const auto data = df.get_array();
    const std::int64_t row_count = df.get_row_count();
    const std::int64_t column_count = df.get_column_count();

    if (first_column_idx < 0) {
        throw invalid_argument{ "first_column_idx should be >= 0" };
    }
    if (first_column_idx >= last_column_idx) {
        throw invalid_argument{ "first_column_idx should be < last_column_idx" };
    }
    if (last_column_idx > column_count) {
        throw invalid_argument{ "last_column_idx should be <= data column count" };
    }

    if (id.get_kind() == table_kind::homogen) {
        return build_homogen_table(policy,
                                   data,
                                   id,
                                   row_count,
                                   column_count,
                                   first_column_idx,
                                   last_column_idx);
    }
    else {
        throw unimplemented{ "Only homogen table is supported" };
    }
}

table dataframe::get_table(host_test_policy& policy, const table_id& id, range r) const {
    return build_table(policy, *this, id, r);
}

#ifdef ONEDAL_DATA_PARALLEL
table dataframe::get_table(device_test_policy& policy, const table_id& id, range r) const {
    return build_table(policy, *this, id, r);
}
#endif

} // namespace oneapi::dal::test::engine
