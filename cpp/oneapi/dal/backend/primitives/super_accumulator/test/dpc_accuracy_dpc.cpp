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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/super_accumulator/super_accumulator.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using accuracy_types = std::tuple<float>;

template <typename Float>
class accuracy_test_random : public te::float_algo_fixture<Float> {
public:
    using target = pr::super_accumulators<Float, false>;

    void generate() {
        length_ = GENERATE(1, 5, 251, 1023, 1025, 32768, 262144);
        upper_bound_ = GENERATE(+1.e-5, +100.0, +201.0, +1024.0);
        lower_bound_ = GENERATE(-1024.0, -5.0, -1.0, -1.0e-3);
        CAPTURE(length_, lower_bound_, upper_bound_);
        generate_input();
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    bool is_initialized() const {
        return length_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "Accuracy test is not initialized" };
        }
    }

    auto temp_buffer(std::int64_t size = target::min_buffer_size) {
        check_if_initialized();
        return ndarray<std::int64_t, 1, ndorder::c>::zeros(this->get_queue(), { size });
    }

    void generate_input() {
        check_if_initialized();
        const auto train_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder(1ul, length_).fill_uniform(lower_bound_, upper_bound_));
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }

    template <typename AccFloat = Float>
    AccFloat compute_gt() {
        const auto data = row_accessor<const Float>{ this->input_table_ }.pull();
        AccFloat accumulator = 0.0;
        for (std::int64_t i = 0; i < length_; ++i) {
            accumulator += AccFloat(data[i]);
        }
        return accumulator;
    }

    Float compute_res() {
        auto [temp_buff, temp_event] = temp_buffer();
        sycl::event temp_buf_fill_event = std::move(temp_event);
        std::int64_t* const temp_ptr = temp_buff.get_mutable_data();
        const auto data = row_accessor<const Float>{ this->input_table_ }.pull(this->get_queue());
        const Float* const data_ptr = data.get_data();
        auto red_event = this->get_queue().submit([&](sycl::handler& h) {
            h.depends_on({ temp_buf_fill_event });
            h.parallel_for(this->length_, [=](sycl::id<1> idx) {
                target(temp_ptr).add(data_ptr[idx]);
            });
        });
        auto [res_buff, res_event] =
            ndarray<Float, 1, ndorder::c>::zeros(this->get_queue(), { 1ul });
        sycl::event res_buf_fill_event = std::move(res_event);
        Float* const res_ptr = res_buff.get_mutable_data();
        auto fin_event = this->get_queue().submit([&](sycl::handler& h) {
            h.depends_on({ red_event, res_buf_fill_event });
            h.parallel_for(1ul, [=](sycl::id<1> idx) {
                res_ptr[idx] = target(temp_ptr).finalize();
            });
        });
        return *(res_buff.to_host(this->get_queue(), { fin_event }).get_data());
    }

    void accuracy_test(const double tol = 1.e-6) {
        check_if_initialized();
        CAPTURE(length_, lower_bound_, upper_bound_);
        const Float gtd = compute_gt<double>();
        const Float gtf = compute_gt<float>();
        const Float res = compute_res();
        const double rerr = double(res) / double(gtd) - 1.0;
        CAPTURE(rerr, gtd, gtf, res);
        if ((-tol < rerr) && (rerr > tol)) {
            FAIL();
        }
    }

private:
    table input_table_;
    Float upper_bound_;
    Float lower_bound_;
    std::int64_t length_;
};

TEMPLATE_LIST_TEST_M(accuracy_test_random,
                     "Randomly filled reduction with super-accumulator",
                     "[reduction][accuracy]",
                     accuracy_types) {
    this->generate();
    this->accuracy_test();
}

} // namespace oneapi::dal::backend::primitives::test
