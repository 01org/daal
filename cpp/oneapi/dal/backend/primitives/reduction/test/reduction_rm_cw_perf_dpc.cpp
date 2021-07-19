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

#include <array>
#include <cmath>
#include <type_traits>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

constexpr auto rm_order = ndorder::c;

using reduction_types = std::tuple<std::tuple<float, sum<float>, square<float>>>;

template <typename Param>
class reduction_rm_test_uniform : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        width_ = GENERATE(16, 128, 1024);
        stride_ = GENERATE(16, 128, 1024);
        height_ = GENERATE(16, 128, 1024, 16384, 32768);
        CAPTURE(width_, stride_, height_);
    }

    void generate_special() {
        width_ = GENERATE(28, 960, 2000, 3072);
        height_ = GENERATE(1024, 4096);
        stride_ = height_;
        CAPTURE(width_, stride_, height_);
    }

    bool is_initialized() const {
        return width_ > 0 && stride_ > 0 && height_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    bool should_be_skipped() {
        if (width_ > stride_) {
            return true;
        }
        return false;
    }

    auto input() {
        check_if_initialized();
        return ndarray<float_t, 2, rm_order>::zeros(this->get_queue(),
                                                    { stride_, height_ },
                                                    sycl::usm::alloc::device);
    }

    auto output(std::int64_t size) {
        check_if_initialized();
        return ndarray<float_t, 1, rm_order>::zeros(this->get_queue(),
                                                    { size },
                                                    sycl::usm::alloc::device);
    }

    auto fpt_desc() {
        if constexpr (std::is_same_v<float_t, float>) {
            return "float";
        }
        if constexpr (std::is_same_v<float_t, double>) {
            return "double";
        }
        REQUIRE(false);
        return "unknown type";
    }

    auto type_desc() {
        return fmt::format("Floating Point Type: {}", fpt_desc());
    }

    auto matrix_desc() {
        check_if_initialized();
        return fmt::format("Row-Major Matrix with parameters: "
                           "width_ = {}, stride_ = {}, height_ = {}",
                           width_,
                           stride_,
                           height_);
    }

    auto unary_desc() {
        if (std::is_same_v<identity<float_t>, unary_t>) {
            return "Identity Unary Functor";
        }
        if (std::is_same_v<abs<float_t>, unary_t>) {
            return "Abs Unary Functor";
        }
        if (std::is_same_v<square<float_t>, unary_t>) {
            return "Square Unary Functor";
        }
        REQUIRE(false);
        return "Unknown Unary Functor";
    }

    auto binary_desc() {
        if (std::is_same_v<sum<float_t>, binary_t>) {
            return "Sum Binary Functor";
        }
        if (std::is_same_v<max<float_t>, binary_t>) {
            return "Max Binary Functor";
        }
        if (std::is_same_v<min<float_t>, binary_t>) {
            return "Min Binary Functor";
        }
        REQUIRE(false);
        return "Unknown Binary Functor";
    }

    auto desc() {
        return fmt::format("{}; {}; {}; {}",
                           matrix_desc(),
                           type_desc(),
                           unary_desc(),
                           binary_desc());
    }

    template <typename reduction_t>
    void test_raw_reduce(const std::string& name) {
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(width_);

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        this->get_queue().wait_and_throw();

        BENCHMARK(name.c_str()) {
            reduction_t reducer(this->get_queue());
            reducer(inp_ptr, out_ptr, width_, height_, stride_, binary_t{}, unary_t{})
                .wait_and_throw();
        };
    }

    void test_raw_cw_reduce_naive() {
        using reduction_t = reduction_rm_cw_naive<float_t, binary_t, unary_t>;
        const auto name = fmt::format("Naive CW Reduction: {}", desc());
        test_raw_reduce<reduction_t>(name);
    }

    void test_raw_cw_reduce_naive_local() {
        using reduction_t = reduction_rm_cw_naive_local<float_t, binary_t, unary_t>;
        const auto name = fmt::format("Local CW Reduction: {}", desc());
        test_raw_reduce<reduction_t>(name);
    }

    void test_raw_cw_reduce_super_accum_wide() {
        using reduction_t = reduction_rm_cw_super_accum_wide<float_t, binary_t, unary_t>;
        const auto name = fmt::format("SuperAccumulator CW Reduction: {}", desc());
        test_raw_reduce<reduction_t>(name);
    }

    void test_raw_cw_reduce_wrapper() {
        using reduction_t = reduction_rm_cw<float_t, binary_t, unary_t>;
        const auto name = fmt::format("Wrapper CW Reduction: {}", desc());
        test_raw_reduce<reduction_t>(name);
    }

private:
    std::int64_t width_;
    std::int64_t stride_;
    std::int64_t height_;
};

TEMPLATE_LIST_TEST_M(reduction_rm_test_uniform,
                     "Uniformly filled Row-Major Col-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    SKIP_IF(this->should_be_skipped());
    this->test_raw_cw_reduce_naive();
    this->test_raw_cw_reduce_naive_local();
    this->test_raw_cw_reduce_super_accum_wide();
    this->test_raw_cw_reduce_wrapper();
}

TEMPLATE_LIST_TEST_M(reduction_rm_test_uniform,
                     "Uniformly filled Row-Major Col-Wise reduction"
                     " - Specififc Dimensions",
                     "[reduction][rm][special][small]",
                     reduction_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate_special();
    SKIP_IF(this->should_be_skipped());
    this->test_raw_cw_reduce_naive();
    this->test_raw_cw_reduce_naive_local();
    this->test_raw_cw_reduce_super_accum_wide();
    this->test_raw_cw_reduce_wrapper();
}

} // namespace oneapi::dal::backend::primitives::test
