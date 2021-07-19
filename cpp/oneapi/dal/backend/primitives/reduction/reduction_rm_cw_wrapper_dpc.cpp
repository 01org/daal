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

#include "oneapi/dal/backend/primitives/reduction/common.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw<Float, BinaryOp, UnaryOp>::reduction_rm_cw(sycl::queue& q) : q_{ q } {}

template <typename Float, typename BinaryOp, typename UnaryOp>
auto reduction_rm_cw<Float, BinaryOp, UnaryOp>::propose_method(std::int64_t width,
                                                               std::int64_t height) const
    -> reduction_method {
    const auto wg = device_max_wg_size(q_);
    if constexpr (is_sum && is_flt) {
        constexpr int sacc_wide_folding = sacc_wide_t::max_folding;
        if (((wg * sacc_wide_folding) >= width) && (width >= wg)) {
            return reduction_method::super_accum_wide;
        }
    }
    if ((height >= wg) && (height > width)) {
        return reduction_method::naive_local;
    }
    return reduction_method::naive;
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw<Float, BinaryOp, UnaryOp>::operator()(reduction_method method,
                                                                  const Float* input,
                                                                  Float* output,
                                                                  std::int64_t width,
                                                                  std::int64_t height,
                                                                  std::int64_t stride,
                                                                  const BinaryOp& binary,
                                                                  const UnaryOp& unary,
                                                                  const event_vector& deps) const {
    // TODO: think about `switch` operator
    if (method == reduction_method::naive) {
        const naive_t kernel{ q_ };
        return kernel(input, output, width, height, stride, binary, unary, deps);
    }
    if (method == reduction_method::naive_local) {
        const naive_local_t kernel{ q_ };
        return kernel(input, output, width, height, stride, binary, unary, deps);
    }
    if constexpr (is_sum && is_flt) {
        if (method == reduction_method::super_accum_wide) {
            const sacc_wide_t kernel{ q_ };
            return kernel(input, output, width, height, stride, binary, unary, deps);
        }
    }
    ONEDAL_ASSERT(false);
    return sycl::event{};
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw<Float, BinaryOp, UnaryOp>::operator()(const Float* input,
                                                                  Float* output,
                                                                  std::int64_t width,
                                                                  std::int64_t height,
                                                                  std::int64_t stride,
                                                                  const BinaryOp& binary,
                                                                  const UnaryOp& unary,
                                                                  const event_vector& deps) const {
    const auto method = propose_method(width, height);
    return this->operator()(method, input, output, width, height, stride, binary, unary, deps);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw<Float, BinaryOp, UnaryOp>::operator()(reduction_method method,
                                                                  const Float* input,
                                                                  Float* output,
                                                                  std::int64_t width,
                                                                  std::int64_t height,
                                                                  const BinaryOp& binary,
                                                                  const UnaryOp& unary,
                                                                  const event_vector& deps) const {
    return this->operator()(method, input, output, width, height, width, binary, unary, deps);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw<Float, BinaryOp, UnaryOp>::operator()(const Float* input,
                                                                  Float* output,
                                                                  std::int64_t width,
                                                                  std::int64_t height,
                                                                  const BinaryOp& binary,
                                                                  const UnaryOp& unary,
                                                                  const event_vector& deps) const {
    const auto method = propose_method(width, height);
    return this->operator()(method, input, output, width, height, width, binary, unary, deps);
}

#define INSTANTIATE(F, B, U) template class reduction_rm_cw<F, B, U>;

#define INSTANTIATE_FLOAT(B, U)                \
    INSTANTIATE(double, B<double>, U<double>); \
    INSTANTIATE(float, B<float>, U<float>);

INSTANTIATE_FLOAT(min, identity)
INSTANTIATE_FLOAT(min, abs)
INSTANTIATE_FLOAT(min, square)

INSTANTIATE_FLOAT(max, identity)
INSTANTIATE_FLOAT(max, abs)
INSTANTIATE_FLOAT(max, square)

INSTANTIATE_FLOAT(sum, identity)
INSTANTIATE_FLOAT(sum, abs)
INSTANTIATE_FLOAT(sum, square)

#undef INSTANTIATE_FLOAT

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
