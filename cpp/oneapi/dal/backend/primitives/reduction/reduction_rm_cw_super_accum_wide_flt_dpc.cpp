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
#include "oneapi/dal/backend/primitives/super_accumulator/super_accumulator.hpp"

namespace oneapi::dal::backend::primitives {

template <typename UnaryOp, int f, int b>
class reduction_kernel {
    using super_accums = super_accumulators<float, false>;

public:
    constexpr static inline float zero = 0.f;
    constexpr static inline int folding = f;
    constexpr static inline int block = b;

    reduction_kernel(std::int32_t width,
                     std::int32_t stride,
                     std::int32_t height,
                     const float* data,
                     std::int64_t* bins,
                     const UnaryOp& unary)
            : width_(width),
              stride_(stride),
              height_(height),
              data_(data),
              bins_(bins),
              unary_(unary) {}

    void operator()(sycl::nd_item<2> it) const {
        // Acumulators for working in width
        float accs[folding] = { zero };
        //
        const std::int32_t vid = it.get_global_id(1);
        const std::int32_t hid = it.get_global_id(0);
        const std::int32_t hwg = it.get_global_range(0);
        //
        // Reduction Section
        //
        for (std::int32_t i = 0; i < block; ++i) {
            // Current dataset row
            const std::int64_t rid = vid * block + i;
            for (std::int32_t j = 0; j < folding; ++j) {
                // Current dataset col number
                const auto cid = hid + j * hwg;
                // Check for row and col to be in dataset
                const bool handle = (rid < height_) && (cid < width_);
                // Access to the value in row-major order
                // All arithmetics should work in std::int64_t
                const auto& val = data_[cid + rid * stride_];
                accs[j] += handle ? unary_(val) : zero;
            }
        }
        //
        // Super counter Section
        //
        for (std::int32_t j = 0; j < folding; ++j) {
            const auto cid = hid + j * hwg;
            if (cid < width_) {
                bins_.add(accs[j], cid);
            }
        }
    }

private:
    const std::int32_t width_;
    const std::int32_t stride_;
    const std::int32_t height_;
    const float* const data_;
    const super_accums bins_;
    const UnaryOp unary_;
};

template <typename UnaryOp, int folding, int block_size, typename = std::enable_if_t<folding != 0>>
sycl::event reduction_impl(sycl::queue& queue,
                           const float* data,
                           std::int64_t width,
                           std::int64_t stride,
                           std::int64_t height,
                           std::int64_t* bins,
                           const UnaryOp& unary = {},
                           const std::vector<sycl::event>& deps = {}) {
    using kernel_t = reduction_kernel<UnaryOp, folding, block_size>;
    constexpr int bl = kernel_t::block;
    const auto n_blocks = height / bl + bool(height % bl);
    const auto wg = std::min<std::int64_t>(device_max_wg_size(queue), width);
    const auto cfolding = width / wg + bool(width % wg);
    if (cfolding == folding) {
        return queue.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for<kernel_t>(make_multiple_nd_range_2d({ wg, n_blocks }, { wg, 1l }),
                                     kernel_t(dal::detail::integral_cast<std::int32_t>(width),
                                              dal::detail::integral_cast<std::int32_t>(stride),
                                              dal::detail::integral_cast<std::int32_t>(height),
                                              data,
                                              bins,
                                              unary));
        });
    }
    if constexpr (folding > 1) {
        return reduction_impl<UnaryOp, folding - 1, block_size>(queue,
                                                                data,
                                                                width,
                                                                stride,
                                                                height,
                                                                bins,
                                                                unary,
                                                                deps);
    }
    ONEDAL_ASSERT(false);
    return sycl::event();
}

class finalize_kernel {
    using super_accums = super_accumulators<float, false>;

public:
    finalize_kernel(float* results_, std::int64_t* bins_) : results(results_), all_bins(bins_) {}
    void operator()(sycl::id<1> idx) const {
        results[idx] = all_bins.finalize(idx[0]);
    }

private:
    float* const results;
    const super_accums all_bins;
};

sycl::event finalization(sycl::queue& queue,
                         float* results,
                         std::int64_t width,
                         std::int64_t* bins,
                         const std::vector<sycl::event>& deps = {}) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(make_range_1d(width), finalize_kernel(results, bins));
    });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_super_accum_wide<Float, BinaryOp, UnaryOp>::reduction_rm_cw_super_accum_wide(
    sycl::queue& q)
        : q_(q) {}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_wide<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    std::int64_t* bins,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    auto reduction_event = reduction_impl<UnaryOp, max_folding, block_size>(q_,
                                                                            input,
                                                                            width,
                                                                            stride,
                                                                            height,
                                                                            bins,
                                                                            unary,
                                                                            deps);
    return finalization(q_, output, width, bins, { reduction_event });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_wide<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t* bins,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, bins, binary, unary, deps);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_wide<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    using super_accums = super_accumulators<float, false>;
    const auto bins_size = width * super_accums::nbins;
    auto* const bins = sycl::malloc_device<std::int64_t>(bins_size, q_);
    std::vector<sycl::event> new_deps(deps.size() + 1);
    new_deps[0] = q_.fill<std::int64_t>(bins, 0ul, bins_size);
    std::copy(deps.cbegin(), deps.cend(), ++(new_deps.begin()));
    auto reduction_event =
        this->operator()(input, output, width, height, stride, bins, binary, unary, new_deps);
    reduction_event.wait_and_throw();
    sycl::free(bins, q_);
    return reduction_event;
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_wide<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, binary, unary, deps);
}

#define INSTANTIATE(F, B, U) template class reduction_rm_cw_super_accum_wide<F, B, U>;

#define INSTANTIATE_FLOAT(B, U) INSTANTIATE(float, B<float>, U<float>);

INSTANTIATE_FLOAT(sum, identity)
INSTANTIATE_FLOAT(sum, abs)
INSTANTIATE_FLOAT(sum, square)

#undef INSTANTIATE_FLOAT

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
