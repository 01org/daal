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

#include "oneapi/dal/backend/primitives/lapack/eigen.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/micromkl/micromkl.hpp"

namespace oneapi::dal::backend::primitives {

template <typename... Args>
inline void syevd(Args&&... args) {
    dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
        using dal::backend::micromkl::syevd;
        syevd<decltype(cpu)>(std::forward<Args>(args)...);
    });
}

template <typename Float>
void sym_eigvals_impl(Float* a, std::int64_t n, std::int64_t lda, Float* w) {
    ONEDAL_ASSERT(a);
    ONEDAL_ASSERT(w);
    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(lda >= n);

    const std::int64_t lwork = 2 * n * n + 6 * n + 1;
    const std::int64_t liwork = 5 * n + 3;

    ONEDAL_ASSERT(lwork > n);
    ONEDAL_ASSERT(liwork > n);

    const auto work = ndarray<Float, 1>::empty(lwork);
    const auto iwork = ndarray<std::int64_t, 1>::empty(liwork);

    Float* work_ptr = work.get_mutable_data();
    std::int64_t* iwork_ptr = iwork.get_mutable_data();

    std::int64_t info = 0;
    syevd('V', 'U', n, a, lda, w, work_ptr, lwork, iwork_ptr, liwork, info);

    if (info != 0) {
        throw internal_error{ dal::detail::error_messages::failed_to_compute_eigenvectors() };
    }
}

template <typename Float>
void flip_eigvals_impl(Float* a,
                       Float* w,
                       std::int64_t n,
                       std::int64_t lda,
                       std::int64_t w_count,
                       Float* a_flipped,
                       std::int64_t lda_flipped,
                       Float* w_flipped) {
    dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
        flip_eigvals_impl_cpu<decltype(cpu)>(a,
                                             w,
                                             n,
                                             lda,
                                             w_count,
                                             a_flipped,
                                             lda_flipped,
                                             w_flipped);
    });
}

#define INSTANTIATE(F)                                                  \
    template void sym_eigvals_impl(F*, std::int64_t, std::int64_t, F*); \
    template void                                                       \
    flip_eigvals_impl(F*, F*, std::int64_t, std::int64_t, std::int64_t, F*, std::int64_t, F*);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::primitives
