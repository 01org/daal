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

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/primitives/rng/partial_shuffle.hpp"

#include <daal/src/externals/service_rng.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename... Args>
inline void uniform_by_cpu(Args&&... args) {
    dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
        daal::internal::
            RNGs<size_t, oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value>{}
                .uniform(std::forward<Args>(args)...);
    });
}

void partial_shuffle::generate(ndview<size_t, 1>& result_array, size_t top) {
    daal::algorithms::engines::EnginePtr engine =
        daal::algorithms::engines::mt19937::Batch<>::create(777);
    ONEDAL_ASSERT(engine.get() != nullptr);
    auto engine_impl =
        dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&(*engine));
    ONEDAL_ASSERT(engine_impl != nullptr);
    const std::uint64_t count = result_array.get_count();
    const auto casted_count = dal::detail::integral_cast<size_t>(count);
    ONEDAL_ASSERT(casted_count < top);
    auto indices_ptr = result_array.get_mutable_data();

    std::uint64_t k = 0;
    for (size_t i = 0; i < casted_count; i++) {
        uniform_by_cpu(1, indices_ptr + i, engine_impl->getState(), i, top);
        ONEDAL_ASSERT(indices_ptr[i] >= 0);
        size_t& value = indices_ptr[i];
        for (size_t j = i; j > 0; j--) {
            if (value == indices_ptr[j - 1]) {
                value = j - 1;
            }
        }
        if (value >= top)
            continue;
        k++;
    }
    ONEDAL_ASSERT(k == count);
};

#endif

} // namespace oneapi::dal::backend::primitives
