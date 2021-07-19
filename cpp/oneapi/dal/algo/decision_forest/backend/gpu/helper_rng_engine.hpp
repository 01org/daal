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

#include <daal/src/externals/service_rng.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>
#include <daal/src/algorithms/engines/engine_types_internal.h>
#include <daal/include/algorithms/engines/mt2203/mt2203.h>

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Integer, typename Size = std::uint64_t>
class rng {
public:
    rng() = default;
    ~rng() = default;

    // add convertion of error into exception
    int uniform(Size count, Integer* dst, void* state, Integer a, Integer b) {
        // add convertion to size_t
        return daal_rng_.uniform(count, dst, state, a, b);
    }

    // add convertion of error into exception
    int uniform_without_replacement(Size count,
                                    Integer* dst,
                                    Integer* buffer,
                                    void* state,
                                    Integer a,
                                    Integer b) {
        // add convertion to size_t
        return daal_rng_.uniformWithoutReplacement(count, dst, buffer, state, a, b);
    }

    cl::sycl::event shuffle(Size count, Integer* dst, void* state) {
        Integer idx[2];

        for (Size i = 0; i < count; ++i) {
            daal_rng_.uniform(2,
                              idx,
                              state,
                              0,
                              count); //TODO add processing of error returned by uniform
            std::swap(dst[idx[0]], dst[idx[1]]);
        }

        return cl::sycl::event{};
    }

private:
    sycl::queue queue_;
    daal::internal::RNGs<Integer, daal::sse2> daal_rng_;
};

class engine {
public:
    engine() : engine_(daal::algorithms::engines::mt2203::Batch<>::create()) {
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(engine_.get());
        ONEDAL_ASSERT(impl_ != nullptr);
    }
    explicit engine(const daal::algorithms::engines::EnginePtr& eng) : engine_(eng) {
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(engine_.get());
        ONEDAL_ASSERT(impl_ != nullptr);
    }
    virtual ~engine() = default;

    engine& operator=(const daal::algorithms::engines::EnginePtr& eng) {
        engine_ = eng;
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(engine_.get());
        ONEDAL_ASSERT(impl_ != nullptr);

        return *this;
    }

    void* get_state() const {
        return impl_->getState();
    }

private:
    daal::algorithms::engines::EnginePtr engine_;
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

class engine_impl {
public:
    engine_impl() {}
    explicit engine_impl(const daal::algorithms::engines::EnginePtr& eng) {
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(eng.get());
        ONEDAL_ASSERT(impl_ != nullptr);
    }
    virtual ~engine_impl() = default;

    engine_impl& operator=(const daal::algorithms::engines::EnginePtr& eng) {
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(eng.get());
        ONEDAL_ASSERT(impl_ != nullptr);

        return *this;
    }

    void* get_state() const {
        return impl_->getState();
    }

private:
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

template <typename Size = std::uint64_t>
class engine_collection {
public:
    engine_collection(Size count)
            : count_(count),
              engine_(daal::algorithms::engines::mt2203::Batch<>::create()),
              params_(count),
              technique_(daal::algorithms::engines::internal::family),
              engines_(count) {}

    template <typename Op>
    dal::array<engine_impl> operator()(Op&& op) {
        daal::services::Status status;
        for (Size i = 0; i < count_; i++) {
            op(i, params_.nSkip[i]);
        }
        select_parallelization_technique(technique_);
        daal::algorithms::engines::internal::EnginesCollection<daal::sse2> enginesCollection(
            engine_,
            technique_,
            params_,
            engines_,
            &status);
        if (!status) {
            //throw;
        }

        dal::array<engine_impl> arr = dal::array<engine_impl>::empty(count_);
        engine_impl* arr_data = arr.get_mutable_data();
        for (Size i = 0; i < count_; i++) {
            if (arr_data)
                arr_data[i] = engines_[i];
        }

        return arr;
    }

private:
    void select_parallelization_technique(
        daal::algorithms::engines::internal::ParallelizationTechnique& technique) {
        auto engineImpl =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(engine_.get());

        daal::algorithms::engines::internal::ParallelizationTechnique techniques[] = {
            daal::algorithms::engines::internal::family,
            daal::algorithms::engines::internal::leapfrog,
            daal::algorithms::engines::internal::skipahead
        };

        for (auto& techn : techniques) {
            if (engineImpl->hasSupport(techn)) {
                technique = techn;
                return;
            }
        }
        // throw exception;
    }

private:
    Size count_;
    daal::algorithms::engines::EnginePtr engine_;
    daal::algorithms::engines::internal::Params<daal::sse2> params_;
    daal::algorithms::engines::internal::ParallelizationTechnique technique_;
    daal::services::internal::TArray<daal::algorithms::engines::EnginePtr, daal::sse2> engines_;
};

#endif

} // namespace oneapi::dal::decision_forest::backend
