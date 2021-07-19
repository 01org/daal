/* file: svm_train.h */
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

/*
//++
//  Implementation of the interface for SVM model-based training in the batch
//  processing mode
//--
*/

#ifndef __SVM_TRAIN_INTERNAL_H__
#define __SVM_TRAIN_INTERNAL_H__

#include "algorithms/algorithm.h"

#include "algorithms/svm/svm_train_types.h"
#include "src/algorithms/svm/svm_model.h"
#include "algorithms/classifier/classifier_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    BatchContainer(daal::services::Environment::env * daalEnv);

    ~BatchContainer();

    services::Status compute() DAAL_C11_OVERRIDE;
};

template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = boser>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef typename super::InputType InputType;
    typedef algorithms::svm::internal::Parameter ParameterType;
    typedef algorithms::svm::training::Result ResultType;

    ParameterType parameter;
    InputType input;

    Batch() { initialize(); };

    Batch(size_t nClasses)
    {
        parameter.nClasses = nClasses;
        initialize();
    }

    Batch(const Batch<algorithmFPType, method> & other) : classifier::training::Batch(other), parameter(other.parameter), input(other.input)
    {
        initialize();
    }

    virtual ~Batch() {}

    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    ResultPtr getResult() { return ResultType::cast(_result); }

    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>(&input, _par, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};

} // namespace internal

} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
