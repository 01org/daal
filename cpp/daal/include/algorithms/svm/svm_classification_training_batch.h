/* file: svm_classification_training_batch.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef __SVM_CLASSIFICATION_TRAINING_BATCH_H__
#define __SVM_CLASSIFICATION_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"

#include "algorithms/svm/svm_train_types.h"
#include "algorithms/classifier/classifier_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace interface1
{
/**
 * @defgroup svm_training_batch Batch
 * @ingroup svm_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__BATCHCONTAINER"></a>
 *  \brief Class containing methods to compute results of the SVM training
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVM training algorithm, double or float
 * \tparam method           SVM training computation method, \ref daal::algorithms::svm::training::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for SVM model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    DAAL_DEPRECATED ~BatchContainer();
    /**
     * Computes the result of SVM  model-based training in the batch processing mode
     *
     * \return Status of computation
     */
    DAAL_DEPRECATED services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__BATCH"></a>
 *  \brief %Algorithm class to train the SVM model
 *  <!-- \n<a href="DAAL-REF-SVM-ALGORITHM">SVM algorithm description and usage models</a> -->
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for the SVM training algorithm, double or float
 *  \tparam method           SVM training method, \ref Method
 *
 *  \par Enumerations
 *      - \ref classifier::training::InputId Identifiers of SVM training input objects
 *      - \ref classifier::training::ResultId Identifiers of SVM training results
 *      - \ref Method   SVM training methods
 *
 * \par References
 *      - \ref interface1::Input "Input" class
 *      - \ref interface1::Model "Model" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = boser>
class DAAL_EXPORT Batch : public classifier::training::interface1::Batch
{
public:
    typedef classifier::training::interface1::Batch super;

    typedef typename super::InputType InputType;
    typedef algorithms::svm::interface1::Parameter ParameterType;
    typedef algorithms::svm::training::Result ResultType;

    ParameterType parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */
    InputType input;         /*!< %Input objects of the algorithm */

    /** Default constructor */
    DAAL_DEPRECATED Batch() { initialize(); };

    /**
     * Constructs an SVM training algorithm by copying input objects and parameters
     * of another SVM training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    DAAL_DEPRECATED Batch(const Batch<algorithmFPType, method> & other)
        : classifier::training::interface1::Batch(other), parameter(other.parameter), input(other.input)
    {
        initialize();
    }

    DAAL_DEPRECATED_VIRTUAL virtual ~Batch() {}

    /**
     * Get input objects for the SVM training algorithm
     * \return %Input objects for the SVM training algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    DAAL_DEPRECATED_VIRTUAL virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns structure that contains computed results of the SVM training algorithm
     * \return Structure that contains computed results of the SVM training algorithm
     */
    DAAL_DEPRECATED interface1::ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the training results of the classification algorithm
     */
    DAAL_DEPRECATED_VIRTUAL services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated SVM training algorithm with a copy of input objects
     * and parameters of this SVM training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    DAAL_DEPRECATED services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

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
/** @} */
} // namespace interface1

namespace interface2
{
/**
 * @defgroup svm_training_batch Batch
 * @ingroup svm_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__BATCHCONTAINER"></a>
 *  \brief Class containing methods to compute results of the SVM training
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVM training algorithm, double or float
 * \tparam method           SVM training computation method, \ref daal::algorithms::svm::training::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for SVM model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    DAAL_DEPRECATED ~BatchContainer();
    /**
     * Computes the result of SVM  model-based training in the batch processing mode
     *
     * \return Status of computation
     */
    DAAL_DEPRECATED services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__BATCH"></a>
 *  \brief %Algorithm class to train the SVM model
 *  <!-- \n<a href="DAAL-REF-SVM-ALGORITHM">SVM algorithm description and usage models</a> -->
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for the SVM training algorithm, double or float
 *  \tparam method           SVM training method, \ref Method
 *
 *  \par Enumerations
 *      - \ref classifier::training::InputId Identifiers of SVM training input objects
 *      - \ref classifier::training::ResultId Identifiers of SVM training results
 *      - \ref Method   SVM training methods
 *
 * \par References
 *      - \ref interface1::Input "Input" class
 *      - \ref interface1::Model "Model" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = boser>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef typename super::InputType InputType;
    typedef algorithms::svm::Parameter ParameterType;
    typedef algorithms::svm::training::Result ResultType;

    ParameterType parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */
    InputType input;         /*!< %Input objects of the algorithm */

    /** Default constructor */
    DAAL_DEPRECATED Batch() { initialize(); };

    /**
     * Constructs an SVM training algorithm with nClasses parameter
     * \param[in] nClasses   number of classes
    */
    DAAL_DEPRECATED Batch(size_t nClasses)
    {
        parameter.nClasses = nClasses;
        initialize();
    }

    /**
     * Constructs an SVM training algorithm by copying input objects and parameters
     * of another SVM training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    DAAL_DEPRECATED Batch(const Batch<algorithmFPType, method> & other)
        : classifier::training::Batch(other), parameter(other.parameter), input(other.input)
    {
        initialize();
    }

    DAAL_DEPRECATED_VIRTUAL virtual ~Batch() {}

    /**
     * Get input objects for the SVM training algorithm
     * \return %Input objects for the SVM training algorithm
     */
    DAAL_DEPRECATED_VIRTUAL InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    DAAL_DEPRECATED_VIRTUAL virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns structure that contains computed results of the SVM training algorithm
     * \return Structure that contains computed results of the SVM training algorithm
     */
    DAAL_DEPRECATED ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the training results of the classification algorithm
     */
    DAAL_DEPRECATED_VIRTUAL services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated SVM training algorithm with a copy of input objects
     * and parameters of this SVM training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    DAAL_DEPRECATED services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

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
/** @} */
} // namespace interface2

using interface2::BatchContainer;
using interface2::Batch;

} // namespace training

namespace classification
{
namespace training
{
namespace interface1
{
/**
 * @defgroup svm_training_batch Batch
 * @ingroup svm_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__BATCHCONTAINER"></a>
 *  \brief Class containing methods to compute results of the SVM training
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SVM training algorithm, double or float
 * \tparam method           SVM training computation method, \ref daal::algorithms::svm::training::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for SVM model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of SVM  model-based training in the batch processing mode
     *
     * \return Status of computation
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__BATCH"></a>
 *  \brief %Algorithm class to train the SVM model
 *  <!-- \n<a href="DAAL-REF-SVM-ALGORITHM">SVM algorithm description and usage models</a> -->
 *
 *  \tparam algorithmFPType  Data type to use in intermediate computations for the SVM training algorithm, double or float
 *  \tparam method           SVM training method, \ref Method
 *
 *  \par Enumerations
 *      - \ref classifier::training::InputId Identifiers of SVM training input objects
 *      - \ref classifier::training::ResultId Identifiers of SVM training results
 *      - \ref Method   SVM training methods
 *
 * \par References
 *      - \ref interface1::Input "Input" class
 *      - \ref interface1::Model "Model" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = thunder>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef typename super::InputType InputType;
    typedef algorithms::svm::Parameter ParameterType;
    typedef algorithms::svm::classification::training::Result ResultType;

    InputType input; /*!< %Input objects of the algorithm */

    /**
     * Constructs an SVM training algorithm with nClasses parameter
     * \param[in] nClasses   number of classes
    */
    Batch(size_t nClasses = 2);
    /**
     * Constructs an SVM training algorithm by copying input objects and parameters
     * of another SVM training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    virtual ~Batch() {}

    /**
     * Get input objects for the SVM training algorithm
     * \return %Input objects for the SVM training algorithm
     */
    virtual InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns structure that contains computed results of the SVM training algorithm
     * \return Structure that contains computed results of the SVM training algorithm
     */
    ResultPtr getResult() { return ResultType::cast(_result); }

    /**
     * Resets the training results of the classification algorithm
     */
    virtual services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated SVM training algorithm with a copy of input objects
     * and parameters of this SVM training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>(&input, _par, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1

using interface1::BatchContainer;
using interface1::Batch;

} // namespace training
} // namespace classification

using namespace classification;

} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
