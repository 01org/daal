/* file: svm_parameter.h */
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
//  Implementation of the class defining the SVM model.
//--
*/

#ifndef __SVM_PARAMETER_H__
#define __SVM_PARAMETER_H__

#include "algorithms/model.h"
#include "algorithms/kernel_function/kernel_function.h"
#include "algorithms/kernel_function/kernel_function_linear.h"
#include "algorithms/kernel_function/kernel_function_types.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup svm Support Vector Machine Classifier
 * \copydoc daal::algorithms::svm
 * @ingroup classification
 * @{
 */
/**
 * \brief Contains classes to work with the support vector machine classifier
 */
namespace svm
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup svm
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__SVM__PARAMETER"></a>
 * \brief Optional parameters
 *
 * \snippet svm/svm_parameter.h Parameter source code
 */
/* [interface1::Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::interface1::Parameter
{
    Parameter(const services::SharedPtr<kernel_function::KernelIface> & kernelForParameter =
                  services::SharedPtr<kernel_function::KernelIface>(new kernel_function::linear::Batch<>()),
              double C = 1.0, double accuracyThreshold = 0.001, double tau = 1.0e-6, size_t maxIterations = 1000000, size_t cacheSize = 8000000,
              bool doShrinking = true, size_t shrinkingStep = 1000)
        : C(C),
          accuracyThreshold(accuracyThreshold),
          tau(tau),
          maxIterations(maxIterations),
          cacheSize(cacheSize),
          doShrinking(doShrinking),
          shrinkingStep(shrinkingStep),
          kernel(kernelForParameter) {};

    double C;                                           /*!< Upper bound in constraints of the quadratic optimization problem */
    double accuracyThreshold;                           /*!< Training accuracy */
    double tau;                                         /*!< Tau parameter of the working set selection scheme */
    size_t maxIterations;                               /*!< Maximal number of iterations for the algorithm */
    size_t cacheSize;                                   /*!< Size of cache in bytes to store values of the kernel matrix.
                                     A non-zero value enables use of a cache optimization technique */
    bool doShrinking;                                   /*!< Flag that enables use of the shrinking optimization technique */
    size_t shrinkingStep;                               /*!< Number of iterations between the steps of shrinking optimization technique */
    algorithms::kernel_function::KernelIfacePtr kernel; /*!< Kernel function */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [interface1::Parameter source code] */
} // namespace interface1

/**
 * \brief Contains version 2.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface2
{
/**
 * @ingroup svm
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__SVM__PARAMETER"></a>
 * \brief Optional parameters
 *
 * \snippet svm/svm_parameter.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter
{
    Parameter(const services::SharedPtr<kernel_function::KernelIface> & kernelForParameter =
                  services::SharedPtr<kernel_function::KernelIface>(new kernel_function::linear::Batch<>()),
              double C = 1.0, double epsilon = 0.1, double accuracyThreshold = 0.001, double tau = 1.0e-6, size_t maxIterations = 1000000,
              size_t cacheSize = 8000000, bool doShrinking = true, size_t shrinkingStep = 1000)
        : C(C),
          accuracyThreshold(accuracyThreshold),
          tau(tau),
          maxIterations(maxIterations),
          cacheSize(cacheSize),
          doShrinking(doShrinking),
          shrinkingStep(shrinkingStep),
          kernel(kernelForParameter) {};

    double C;                                           /*!< Upper bound in constraints of the quadratic optimization problem */
    double epsilon;                                     /*!< Upper bound in constraints of the quadratic optimization problem */
    double accuracyThreshold;                           /*!< Training accuracy */
    double tau;                                         /*!< Tau parameter of the working set selection scheme */
    size_t maxIterations;                               /*!< Maximal number of iterations for the algorithm */
    size_t cacheSize;                                   /*!< Size of cache in bytes to store values of the kernel matrix.
                                     A non-zero value enables use of a cache optimization technique */
    bool doShrinking;                                   /*!< Flag that enables use of the shrinking optimization technique */
    size_t shrinkingStep;                               /*!< Number of iterations between the steps of shrinking optimization technique */
    algorithms::kernel_function::KernelIfacePtr kernel; /*!< Kernel function */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */
} // namespace interface2

/**
 * \brief Contains version 3.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface3
{
/**
 * @ingroup svm
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__SVM__PARAMETER"></a>
 * \brief Optional parameters
 *
 * \snippet svm/svm_parameter.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter
{
    Parameter();

    double C;                                           /*!< Upper bound in constraints of the quadratic optimization problem */
    double epsilon;                                     /*!< The error tolerance parameter of the loss function for regression task. */
    double accuracyThreshold;                           /*!< Training accuracy */
    double tau;                                         /*!< Tau parameter of the working set selection scheme */
    size_t maxIterations;                               /*!< Maximal number of iterations for the algorithm */
    size_t cacheSize;                                   /*!< Size of cache in bytes to store values of the kernel matrix.
                                                            A non-zero value enables use of a cache optimization technique */
    bool doShrinking;                                   /*!< Flag that enables use of the shrinking optimization technique */
    size_t shrinkingStep;                               /*!< Number of iterations between the steps of shrinking optimization technique */
    size_t maxInnerIteration;                           /*!< TDB */
    algorithms::kernel_function::KernelIfacePtr kernel; /*!< Kernel function */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */
} // namespace interface3

using interface3::Result;

#endif
