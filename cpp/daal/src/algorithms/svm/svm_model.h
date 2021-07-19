/* file: svm_model.h */
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
//  Implementation of the class defining the SVM model.
//--
*/

#ifndef __SVM_INTERNAL_MODEL_H__
#define __SVM_INTERNAL_MODEL_H__

#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "algorithms/model.h"
#include "algorithms/kernel_function/kernel_function.h"
#include "algorithms/kernel_function/kernel_function_linear.h"
#include "algorithms/kernel_function/kernel_function_types.h"
#include "algorithms/classifier/classifier_model.h"

#include "algorithms/svm/svm_model.h"
#include "src/algorithms/svm/svm_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace internal
{
struct DAAL_EXPORT Parameter : public classifier::Parameter
{
    Parameter(const services::SharedPtr<kernel_function::KernelIface> & kernelForParameter =
                  services::SharedPtr<kernel_function::KernelIface>(new kernel_function::linear::Batch<>()),
              double C = 1.0, double accuracyThreshold = 0.001, double tau = 1.0e-6, size_t maxIterations = 1000000, size_t cacheSize = 8000000,
              bool doShrinking = true, size_t shrinkingStep = 1000, double epsilon = 0.1, double nu = 0.5,
              svm::training::internal::SvmType svmType = svm::training::internal::SvmType::classification)
        : C(C),
          accuracyThreshold(accuracyThreshold),
          tau(tau),
          maxIterations(maxIterations),
          cacheSize(cacheSize),
          doShrinking(doShrinking),
          shrinkingStep(shrinkingStep),
          kernel(kernelForParameter),
          epsilon(epsilon),
          nu(nu),
          svmType(svmType) {};

    Parameter(const svm::training::internal::KernelParameter & kernelParameter)
        : C(kernelParameter.C),
          accuracyThreshold(kernelParameter.accuracyThreshold),
          tau(kernelParameter.tau),
          maxIterations(kernelParameter.maxIterations),
          cacheSize(kernelParameter.cacheSize),
          doShrinking(kernelParameter.doShrinking),
          shrinkingStep(kernelParameter.shrinkingStep),
          kernel(kernelParameter.kernel),
          epsilon(kernelParameter.epsilon),
          nu(kernelParameter.nu),
          svmType(kernelParameter.svmType) {};

    double C;                                           /*!< Upper bound in constraints of the quadratic optimization
               problem */
    double accuracyThreshold;                           /*!< Training accuracy */
    double tau;                                         /*!< Tau parameter of the working set selection scheme */
    size_t maxIterations;                               /*!< Maximal number of iterations for the algorithm */
    size_t cacheSize;                                   /*!< Size of cache in bytes to store values of the kernel
 matrix.
 A non-zero value enables use of a cache optimization technique */
    bool doShrinking;                                   /*!< Flag that enables use of the shrinking optimization
                       technique */
    size_t shrinkingStep;                               /*!< Number of iterations between the steps of shrinking
                        optimization technique */
    algorithms::kernel_function::KernelIfacePtr kernel; /*!< Kernel function */
    double epsilon;
    double nu;
    svm::training::internal::SvmType svmType;

    services::Status check() const DAAL_C11_OVERRIDE;
};

} // namespace internal
} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
