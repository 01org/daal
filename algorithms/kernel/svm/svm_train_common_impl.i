/* file: svm_train_common_impl.i */
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
//  SVM training algorithm implementation
//--
*/

#ifndef __SVM_TRAIN_COMMON_IMPL_I__
#define __SVM_TRAIN_COMMON_IMPL_I__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_data_utils.h"
#include "externals/service_ittnotify.h"
#include "algorithms/kernel/svm/svm_train_result.h"
#include "algorithms/kernel/svm/svm_train_common.h"

#if defined(__INTEL_COMPILER)
    #if defined(_M_AMD64) || defined(__amd64) || defined(__x86_64) || defined(__x86_64__)
        #if (__CPUID__(DAAL_CPU) == __avx512__)

            #include <immintrin.h>
            #include "algorithms/kernel/svm/svm_train_avx512_impl.i"

        #endif // __CPUID__(DAAL_CPU) == __avx512__
    #endif     // defined (_M_AMD64) || defined (__amd64) || defined (__x86_64) || defined (__x86_64__)
#endif         // __INTEL_COMPILER

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
/**
 * \brief Working set selection (WSS3) function.
 *        Select an index i from a pair of indices B = {i, j} using WSS 3 algorithm from [1].
 *
 * \param[in] nActiveVectors    number of observations in a training data set that are used
 *                              in sequential minimum optimization at the current iteration
 * \param[out] Bi            resulting index i
 *
 * \return The function returns m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha)
 */
template <typename algorithmFPType, CpuType cpu>
algorithmFPType HelperTrainSVM<algorithmFPType, cpu>::WSSi(size_t nActiveVectors, const algorithmFPType * grad, const char * I, int & Bi)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(findMaximumViolatingPair.WSSi);

    Bi                   = -1;
    algorithmFPType GMin = (MaxVal<algorithmFPType>::get()); // some big negative number

    /* Find i index of the working set (Bi) */
    for (size_t i = 0; i < nActiveVectors; i++)
    {
        const algorithmFPType objFunc = grad[i];
        if ((I[i] & up) && objFunc < GMin)
        {
            GMin = objFunc;
            Bi   = i;
        }
    }
    return GMin;
}

template <typename algorithmFPType, CpuType cpu>
void HelperTrainSVM<algorithmFPType, cpu>::WSSjLocalBaseline(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                                             const algorithmFPType * kernelDiag, const algorithmFPType * grad, const char * I,
                                                             const algorithmFPType GMin, const algorithmFPType Kii, const algorithmFPType tau,
                                                             int & Bj, algorithmFPType & GMax, algorithmFPType & GMax2, algorithmFPType & delta)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(findMaximumViolatingPair.WSSj.WSSjLocal.WSSjLocalBaseline);
    algorithmFPType fpMax = MaxVal<algorithmFPType>::get();
    GMax                  = -fpMax; // some big negative number
    GMax2                 = -fpMax;

    const algorithmFPType zero(0.0);
    const algorithmFPType two(2.0);

    for (size_t j = jStart; j < jEnd; j++)
    {
        const algorithmFPType gradj = grad[j];
        if ((I[j] & low) != low)
        {
            continue;
        }
        if (gradj > GMax2)
        {
            GMax2 = gradj;
        }
        if (gradj < GMin)
        {
            continue;
        }

        algorithmFPType b = GMin - gradj;
        algorithmFPType a = Kii + kernelDiag[j] - two * KiBlock[j - jStart];
        if (a <= zero)
        {
            a = tau;
        }
        algorithmFPType dt      = b / a;
        algorithmFPType objFunc = b * dt;
        if (objFunc > GMax)
        {
            GMax  = objFunc;
            Bj    = j;
            delta = -dt;
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
void HelperTrainSVM<algorithmFPType, cpu>::WSSjLocal(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                                     const algorithmFPType * kernelDiag, const algorithmFPType * grad, const char * I,
                                                     const algorithmFPType GMin, const algorithmFPType Kii, const algorithmFPType tau, int & Bj,
                                                     algorithmFPType & GMax, algorithmFPType & GMax2, algorithmFPType & delta)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(findMaximumViolatingPair.WSSj.WSSjLocal);

    WSSjLocalBaseline(jStart, jEnd, KiBlock, kernelDiag, grad, I, GMin, Kii, tau, Bj, GMax, GMax2, delta);
}

template <typename algorithmFPType, CpuType cpu>
services::Status SubDataTaskCSR<algorithmFPType, cpu>::copyDataByIndices(const uint32_t * wsIndices, const NumericTablePtr & xTable)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(cache.copyDataByIndices.CSR);
    CSRNumericTableIface * csrIface = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(xTable.get()));
    _rowOffsets[0]                  = 1;
    size_t dataIndex                = 0;
    const size_t nRows              = this->_nSubsetVectors;
    for (size_t i = 0; i < nRows; i++)
    {
        size_t iRows = wsIndices[i];
        ReadRowsCSR<algorithmFPType, cpu> mtX(csrIface, iRows, 1);
        DAAL_CHECK_BLOCK_STATUS(mtX);
        const size_t nNonZeroValuesInRow = mtX.rows()[1] - mtX.rows()[0];
        const size_t * colIndices        = mtX.cols();
        const algorithmFPType * values   = mtX.values();
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nNonZeroValuesInRow; j++, dataIndex++)
        {
            this->_data[dataIndex] = values[j];
            _colIndices[dataIndex] = colIndices[j];
        }
        _rowOffsets[i + 1] = _rowOffsets[i] + nNonZeroValuesInRow;
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status SubDataTaskDense<algorithmFPType, cpu>::copyDataByIndices(const uint32_t * wsIndices, const NumericTablePtr & xTable)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(cache.copyDataByIndices.Dense);
    NumericTable & x       = *xTable.get();
    const size_t p         = x.getNumberOfColumns();
    const size_t blockSize = 1;
    const size_t nBlock    = this->_nSubsetVectors;
    SafeStatus safeStat;
    daal::threader_for(nBlock, nBlock, [&](const size_t iBlock) {
        size_t iRows = wsIndices[iBlock];
        ReadRows<algorithmFPType, cpu> mtX(x, iRows, 1);
        DAAL_CHECK_BLOCK_STATUS_THR(mtX);
        const algorithmFPType * const dataIn = mtX.get();
        algorithmFPType * dataOut            = this->_data.get() + iBlock * p;
        DAAL_CHECK_THR(!services::internal::daal_memcpy_s(dataOut, p * sizeof(algorithmFPType), dataIn, p * sizeof(algorithmFPType)),
                       services::ErrorMemoryCopyFailedInternal);
    });

    return safeStat.detach();
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
