/* file: service_kernel_math.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Implementation of math functions.
//--
*/

#ifndef __SERVICE_KERNEL_MATH_H__
#define __SERVICE_KERNEL_MATH_H__

#include "services/daal_defines.h"
#include "services/error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_allocators.h"
#include "src/services/service_arrays.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_lapack.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace internal
{
template <typename FPType, CpuType cpu>
FPType distancePow2(const FPType * a, const FPType * b, size_t dim)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += (b[i] - a[i]) * (b[i] - a[i]);
    }

    return sum;
}

template <typename FPType, CpuType cpu>
FPType distancePow(const FPType * a, const FPType * b, size_t dim, FPType p)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += daal::internal::Math<FPType, cpu>::sPowx(b[i] - a[i], p);
    }

    return sum;
}

template <typename FPType, CpuType cpu>
FPType distance(const FPType * a, const FPType * b, size_t dim, FPType p)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += daal::internal::Math<FPType, cpu>::sPowx(b[i] - a[i], p);
    }

    return daal::internal::Math<FPType, cpu>::sPowx(sum, (FPType)1.0 / p);
}

template <typename FPType, CpuType cpu>
class PairwiseDistances
{
public:
    virtual ~PairwiseDistances() {};

    virtual services::Status init() = 0;

    virtual services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                          FPType * const res)                                                             = 0;
    virtual services::Status computeBatch(size_t aOffset, size_t aSize, size_t bOffset, size_t bSize, FPType * const res) = 0;
    virtual services::Status computeFull(FPType * const res)                                                              = 0;
};

// compute: sum(A^2, 2) + sum(B^2, 2) -2*A*B'
template <typename FPType, CpuType cpu>
class EuclideanDistances : public PairwiseDistances<FPType, cpu>
{
public:
    EuclideanDistances(const NumericTable & a, const NumericTable & b, bool squared = true) : _a(a), _b(b), _squared(squared) {}

    virtual ~EuclideanDistances() {}

    virtual services::Status init()
    {
        services::Status s;

        normBufferA.reset(_a.getNumberOfRows());
        DAAL_CHECK_MALLOC(normBufferA.get());
        s |= computeNorm(_a, normBufferA.get());

        if (&_a != &_b)
        {
            normBufferB.reset(_b.getNumberOfRows());
            DAAL_CHECK_MALLOC(normBufferB.get());
            s |= computeNorm(_b, normBufferB.get());
        }

        return s;
    }

    // output:  Row-major matrix of size { aSize x bSize }
    virtual services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                          FPType * const res)
    {
        const size_t nRowsA = aSize;
        const size_t nColsA = _a.getNumberOfColumns();
        const size_t nRowsB = bSize;

        const size_t nRowsC = nRowsA;
        const size_t nColsC = nRowsB;

        computeABt(a, b, nRowsA, nColsA, nRowsB, res);

        const FPType * const aa = normBufferA.get() + aOffset;
        const FPType * const bb = (&_a == &_b) ? normBufferA.get() + bOffset : normBufferB.get() + bOffset;

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nRowsC; i++)
        {
            for (size_t j = 0; j < nColsC; j++)
            {
                res[i * nColsC + j] = aa[i] + bb[j] - 2 * res[i * nColsC + j];
            }
        }

        if (!_squared)
        {
            daal::internal::Math<FPType, cpu> math;
            daal::services::internal::TArray<FPType, cpu> tmpArr(nRowsC * nColsC);
            FPType * tmp = tmpArr.get();
            math.vSqrt(nRowsC * nColsC, res, tmp);

            services::internal::daal_memcpy_s(res, nRowsC * nColsC * sizeof(FPType), tmp, nRowsC * nColsC * sizeof(FPType));
        }

        return services::Status();
    }

    // output:  Row-major matrix of size { aSize x bSize }
    virtual services::Status computeBatch(size_t aOffset, size_t aSize, size_t bOffset, size_t bSize, FPType * const res)
    {
        ReadRows<FPType, cpu> aDataRows(const_cast<NumericTable *>(&_a), aOffset, aSize);
        DAAL_CHECK_BLOCK_STATUS(aDataRows);
        const FPType * const aData = aDataRows.get();

        ReadRows<FPType, cpu> bDataRows(const_cast<NumericTable *>(&_b), bOffset, bSize);
        DAAL_CHECK_BLOCK_STATUS(bDataRows);
        const FPType * const bData = bDataRows.get();

        return computeBatch(aData, bData, aOffset, aSize, bOffset, bSize, res);
    }

    // output:  Row-major matrix of size { nrows(A) x nrows(B) }
    virtual services::Status computeFull(FPType * const res)
    {
        SafeStatus safeStat;

        const size_t nRowsA    = _a.getNumberOfRows();
        const size_t blockSize = 256;
        const size_t nBlocks   = nRowsA / blockSize + (nRowsA % blockSize > 0);

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t i1    = iBlock * blockSize;
            const size_t i2    = (iBlock + 1 == nBlocks ? nRowsA : i1 + blockSize);
            const size_t iSize = i2 - i1;

            const size_t nRowsB = _b.getNumberOfRows();

            DAAL_CHECK_STATUS_THR(computeBatch(i1, iSize, 0, nRowsB, res + i1 * nRowsB));
        });

        return safeStat.detach();
    }

protected:
    // compute (sum(A^2, 2))
    services::Status computeNorm(const NumericTable & ntData, FPType * const res)
    {
        const size_t nRows = ntData.getNumberOfRows();
        const size_t nCols = ntData.getNumberOfColumns();

        const size_t blockSize = 512;
        const size_t nBlocks   = nRows / blockSize + !!(nRows % blockSize);

        SafeStatus safeStat;

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            size_t begin = iBlock * blockSize;
            size_t end   = services::internal::min<cpu, size_t>(begin + blockSize, nRows);

            ReadRows<FPType, cpu> dataRows(const_cast<NumericTable &>(ntData), begin, end - begin);
            DAAL_CHECK_BLOCK_STATUS_THR(dataRows);
            const FPType * const data = dataRows.get();

            FPType * r = res + begin;

            for (size_t i = 0; i < end - begin; i++)
            {
                FPType sum = FPType(0);
                PRAGMA_IVDEP
                PRAGMA_ICC_NO16(omp simd reduction(+ : sum))
                for (size_t j = 0; j < nCols; j++)
                {
                    sum += data[i * nCols + j] * data[i * nCols + j];
                }
                r[i] = sum;
            }
        });

        return safeStat.detach();
    }

    // compute (A x B')
    void computeABt(const FPType * const a, const FPType * const b, const size_t nRowsA, const size_t nColsA, const size_t nRowsB, FPType * const out)
    {
        const char transa    = 't';
        const char transb    = 'n';
        const DAAL_INT _m    = nRowsB;
        const DAAL_INT _n    = nRowsA;
        const DAAL_INT _k    = nColsA;
        const FPType alpha   = 1.0;
        const DAAL_INT lda   = nColsA;
        const DAAL_INT ldy   = nColsA;
        const FPType beta    = 0.0;
        const DAAL_INT ldaty = nRowsB;

        Blas<FPType, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, b, &lda, a, &ldy, &beta, out, &ldaty);
    }

    const NumericTable & _a;
    const NumericTable & _b;
    const bool _squared;

    TArray<FPType, cpu> normBufferA;
    TArray<FPType, cpu> normBufferB;
};

template <typename FPType, CpuType cpu>
bool solveEquationsSystemWithCholesky(FPType * a, FPType * b, size_t n, size_t nX, bool sequential)
{
    /* POTRF and POTRS parameters */
    char uplo     = 'U';
    DAAL_INT info = 0;

    /* Perform L*L' factorization of A */
    if (sequential)
    {
        Lapack<FPType, cpu>::xxpotrf(&uplo, (DAAL_INT *)&n, a, (DAAL_INT *)&n, &info);
    }
    else
    {
        Lapack<FPType, cpu>::xpotrf(&uplo, (DAAL_INT *)&n, a, (DAAL_INT *)&n, &info);
    }
    if (info != 0) return false;

    /* Solve L*L' * x = b */
    if (sequential)
    {
        Lapack<FPType, cpu>::xxpotrs(&uplo, (DAAL_INT *)&n, (DAAL_INT *)&nX, a, (DAAL_INT *)&n, b, (DAAL_INT *)&n, &info);
    }
    else
    {
        Lapack<FPType, cpu>::xpotrs(&uplo, (DAAL_INT *)&n, (DAAL_INT *)&nX, a, (DAAL_INT *)&n, b, (DAAL_INT *)&n, &info);
    }
    return (info == 0);
}

template <typename FPType, CpuType cpu>
bool solveEquationsSystemWithPLU(FPType * a, FPType * b, size_t n, size_t nX, bool sequential, bool extendFromSymmetric)
{
    if (extendFromSymmetric)
    {
        /* Extend symmetric matrix to generic through filling of upper triangle */
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < i; ++j)
            {
                a[j * n + i] = a[i * n + j];
            }
        }
    }

    /* GETRF and GETRS parameters */
    char trans    = 'N';
    DAAL_INT info = 0;

    TArrayScalable<DAAL_INT, cpu> ipiv(n);
    DAAL_CHECK_MALLOC(ipiv.get());

    /* Perform P*L*U factorization of A */
    if (sequential)
    {
        Lapack<FPType, cpu>::xxgetrf((DAAL_INT *)&n, (DAAL_INT *)&n, a, (DAAL_INT *)&n, ipiv.get(), &info);
    }
    else
    {
        Lapack<FPType, cpu>::xgetrf((DAAL_INT *)&n, (DAAL_INT *)&n, a, (DAAL_INT *)&n, ipiv.get(), &info);
    }
    if (info != 0) return false;

    /* Solve P*L*U * x = b */
    if (sequential)
    {
        Lapack<FPType, cpu>::xxgetrs(&trans, (DAAL_INT *)&n, (DAAL_INT *)&nX, a, (DAAL_INT *)&n, ipiv.get(), b, (DAAL_INT *)&n, &info);
    }
    else
    {
        Lapack<FPType, cpu>::xgetrs(&trans, (DAAL_INT *)&n, (DAAL_INT *)&nX, a, (DAAL_INT *)&n, ipiv.get(), b, (DAAL_INT *)&n, &info);
    }
    return (info == 0);
}

template <typename FPType, CpuType cpu>
bool solveSymmetricEquationsSystem(FPType * a, FPType * b, size_t n, size_t nX, bool sequential)
{
    /* Copy data for fallback from Cholesky to PLU factorization */
    TArrayScalable<FPType, cpu> aCopy(n * n);
    TArrayScalable<FPType, cpu> bCopy(n);
    DAAL_CHECK_MALLOC(aCopy.get());
    DAAL_CHECK_MALLOC(bCopy.get());

    int copy_status = services::internal::daal_memcpy_s(aCopy.get(), n * n * sizeof(FPType), a, n * n * sizeof(FPType));
    copy_status |= services::internal::daal_memcpy_s(bCopy.get(), n * sizeof(FPType), b, n * sizeof(FPType));

    if (copy_status != 0) return false;

    /* Try to solve with Cholesky factorization */
    if (!solveEquationsSystemWithCholesky<FPType, cpu>(a, b, n, nX, sequential))
    {
        /* Fallback to PLU factorization */
        bool status = solveEquationsSystemWithPLU<FPType, cpu>(aCopy.get(), bCopy.get(), n, nX, sequential, true);
        if (status)
        {
            status |= services::internal::daal_memcpy_s(b, n * sizeof(FPType), bCopy.get(), n * sizeof(FPType));
        }
        return status;
    }
    return true;
}
} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
