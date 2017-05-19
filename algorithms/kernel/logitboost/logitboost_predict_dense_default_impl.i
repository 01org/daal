/* file: logitboost_predict_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Common functions for Logit Boost predictions calculation
//--
*/

#ifndef __LOGITBOOST_PREDICT_DENSE_DEFAULT_IMPL_I__
#define __LOGITBOOST_PREDICT_DENSE_DEFAULT_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "logitboost_model.h"
#include "threading.h"
#include "daal_defines.h"

#include "service_memory.h"
#include "service_numeric_table.h"
#include "logitboost_impl.i"

using namespace daal::algorithms::logitboost::internal;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;

template<typename algorithmFPType, CpuType cpu>
services::Status LogitBoostPredictKernel<defaultDense, algorithmFPType, cpu>::compute( const NumericTablePtr& a, const Model *m, NumericTable *r, const Parameter *par )
{
    Parameter *parameter = const_cast<Parameter *>(par);
    const size_t dim = a->getNumberOfColumns();       /* Number of features in input dataset */
    const size_t n   = a->getNumberOfRows();          /* Number of observations in input dataset */
    const size_t nc  = parameter->nClasses;           /* Number of classes */
    const size_t M   = m->getIterations();            /* Number of terms of additive regression in the model */
    Model *boostModel = const_cast<Model *>(m);

    /* Allocate memory */
    TArray<algorithmFPType, cpu> pred(n * nc);
    TArray<algorithmFPType, cpu> F(n * nc); /* Additive function values */
    DAAL_CHECK(pred.get() && F.get(), services::ErrorMemoryAllocationFailed);

    daal::services::internal::service_memset<algorithmFPType, cpu>(F.get(), 0, n * nc);

    services::Status s;
    services::SharedPtr<weak_learner::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction;
    learnerPredict->inputBase->set(classifier::prediction::data, a);

    /* Calculate additive function values */
    for ( size_t m = 0; m < M; m++ )
    {
        for (size_t j = 0; j < nc; j++)
        {
            HomogenNTPtr predTable(new HomogenNT(pred.get() + j * n, 1, n));
            classifier::prediction::ResultPtr predictionRes(new classifier::prediction::Result());
            predictionRes->set(classifier::prediction::prediction, predTable);
            DAAL_CHECK_STATUS(s, learnerPredict->setResult(predictionRes));
            weak_learner::ModelPtr learnerModel = boostModel->getWeakLearnerModel(m * nc + j);
            learnerPredict->inputBase->set(classifier::prediction::model, learnerModel);
            DAAL_CHECK_STATUS(s, learnerPredict->computeNoThrow());
        }
        UpdateF<algorithmFPType, cpu>(dim, n, nc, pred.get(), F.get());
    }

    /* Calculate classes labels for input data */
    WriteOnlyColumns<int, cpu> rCols(*r, 0, 0, n);
    DAAL_CHECK_BLOCK_STATUS(rCols);
    int *cl = rCols.get();
    DAAL_ASSERT(cl);

    for ( size_t i = 0; i < n; i++ )
    {
        int idx = 0;
        algorithmFPType fmax = F[i * nc];
        for ( int j = 1; j < nc; j++ )
        {
            if ( F[i * nc + j] > fmax )
            {
                idx = j;
                fmax = F[i * nc + j];
            }
        }
        cl[i] = idx;
    }
    return s;
}

} // namepsace internal
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
