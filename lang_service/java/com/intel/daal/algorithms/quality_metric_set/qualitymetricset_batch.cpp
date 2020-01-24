/* file: qualitymetricset_batch.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
#include "daal.h"
#include "com_intel_daal_algorithms_quality_metric_set_QualityMetricSetBatch.h"

/*
 * Class:     com_intel_daal_algorithms_QualityMetricSetBatch
 * Method:    cCompute
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_quality_1metric_1set_QualityMetricSetBatch_cCompute
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    (*(daal::algorithms::quality_metric_set::Batch *)algAddr).compute();
    if((*(daal::algorithms::quality_metric_set::Batch *)algAddr).getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), (*(daal::algorithms::quality_metric_set::Batch *)
                                                              algAddr).getErrors()->getDescription());
    }

}

/*
 * Class:     com_intel_daal_algorithms_QualityMetricSetBatch
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_quality_1metric_1set_QualityMetricSetBatch_cDispose
  (JNIEnv *, jobject, jlong algAddr)
{
    delete(daal::algorithms::quality_metric_set::Batch *)algAddr;
}
