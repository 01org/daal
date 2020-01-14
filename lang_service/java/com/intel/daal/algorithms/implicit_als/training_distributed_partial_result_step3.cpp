/* file: training_distributed_partial_result_step3.cpp */
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

#include <jni.h>

#include "daal.h"

#include "com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep3.h"

#include "implicit_als_training_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::implicit_als;
using namespace daal::algorithms::implicit_als::training;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep3
 * Method:    cNewDistributedPartialResultStep3
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedPartialResultStep3_cNewDistributedPartialResultStep3(
    JNIEnv * env, jobject thisObj)
{
    return jniArgument<DistributedPartialResultStep3>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep3
 * Method:    cGetDataCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedPartialResultStep3_cGetDataCollection(
    JNIEnv * env, jobject thisObj, jlong partialResultAddr, jint id)
{
    return jniArgument<implicit_als::training::DistributedPartialResultStep3>::get<DistributedPartialResultStep3Id, KeyValueDataCollection>(
        partialResultAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedPartialResultStep3
 * Method:    cSetDataCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedPartialResultStep3_cSetDataCollection(
    JNIEnv * env, jobject thisObj, jlong partialResultAddr, jint id, jlong collectionAddr)
{
    jniArgument<implicit_als::training::DistributedPartialResultStep3>::set<DistributedPartialResultStep3Id, KeyValueDataCollection>(
        partialResultAddr, id, collectionAddr);
}
