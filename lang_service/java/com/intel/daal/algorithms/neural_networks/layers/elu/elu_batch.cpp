/* file: elu_batch.cpp */
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
#include "com_intel_daal_algorithms_neural_networks_layers_elu_EluBatch.h"

#include "daal.h"
#include "lang_service/java/com/intel/daal/include/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_elu_EluBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_elu_EluBatch_cInit(JNIEnv * env, jobject thisObj, jint prec,
                                                                                                  jint method)
{
    return jniBatchLayer<elu::Method, elu::Batch, elu::defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_elu_EluBatch
 * Method:    cGetForwardLayer
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_elu_EluBatch_cGetForwardLayer(JNIEnv * env, jobject thisObj,
                                                                                                             jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<elu::Method, elu::Batch, elu::defaultDense>::getForwardLayer(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_elu_EluBatch
 * Method:    cGetBackwardLayer
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_elu_EluBatch_cGetBackwardLayer(JNIEnv * env, jobject thisObj,
                                                                                                              jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<elu::Method, elu::Batch, elu::defaultDense>::getBackwardLayer(prec, method, algAddr);
}
