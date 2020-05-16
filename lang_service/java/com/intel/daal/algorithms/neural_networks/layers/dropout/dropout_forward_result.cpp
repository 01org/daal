/* file: dropout_forward_result.cpp */
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
#include "com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutForwardResult.h"

#include "daal.h"

#include "lang_service/java/com/intel/daal/include/common_helpers.h"

#include "com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutLayerDataId.h"
#define auxRetainMaskId com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutLayerDataId_auxRetainMaskId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers::dropout;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutForwardResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_DropoutForwardResult_cNewResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<forward::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutForwardResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_DropoutForwardResult_cGetValue(JNIEnv * env, jobject thisObj,
                                                                                                                      jlong resAddr, jint id)
{
    if (id == auxRetainMaskId)
    {
        return jniArgument<forward::Result>::get<LayerDataId, Tensor>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_DropoutForwardResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_DropoutForwardResult_cSetValue(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong resAddr, jint id,
                                                                                                                     jlong ntAddr)
{
    if (id == auxRetainMaskId)
    {
        jniArgument<forward::Result>::set<LayerDataId, Tensor>(resAddr, id, ntAddr);
    }
}
