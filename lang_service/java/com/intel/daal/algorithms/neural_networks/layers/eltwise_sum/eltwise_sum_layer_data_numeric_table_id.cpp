/* file: eltwise_sum_layer_data_numeric_table_id.cpp */
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
#include "com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumLayerDataNumericTableId.h"

#include "daal.h"
#include "lang_service/java/com/intel/daal/include/common_helpers.h"

using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumLayerDataNumericTableId
 * Method:    cGetAuxNumberOfCoefficientsId
 * Signature: ()I
 */
JNIEXPORT jint JNICALL
    Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumLayerDataNumericTableId_cGetAuxNumberOfCoefficientsId(JNIEnv * env,
                                                                                                                                        jclass)
{
    return (jint)eltwise_sum::auxNumberOfCoefficients;
}
