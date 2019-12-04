/* file: parameter.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include <jni.h>/* Header for class com_intel_daal_algorithms_pivoted_qr_Offline */
#include "pivoted_qr_types.i"

#include "daal.h"
#include "com_intel_daal_algorithms_pivoted_qr_Parameter.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_pivoted_qr_Parameter
 * Method:    cSetPermutedColumns
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Parameter_cSetPermutedColumns(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                jlong permutedColumnsAddr)
{
    SerializationIfacePtr * ntShPtr                     = (SerializationIfacePtr *)permutedColumnsAddr;
    (*(pivoted_qr::Parameter *)parAddr).permutedColumns = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_pivoted_qr_Parameter
 * Method:    cGetPermutedColumns
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pivoted_1qr_Parameter_cGetPermutedColumns(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    pivoted_qr::Parameter * parameter = (pivoted_qr::Parameter *)parAddr;
    NumericTablePtr * ntShPtr         = new NumericTablePtr();
    *ntShPtr                          = parameter->permutedColumns;
    return (jlong)ntShPtr;
}
