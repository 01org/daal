/* file: train_input.cpp */
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
#include "com_intel_daal_algorithms_decision_tree_regression_training_TrainingInput.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()

namespace dtrt = daal::algorithms::decision_tree::regression::training;

/*
* Class:     com_intel_daal_algorithms_decision_tree_regression_training_TrainingInput
* Method:    cSetInput
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_decision_1tree_regression_training_TrainingInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<dtrt::Input>::set<dtrt::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_regression_training_TrainingInput
* Method:    cGetInput
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_regression_training_TrainingInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<dtrt::Input>::get<dtrt::InputId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_regression_training_TrainingInput
* Method:    cGetNumberOfFeatures
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_regression_training_TrainingInput_cGetNumberOfFeatures
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
    return ((dtrt::Input*)inputAddr)->getNumberOfFeatures();
}

/*
* Class:     com_intel_daal_algorithms_decision_tree_regression_training_TrainingInput
* Method:    cGetNumberOfDependentVariables
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_decision_1tree_regression_training_TrainingInput_cGetNumberOfDependentVariables
(JNIEnv *env, jobject thisObj, jlong inputAddr)
{
    return ((dtrt::Input*)inputAddr)->getNumberOfDependentVariables();
}
