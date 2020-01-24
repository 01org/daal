/* file: parameter.cpp */
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
#include "com_intel_daal_algorithms_pca_transform_TransformParameter.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformParameter
 * Method:    cGetLowerBound
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformParameter_cGetNumberOfComponents
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return (jlong)(reinterpret_cast<pca::transform::Parameter *>(parAddr)->nComponents);
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformParameter
 * Method:    cSetLowerBound
 * Signature:(JJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformParameter_cSetNumberOfComponents
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong nComponents)
{
    reinterpret_cast<pca::transform::Parameter *>(parAddr)->nComponents = nComponents;
}
