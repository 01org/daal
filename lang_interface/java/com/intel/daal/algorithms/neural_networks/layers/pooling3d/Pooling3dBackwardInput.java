/* file: Pooling3dBackwardInput.java */
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

/**
 * @defgroup pooling3d_backward Backward Three-dimensional Pooling Layer
 * @brief Contains classes for backward three-dimensional (3D) pooling layer
 * @ingroup pooling3d
 * @{
 */
/**
 * @brief Contains classes of the three-dimensional (3D) pooling layers
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling3d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DBACKWARDINPUT"></a>
 * @brief Input object for the backward pooling layer
 */
public class Pooling3dBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Pooling3dBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
