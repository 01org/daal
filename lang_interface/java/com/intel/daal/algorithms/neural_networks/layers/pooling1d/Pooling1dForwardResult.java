/* file: Pooling1dForwardResult.java */
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
 * @ingroup pooling1d_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling1d;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__POOLING1DFORWARDRESULT"></a>
 * @brief Class that provides methods to access the result obtained with the compute() method of the forward one-dimensional pooling layer
 */
public class Pooling1dForwardResult extends com.intel.daal.algorithms.neural_networks.layers.ForwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
    * Constructs the forward one-dimensional pooling layer
    * @param context   Context to manage the forward one-dimensional pooling layer
    */
    public Pooling1dForwardResult(DaalContext context) {
        super(context);
    }

    public Pooling1dForwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
