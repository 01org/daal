/* file: DistributedPartialResultStep4Id.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup gbt_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP4ID"></a>
 * @brief Available identifiers of partial results of the model-based training in the fourth step
 *        of the distributed processing mode
 */
public final class DistributedPartialResultStep4Id {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep4Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int totalHistogramsValue = 0;
    private static final int bestSplitsValue = 1;

    public static final DistributedPartialResultStep4Id totalHistograms = new DistributedPartialResultStep4Id(totalHistogramsValue);
        /*!<  */
    public static final DistributedPartialResultStep4Id bestSplits = new DistributedPartialResultStep4Id(bestSplitsValue);
        /*!<  */
}
/** @} */
