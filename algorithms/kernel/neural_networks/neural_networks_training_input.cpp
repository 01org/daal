/* file: neural_networks_training_input.cpp */
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

#include "neural_networks_training_input.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{
Input::Input(size_t nElements) : daal::algorithms::Input(nElements)
{
    set(groundTruthCollection, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
}

Input::Input(const Input& other) : daal::algorithms::Input(other){}

TensorPtr Input::get(InputId id) const
{
    if (id == groundTruth)
    {
        KeyValueDataCollectionPtr collection = get(groundTruthCollection);
        if (!collection) { return TensorPtr(); }
        if (collection->size() == 0) { return TensorPtr(); }
        return Tensor::cast(collection->getValueByIndex(0));
    }
    else
    {
        return Tensor::cast(Argument::get(id));
    }
}

KeyValueDataCollectionPtr Input::get(InputCollectionId id) const
{
    return KeyValueDataCollection::cast(Argument::get(id));
}

TensorPtr Input::get(InputCollectionId id, size_t key) const
{
    KeyValueDataCollectionPtr collection = get(id);
    if (!collection) { return TensorPtr(); }
    return Tensor::cast((*collection)[key]);
}

void Input::set(InputId id, const TensorPtr &value)
{
    if (id == groundTruth)
    {
        add(groundTruthCollection, 0, value);
    }
    else
    {
        Argument::set(id, value);
    }
}

void Input::set(InputCollectionId id, const KeyValueDataCollectionPtr &value)
{
    Argument::set(id, value);
}

void Input::add(InputCollectionId id, size_t key, const TensorPtr &value)
{
    KeyValueDataCollectionPtr collection = get(id);
    if (!collection) { return; }
    (*collection)[key] = value;
}

Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(par);
    TensorPtr dataTensor = get(data);
    Status s;
    DAAL_CHECK_STATUS(s, checkTensor(dataTensor.get(), dataStr()))

    size_t nSamples = dataTensor->getDimensionSize(0);
    DAAL_CHECK_EX(param->optimizationSolver, ErrorIncorrectParameter, ParameterName, optimizationSolverStr());
    DAAL_CHECK_EX(nSamples >= param->optimizationSolver->getParameter()->batchSize, ErrorIncorrectParameter, ParameterName, batchSizeStr());

    DAAL_CHECK_STATUS(s, checkImpl(par, method));
    return s;
}

Status Input::checkImpl(const daal::algorithms::Parameter *par, int method) const
{
    KeyValueDataCollectionPtr groundTruthTensorCollection = get(groundTruthCollection);

    DAAL_CHECK(groundTruthTensorCollection, ErrorNullInputDataCollection)
    DAAL_CHECK(groundTruthTensorCollection->size(), ErrorIncorrectNumberOfElementsInInputCollection)

    size_t nLastLayers = groundTruthTensorCollection->size();
    TensorPtr dataTensor = get(data);
    size_t nSamples = dataTensor->getDimensionSize(0);
    Status s;
    for (size_t i = 0; i < nLastLayers; i++)
    {
        size_t layerId = groundTruthTensorCollection->getKeyByIndex((int)i);
        TensorPtr groundTruthTensor = get(groundTruthCollection, layerId);
        if (!groundTruthTensor)
        {
            SharedPtr<Error> error = Error::create(ErrorNullTensor, ArgumentName, groundTruthLabelsStr());
            error->addIntDetail(ElementInCollection, (int)layerId);
            return s.add(error);
        }
        Collection<size_t> expectedDims = groundTruthTensor->getDimensions();
        expectedDims[0] = nSamples;
        DAAL_CHECK_STATUS(s, checkTensor(groundTruthTensor.get(), groundTruthLabelsStr(), &expectedDims))
    }
    return s;
}

}
}
}
}
