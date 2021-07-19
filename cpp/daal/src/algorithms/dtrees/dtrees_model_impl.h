/* file: dtrees_model_impl.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

/*
//++
//  Implementation of the class defining the decision forest model
//--
*/

#ifndef __DTREES_MODEL_IMPL__
#define __DTREES_MODEL_IMPL__

#include "services/env_detect.h"
#include "services/daal_shared_ptr.h"
#include "src/services/service_defines.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/aos_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/services/service_utils.h"

typedef size_t ClassIndexType;
typedef double ModelFPType;
typedef ModelFPType ClassificationFPType; //type of features stored in classification model
typedef ModelFPType RegressionFPType;     //type of features and regression response stored in the model

//#define DEBUG_CHECK_IMPURITY

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace internal
{
struct DecisionTreeNode
{
    int featureIndex;                   //split: index of the feature, leaf: -1
    ClassIndexType leftIndexOrClass;    //split: left node index, classification leaf: class index
    ModelFPType featureValueOrResponse; //split: feature value, regression tree leaf: response
    DAAL_FORCEINLINE bool isSplit() const { return featureIndex != -1; }
    ModelFPType featureValue() const { return featureValueOrResponse; }
};

class DecisionTreeTable : public data_management::AOSNumericTable
{
public:
    DecisionTreeTable(size_t rowCount = 0) : data_management::AOSNumericTable(sizeof(DecisionTreeNode), 3, rowCount)
    {
        setFeature<int>(0, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, featureIndex));
        setFeature<ClassIndexType>(1, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, leftIndexOrClass));
        setFeature<ModelFPType>(2, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, featureValueOrResponse));
        allocateDataMemory();
    }
};
typedef services::SharedPtr<DecisionTreeTable> DecisionTreeTablePtr;
typedef services::SharedPtr<const DecisionTreeTable> DecisionTreeTableConstPtr;

template <typename TResponse, typename THistogramm>
class ClassifierResponse
{
public:
    TResponse value; //majority votes response

#ifdef KEEP_CLASSES_PROBABILITIIES
    THistogramm * hist; //histogramm
    size_t size;        //number of classes in histogramm
    ClassifierResponse() : hist(nullptr), value(0) {}
    ~ClassifierResponse()
    {
        if (hist) daal::services::daal_free(hist);
        hist = nullptr;
    }
#else
    ClassifierResponse() : value(0) {}
#endif
    ClassifierResponse(const ClassifierResponse &) = delete;
    ClassifierResponse & operator=(const ClassifierResponse & o) = delete;
};

struct TreeNodeBase
{
    virtual ~TreeNodeBase() {}
    virtual bool isSplit() const       = 0;
    virtual size_t numChildren() const = 0;

    TreeNodeBase() : count(0), impurity(0) {}
    size_t count;
    double impurity;
};

template <typename algorithmFPType>
struct TreeNodeSplit : public TreeNodeBase
{
    typedef algorithmFPType FeatureType;
    FeatureType featureValue;
    TreeNodeBase * kid[2];
    int featureIdx;
    bool featureUnordered;

    TreeNodeSplit() { kid[0] = kid[1] = nullptr; }
    const TreeNodeBase * left() const { return kid[0]; }
    const TreeNodeBase * right() const { return kid[1]; }
    TreeNodeBase * left() { return kid[0]; }
    TreeNodeBase * right() { return kid[1]; }

    void set(int featIdx, algorithmFPType featValue, bool bUnordered)
    {
        DAAL_ASSERT(featIdx >= 0);
        featureValue     = featValue;
        featureIdx       = featIdx;
        featureUnordered = bUnordered;
    }
    virtual bool isSplit() const DAAL_C11_OVERRIDE { return true; }
    virtual size_t numChildren() const DAAL_C11_OVERRIDE
    {
        return (kid[0] ? kid[0]->numChildren() + 1 : 0) + (kid[1] ? kid[1]->numChildren() + 1 : 0);
    }
};

template <typename TResponseType>
struct TreeNodeLeaf : public TreeNodeBase
{
    TResponseType response;
    double * hist;

    TreeNodeLeaf() {}

    // nCLasses = 0 for regression
    TreeNodeLeaf(double * memoryForHist) : hist(memoryForHist) {}

    virtual ~TreeNodeLeaf() {}

    virtual bool isSplit() const DAAL_C11_OVERRIDE { return false; }
    virtual size_t numChildren() const DAAL_C11_OVERRIDE { return 0; }
};

template <typename algorithmFPType>
struct TreeNodeRegression
{
    typedef TreeNodeBase Base;
    typedef TreeNodeSplit<algorithmFPType> Split;
    typedef TreeNodeLeaf<algorithmFPType> Leaf;

    static Leaf * castLeaf(Base * n) { return static_cast<Leaf *>(n); }
    static const Leaf * castLeaf(const Base * n) { return static_cast<const Leaf *>(n); }
    static Split * castSplit(Base * n) { return static_cast<Split *>(n); }
    static const Split * castSplit(const Base * n) { return static_cast<const Split *>(n); }
};

template <typename algorithmFPType>
struct TreeNodeClassification
{
    typedef TreeNodeBase Base;
    typedef TreeNodeSplit<algorithmFPType> Split;
    typedef TreeNodeLeaf<ClassifierResponse<ClassIndexType, size_t> > Leaf;

    static Leaf * castLeaf(Base * n) { return static_cast<Leaf *>(n); }
    static const Leaf * castLeaf(const Base * n) { return static_cast<const Leaf *>(n); }
    static Split * castSplit(Base * n) { return static_cast<Split *>(n); }
    static const Split * castSplit(const Base * n) { return static_cast<const Split *>(n); }
};

template <typename NodeType>
class HeapMemoryAllocator
{
public:
    HeapMemoryAllocator(size_t dummy) {}
    typename NodeType::Leaf * allocLeaf();
    typename NodeType::Split * allocSplit();
    void free(typename NodeType::Base * n);
    void reset() {}
    bool deleteRecursive() const { return true; }
};
template <typename NodeType>
typename NodeType::Leaf * HeapMemoryAllocator<NodeType>::allocLeaf()
{
    return new typename NodeType::Leaf();
}

template <typename NodeType>
typename NodeType::Split * HeapMemoryAllocator<NodeType>::allocSplit()
{
    return new typename NodeType::Split();
}

template <typename NodeType>
void HeapMemoryAllocator<NodeType>::free(typename NodeType::Base * n)
{
    delete n;
}

class DAAL_EXPORT MemoryManager
{
public:
    MemoryManager(size_t chunkSize) : _chunkSize(chunkSize), _posInChunk(0), _iCurChunk(-1) {}
    ~MemoryManager() { destroy(); }

    void * alloc(size_t nBytes);
    //free all allocated memory without destroying of internal storage
    void reset();
    //physically destroy internal storage
    void destroy();

private:
    services::Collection<byte *> _aChunk;
    const size_t _chunkSize; //size of a chunk to be allocated
    size_t _posInChunk;      //index of the first free byte in the current chunk
    int _iCurChunk;          //index of the current chunk to allocate from
};

template <typename NodeType>
class ChunkAllocator
{
public:
    ChunkAllocator(size_t nNodesInChunk, size_t nClasses = 0)
        : _man(nNodesInChunk * (sizeof(typename NodeType::Leaf) + sizeof(typename NodeType::Split)))
    {}
    typename NodeType::Leaf * allocLeaf(size_t nClasses);
    typename NodeType::Leaf * allocLeaf();
    typename NodeType::Split * allocSplit();
    void free(typename NodeType::Base * n);
    void reset() { _man.reset(); }
    bool deleteRecursive() const { return false; }

private:
    MemoryManager _man;
};
template <typename NodeType>
typename NodeType::Leaf * ChunkAllocator<NodeType>::allocLeaf(size_t nClasses)
{
    void * memory = _man.alloc(sizeof(typename NodeType::Leaf) + nClasses * sizeof(double));
    return new (memory) typename NodeType::Leaf(reinterpret_cast<double *>(static_cast<typename NodeType::Leaf *>(memory) + 1));
}

template <typename NodeType>
typename NodeType::Leaf * ChunkAllocator<NodeType>::allocLeaf()
{
    return new (_man.alloc(sizeof(typename NodeType::Leaf))) typename NodeType::Leaf();
}

template <typename NodeType>
typename NodeType::Split * ChunkAllocator<NodeType>::allocSplit()
{
    return new (_man.alloc(sizeof(typename NodeType::Split))) typename NodeType::Split();
}

template <typename NodeType>
void ChunkAllocator<NodeType>::free(typename NodeType::Base * n)
{}

template <typename NodeType, typename Allocator>
void deleteNode(typename NodeType::Base * n, Allocator & a)
{
    if (n->isSplit())
    {
        typename NodeType::Split * s = static_cast<typename NodeType::Split *>(n);
        if (s->left()) deleteNode<NodeType, Allocator>(s->left(), a);
        if (s->right()) deleteNode<NodeType, Allocator>(s->right(), a);
    }
    a.free(n);
}

class DAAL_EXPORT Tree : public Base
{
public:
    Tree() {}
    virtual ~Tree();
};
typedef services::SharedPtr<Tree> TreePtr;

template <typename TNodeType, typename TAllocator = ChunkAllocator<TNodeType> >
class TreeImpl : public Tree
{
public:
    typedef TAllocator Allocator;
    typedef TNodeType NodeType;
    typedef TreeImpl<TNodeType, TAllocator> ThisType;

    TreeImpl(typename NodeType::Base * t, bool bHasUnorderedFeatureSplits)
        : _allocator(_cNumNodesHint), _top(t), _hasUnorderedFeatureSplits(bHasUnorderedFeatureSplits)
    {}
    TreeImpl() : _allocator(_cNumNodesHint), _top(nullptr), _hasUnorderedFeatureSplits(false) {}
    ~TreeImpl() { destroy(); }
    void destroy();
    void reset(typename NodeType::Base * t, bool bHasUnorderedFeatureSplits)
    {
        destroy();
        _top                       = t;
        _hasUnorderedFeatureSplits = bHasUnorderedFeatureSplits;
    }
    const typename NodeType::Base * top() const { return _top; }
    Allocator & allocator() { return _allocator; }
    bool hasUnorderedFeatureSplits() const { return _hasUnorderedFeatureSplits; }
    size_t getNumberOfNodes() const { return top() ? top()->numChildren() + 1 : 0; }
    void convertToTable(DecisionTreeTable * treeTable, data_management::HomogenNumericTable<double> * impurities,
                        data_management::HomogenNumericTable<int> * nNodeSamples, data_management::HomogenNumericTable<double> * prob,
                        size_t nClasses) const;

private:
    static const size_t _cNumNodesHint = 512; //number of nodes as a hint for allocator to grow by
    Allocator _allocator;
    typename NodeType::Base * _top;
    bool _hasUnorderedFeatureSplits;
};
template <typename Allocator = ChunkAllocator<TreeNodeRegression<RegressionFPType> > >
using TreeImpRegression = TreeImpl<TreeNodeRegression<RegressionFPType>, Allocator>;

template <typename Allocator = ChunkAllocator<TreeNodeClassification<ClassificationFPType> > >
using TreeImpClassification = TreeImpl<TreeNodeClassification<ClassificationFPType>, Allocator>;

#define __NODE_RESERVED_ID -2
#define __NODE_FREE_ID     -3
#define __N_CHILDS         2

template <typename ModelImplType, typename ModelTypePtr>
ModelImplType & getModelRef(ModelTypePtr & modelPtr)
{
    ModelImplType * modelImplPtr = static_cast<ModelImplType *>(modelPtr.get());
    DAAL_ASSERT(modelImplPtr);
    return *modelImplPtr;
}

services::Status createTreeInternal(data_management::DataCollectionPtr & serializationData, size_t nNodes, size_t & resId);

void setNode(DecisionTreeNode & node, int featureIndex, size_t classLabel);

void setNode(DecisionTreeNode & node, int featureIndex, double response);

services::Status addSplitNodeInternal(data_management::DataCollectionPtr & serializationData, size_t treeId, size_t parentId, size_t position,
                                      size_t featureIndex, double featureValue, size_t & res);

void setProbabilities(const size_t treeId, const size_t nodeId, const size_t response, const data_management::DataCollectionPtr probTbl,
                      const double * const prob);

template <typename ClassOrResponseType>
static services::Status addLeafNodeInternal(const data_management::DataCollectionPtr & serializationData, const size_t treeId, const size_t parentId,
                                            const size_t position, ClassOrResponseType response, size_t & res,
                                            const data_management::DataCollectionPtr probTbl = data_management::DataCollectionPtr(),
                                            const double * const prob = nullptr, const size_t nClasses = 0)
{
    const size_t noParent = static_cast<size_t>(-1);
    if (prob != nullptr)
    {
        response = services::internal::getMaxElementIndex<double, sse2>(prob, nClasses);
    }

    services::Status s;
    if ((treeId > (*(serializationData)).size()) || (position != 0 && position != 1))
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }
    const DecisionTreeTable * const pTreeTable = static_cast<DecisionTreeTable *>((*(serializationData))[treeId].get());
    if (!pTreeTable) return services::Status(services::ErrorID::ErrorNullPtr);
    const size_t nRows             = pTreeTable->getNumberOfRows();
    DecisionTreeNode * const aNode = (DecisionTreeNode *)pTreeTable->getArray();
    size_t nodeId                  = 0;
    if (parentId == noParent)
    {
        setNode(aNode[0], -1, response);
        setProbabilities(treeId, 0, response, probTbl, prob);
        nodeId = 0;
    }
    else if (aNode[parentId].featureIndex < 0)
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }
    else
    {
        /*if not leaf, and parent has child already*/
        if ((aNode[parentId].leftIndexOrClass > 0) && (position == 1))
        {
            const size_t reservedId = aNode[parentId].leftIndexOrClass + 1;
            nodeId                  = reservedId;
            if (aNode[reservedId].featureIndex == __NODE_RESERVED_ID)
            {
                setNode(aNode[nodeId], -1, response);
                setProbabilities(treeId, nodeId, response, probTbl, prob);
            }
        }
        else if ((aNode[parentId].leftIndexOrClass > 0) && (position == 0))
        {
            const size_t reservedId = aNode[parentId].leftIndexOrClass;
            nodeId                  = reservedId;
            if (aNode[reservedId].featureIndex == __NODE_RESERVED_ID)
            {
                setNode(aNode[nodeId], -1, response);
                setProbabilities(treeId, nodeId, response, probTbl, prob);
            }
        }
        else if ((aNode[parentId].leftIndexOrClass == 0) && (position == 0))
        {
            size_t i;
            for (i = parentId + 1; i < nRows; i++)
            {
                if (aNode[i].featureIndex == __NODE_FREE_ID)
                {
                    nodeId = i;
                    break;
                }
            }
            /* no space left */
            if (i == nRows)
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
            setNode(aNode[nodeId], -1, response);
            setProbabilities(treeId, nodeId, response, probTbl, prob);
            aNode[parentId].leftIndexOrClass = nodeId;
            if (((nodeId + 1) < nRows) && (aNode[nodeId + 1].featureIndex == __NODE_FREE_ID))
            {
                aNode[nodeId + 1].featureIndex = __NODE_RESERVED_ID;
            }
            else
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
        }
        else if ((aNode[parentId].leftIndexOrClass == 0) && (position == 1))
        {
            size_t leftEmptyId = 0;
            size_t i;
            for (i = parentId + 1; i < nRows; i++)
            {
                if (aNode[i].featureIndex == __NODE_FREE_ID)
                {
                    leftEmptyId = i;
                    break;
                }
            }
            /*if no free nodes leftBound is not initialized and no space left*/
            if (i == nRows)
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
            aNode[leftEmptyId].featureIndex  = __NODE_RESERVED_ID;
            aNode[parentId].leftIndexOrClass = leftEmptyId;
            nodeId                           = leftEmptyId + 1;
            if (nodeId < nRows)
            {
                setNode(aNode[nodeId], -1, response);
                setProbabilities(treeId, nodeId, response, probTbl, prob);
            }
            else
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
        }
    }
    res = nodeId;
    return s;
}

class DAAL_EXPORT ModelImpl
{
public:
    ModelImpl();
    virtual ~ModelImpl();

    size_t size() const { return _nTree.get(); }
    bool reserve(const size_t nTrees);
    bool resize(const size_t nTrees);
    void clear();

    const data_management::DataCollection * serializationData() const { return _serializationData.get(); }

    const DecisionTreeTable * at(const size_t i) const { return (const DecisionTreeTable *)(*_serializationData)[i].get(); }

    const double * getImpVals(size_t i) const
    {
        return _impurityTables ? ((const data_management::HomogenNumericTable<double> *)(*_impurityTables)[i].get())->getArray() : nullptr;
    }

    const int * getNodeSampleCount(size_t i) const
    {
        return _nNodeSampleTables ? ((const data_management::HomogenNumericTable<int> *)(*_nNodeSampleTables)[i].get())->getArray() : nullptr;
    }

    const double * getProbas(size_t i) const
    {
        return _probTbl ? ((const data_management::HomogenNumericTable<double> *)(*_probTbl)[i].get())->getArray() : nullptr;
    }

    size_t getNumClasses() const
    {
        if (_probTbl.get() == nullptr || _probTbl->size() == 0)
        {
            return 0;
        }
        return ((const data_management::HomogenNumericTable<double> *)(*_probTbl)[0].get())->getNumberOfRows();
    }

protected:
    void destroy();
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch, int daalVersion = INTEL_DAAL_VERSION)
    {
        arch->setSharedPtrObj(_serializationData);

        if ((daalVersion >= COMPUTE_DAAL_VERSION(2019, 0, 0)))
        {
            arch->setSharedPtrObj(_impurityTables);
            arch->setSharedPtrObj(_nNodeSampleTables);
        }
        if ((daalVersion > COMPUTE_DAAL_VERSION(2020, 0, 0)))
        {
            arch->setSharedPtrObj(_probTbl);
        }

        if (onDeserialize) _nTree.set(_serializationData->size());

        return services::Status();
    }

protected:
    data_management::DataCollectionPtr _serializationData; //collection of DecisionTreeTables
    daal::services::Atomic<size_t> _nTree;

    data_management::DataCollectionPtr _impurityTables;
    data_management::DataCollectionPtr _nNodeSampleTables;
    data_management::DataCollectionPtr _probTbl;
};

template <typename NodeType, typename Allocator>
void TreeImpl<NodeType, Allocator>::destroy()
{
    if (_top)
    {
        if (allocator().deleteRecursive()) deleteNode<NodeType, Allocator>(_top, allocator());
        _top = nullptr;
        allocator().reset();
    }
}

} // namespace internal
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
