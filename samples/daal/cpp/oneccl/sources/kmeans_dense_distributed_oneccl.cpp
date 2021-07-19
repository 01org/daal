/* file: kmeans_dense_distributed_oneccl.cpp */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
!  Content:
!    C++ sample of K-Means clustering in the distributed processing mode
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-KMEANS_DENSE_DISTRIBUTED"></a>
 * \example kmeans_dense_distributed_oneccl.cpp
 */

#include "daal_sycl.h"
#include "service_sycl.h"
#include "oneapi/ccl.hpp"
#include "mpi.h"
#include "stdio.h"
#include <memory>

using namespace std;
using namespace daal;
using namespace daal::algorithms;

//typedef std::vector<char> ByteBuffer;
typedef float algorithmFPType; /* Algorithm floating-point type */

/* K-Means algorithm parameters */
const size_t nClusters   = 20;
const size_t nIterations = 5;
const size_t nProcs      = 4;

/* Input data set parameters */
const string dataFileNames[4] = { "./data/kmeans_dense.csv", "./data/kmeans_dense.csv", "./data/kmeans_dense.csv", "./data/kmeans_dense.csv" };

#define ccl_root 0

int getLocalRank(ccl::communicator & comm, int size, int rank)
{
    /* Obtain local rank among nodes sharing the same host name */
    char zero = static_cast<char>(0);
    std::vector<char> name(MPI_MAX_PROCESSOR_NAME + 1, zero);
    int resultlen = 0;
    MPI_Get_processor_name(name.data(), &resultlen);
    std::string str(name.begin(), name.end());
    std::vector<char> allNames((MPI_MAX_PROCESSOR_NAME + 1) * size, zero);
    std::vector<size_t> aReceiveCount(size, MPI_MAX_PROCESSOR_NAME + 1);
    ccl::allgatherv((int8_t *)name.data(), name.size(), (int8_t *)allNames.data(), aReceiveCount, comm).wait();
    int localRank = 0;
    for (int i = 0; i < rank; i++)
    {
        auto nameBegin = allNames.begin() + i * (MPI_MAX_PROCESSOR_NAME + 1);
        std::string nbrName(nameBegin, nameBegin + (MPI_MAX_PROCESSOR_NAME + 1));
        if (nbrName == str) localRank++;
    }
    return localRank;
}

NumericTablePtr loadData(int rankId)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(dataFileNames[rankId], DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();
    return dataSource.getNumericTable();
}

NumericTablePtr init(int rankId, const NumericTablePtr & pData, ccl::communicator & comm);
NumericTablePtr compute(int rankId, const NumericTablePtr & pData, const NumericTablePtr & initialCentroids, ccl::communicator & comm);

int main(int argc, char * argv[])
{
    /* Initialize oneCCL */
    ccl::init();

    MPI_Init(NULL, NULL);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0)
    {
        kvs       = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto comm = ccl::create_communicator(size, rank, kvs);

    /* Create GPU device from local rank and set execution context */
    auto gpus       = get_gpus();

    if (gpus.size() > 0) {
        auto local_rank = getLocalRank(comm, size, rank);
        auto rank_gpu   = gpus[local_rank % gpus.size()];
        cl::sycl::queue queue(rank_gpu);
        daal::services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);
    } else {
        cl::sycl::cpu_selector cpu;
        cl::sycl::queue queue(cpu);
        daal::services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);
    }

    /* Start data processing */
    NumericTablePtr pData     = loadData(rank);
    NumericTablePtr centroids = init(rank, pData, comm);

    for (size_t it = 0; it < nIterations; it++) centroids = compute(rank, pData, centroids, comm);

    /* Print the clusterization results */
    if (rank == ccl_root) printNumericTable(centroids, "First 10 dimensions of centroids:", 20, 10);

    MPI_Finalize();
    return 0;
}

NumericTablePtr init(int rankId, const NumericTablePtr & pData, ccl::communicator & comm)
{
    const bool isRoot = (rankId == ccl_root);

    const size_t nVectorsInBlock = pData->getNumberOfRows();

    /* Create an algorithm to compute k-means on local nodes */
    kmeans::init::Distributed<step1Local, algorithmFPType, kmeans::init::randomDense> localInit(nClusters, nProcs * nVectorsInBlock,
                                                                                                rankId * nVectorsInBlock);

    /* Set the input data set to the algorithm */
    localInit.input.set(kmeans::init::data, pData);

    /* Compute k-means */
    localInit.compute();

    /* Serialize partial results required by step 2 */
    InputDataArchive dataArch;
    localInit.getPartialResult()->serialize(dataArch);
    const uint64_t perNodeArchLength = (size_t)dataArch.getSizeOfArchive();

    std::vector<uint64_t> aPerNodeArchLength(comm.size());
    std::vector<size_t> aReceiveCount(comm.size(), 1);
    /* Transfer archive length to the step 2 on the root node */
    ccl::allgatherv(&perNodeArchLength, 1, aPerNodeArchLength.data(), aReceiveCount, comm).wait();

    ByteBuffer serializedData;
    /* Calculate total archive length */
    int totalArchLength = 0;
    int displs[nProcs];
    for (size_t i = 0; i < nProcs; ++i)
    {
        totalArchLength += aPerNodeArchLength[i];
    }
    aReceiveCount[ccl_root] = totalArchLength;

    serializedData.resize(totalArchLength);

    ByteBuffer nodeResults(perNodeArchLength);
    dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    ccl::allgatherv((int8_t *)&nodeResults[0], perNodeArchLength, (int8_t *)&serializedData[0], aPerNodeArchLength, comm).wait();
    if (isRoot)
    {
        /* Create an algorithm to compute k-means on the master node */
        kmeans::init::Distributed<step2Master, algorithmFPType, kmeans::init::randomDense> masterInit(nClusters);
        for (size_t i = 0, shift = 0; i < nProcs; shift += aPerNodeArchLength[i], ++i)
        {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(&serializedData[shift], aPerNodeArchLength[i]);

            kmeans::init::PartialResultPtr dataForStep2FromStep1(new kmeans::init::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm */
            masterInit.input.add(kmeans::init::partialResults, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute k-means on the master node */
        masterInit.compute();
        masterInit.finalizeCompute();
        return masterInit.getResult()->get(kmeans::init::centroids);
    }
    return NumericTablePtr();
}

NumericTablePtr compute(int rankId, const NumericTablePtr & pData, const NumericTablePtr & initialCentroids, ccl::communicator & comm)
{
    const bool isRoot            = (rankId == ccl_root);
    uint64_t CentroidsArchLength = 0;
    InputDataArchive inputArch;
    if (isRoot)
    {
        /*Retrieve the algorithm results and serialize them */
        initialCentroids->serialize(inputArch);
        CentroidsArchLength = inputArch.getSizeOfArchive();
    }

    /* Get partial results from the root node */
    ccl::broadcast(&CentroidsArchLength, 1, ccl_root, comm).wait();

    ByteBuffer nodeCentroids(CentroidsArchLength);
    if (isRoot) inputArch.copyArchiveToArray(&nodeCentroids[0], CentroidsArchLength);

    ccl::broadcast((int8_t *)&nodeCentroids[0], CentroidsArchLength, ccl_root, comm).wait();

    /* Deserialize centroids data */
    OutputDataArchive outArch(nodeCentroids.size() ? &nodeCentroids[0] : NULL, CentroidsArchLength);

    NumericTablePtr centroids(new HomogenNumericTable<>());

    centroids->deserialize(outArch);

    /* Create an algorithm to compute k-means on local nodes */
    kmeans::Distributed<step1Local> localAlgorithm(nClusters);

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(kmeans::data, pData);
    localAlgorithm.input.set(kmeans::inputCentroids, centroids);

    /* Compute k-means */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();
    ByteBuffer serializedData;

    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    serializedData.resize(perNodeArchLength * nProcs);

    ByteBuffer nodeResults(perNodeArchLength);
    dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);
    std::vector<size_t> aReceiveCount(comm.size(), perNodeArchLength);
    /* Transfer partial results to step 2 on the root node */
    ccl::allgatherv((int8_t *)&nodeResults[0], perNodeArchLength, (int8_t *)&serializedData[0], aReceiveCount, comm).wait();

    if (isRoot)
    {
        /* Create an algorithm to compute k-means on the master node */
        kmeans::Distributed<step2Master> masterAlgorithm(nClusters);

        for (size_t i = 0; i < nProcs; i++)
        {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(&serializedData[perNodeArchLength * i], perNodeArchLength);

            kmeans::PartialResultPtr dataForStep2FromStep1(new kmeans::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm */
            masterAlgorithm.input.add(kmeans::partialResults, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute k-means on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        return masterAlgorithm.getResult()->get(kmeans::centroids);
    }
    return NumericTablePtr();
}
