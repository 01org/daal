/* file: pca_dense_batch_distributed_oneccl.cpp */
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
!    C++ sample of principal component analysis (PCA) using the correlation
!    method in the distributed processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-PCA_DENSE_DISTRIBUTED"></a>
 * \example pca_dense_batch_distributed_oneccl.cpp
 */

#include "daal_sycl.h"
#include "mpi.h"
#include "oneapi/ccl.hpp"
#include "service_sycl.h"
#include "stdio.h"
#include <memory>

using namespace std;
using namespace daal;
using namespace daal::algorithms;

// typedef std::vector<char> ByteBuffer;
typedef float algorithmFPType; /* Algorithm floating-point type */

/* PCA algorithm parameters */
const size_t nProcs = 4;

/* Input data set parameters */
const string dataFileNames[4] = {
    "./data/pca_normalized_1.csv", "./data/pca_normalized_2.csv",
    "./data/pca_normalized_3.csv", "./data/pca_normalized_4.csv"};

#define ccl_root 0

NumericTablePtr loadData(int rankId) {
  /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data
   * from a .csv file */
  FileDataSource<CSVFeatureManager> dataSource(
      dataFileNames[rankId], DataSource::doAllocateNumericTable,
      DataSource::doDictionaryFromContext);

  /* Retrieve the data from the input file */
  dataSource.loadDataBlock();
  return dataSource.getNumericTable();
}

int main(int argc, char *argv[]) {
  /* Initialize oneCCL */
  ccl::init();

  MPI_Init(NULL, NULL);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type main_addr;
  if (rank == 0) {
    kvs = ccl::create_main_kvs();
    main_addr = kvs->get_address();
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
              MPI_COMM_WORLD);
  } else {
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    kvs = ccl::create_kvs(main_addr);
  }

  auto comm = ccl::create_communicator(size, rank, kvs);
  /* Create GPU device from local rank and set execution context if possible */
  set_gpu_by_rank_if_possible(comm, size, rank);

  /* Start data processing */
  NumericTablePtr pData = loadData(rank);

  const bool isRoot = (rank == ccl_root);

  pca::Distributed<step1Local> localAlgorithm;

  /* Set the input data set to the algorithm */
  localAlgorithm.input.set(pca::data, pData);

  /* Compute PCA decomposition */
  localAlgorithm.compute();

  /* Serialize partial results required by step 2 */
  InputDataArchive dataArch;
  localAlgorithm.getPartialResult()->serialize(dataArch);
  const uint64_t perNodeArchLength = (size_t)dataArch.getSizeOfArchive();

  std::vector<uint64_t> aPerNodeArchLength(comm.size());
  std::vector<size_t> aReceiveCount(comm.size(), 1);
  /* Transfer archive length to the step 2 on the root node */
  ccl::allgatherv(&perNodeArchLength, 1, aPerNodeArchLength.data(),
                  aReceiveCount, comm)
      .wait();

  ByteBuffer serializedData;
  /* Calculate total archive length */
  int totalArchLength = 0;
  int displs[nProcs];
  for (size_t i = 0; i < nProcs; ++i) {
    totalArchLength += aPerNodeArchLength[i];
  }
  aReceiveCount[ccl_root] = totalArchLength;

  serializedData.resize(totalArchLength);

  ByteBuffer nodeResults(perNodeArchLength);
  dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);

  /* Transfer partial results to step 2 on the root node */
  ccl::allgatherv((int8_t *)&nodeResults[0], perNodeArchLength,
                  (int8_t *)&serializedData[0], aPerNodeArchLength, comm)
      .wait();

  if (isRoot) {
    /* Create an algorithm for principal component analysis using the
     * correlation method on the master node */
    pca::Distributed<step2Master> masterAlgorithm;
    for (size_t i = 0, shift = 0; i < nProcs;
         shift += aPerNodeArchLength[i], ++i) {
      /* Deserialize partial results from step 1 */
      OutputDataArchive dataArch(&serializedData[shift], aPerNodeArchLength[i]);

      services::SharedPtr<pca::PartialResult<pca::correlationDense>>
          dataForStep2FromStep1(
              new pca::PartialResult<pca::correlationDense>());
      dataForStep2FromStep1->deserialize(dataArch);

      /* Set local partial results as input for the master-node algorithm */
      masterAlgorithm.input.add(pca::partialResults, dataForStep2FromStep1);
    }

    /* Merge and finalizeCompute PCA decomposition on the master node */
    masterAlgorithm.compute();
    masterAlgorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    pca::ResultPtr result = masterAlgorithm.getResult();

    /* Print the results */
    printNumericTable(result->get(pca::eigenvalues), "Eigenvalues:");
    printNumericTable(result->get(pca::eigenvectors), "Eigenvectors:");
  }

  MPI_Finalize();
  return 0;
}
