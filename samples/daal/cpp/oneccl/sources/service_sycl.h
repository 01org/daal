/* file: service_sycl.h */
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
!    Auxiliary sycl functions used in C++ examples
!******************************************************************************/

#ifndef _SERVICE_SYCL_H
#define _SERVICE_SYCL_H

#include <list>
#include <memory>

#include "mpi.h"
#include "oneapi/ccl.hpp"
#include <CL/cl.h>
#include <CL/sycl.hpp>

#include "service.h"

std::vector<sycl::device> get_gpus() {
  auto platforms = sycl::platform::get_platforms();
  for (auto p : platforms) {
    auto devices = p.get_devices(sycl::info::device_type::gpu);
    if (!devices.empty()) {
      return devices;
    }
  }
  return {};
}

int getLocalRank(ccl::communicator &comm, int size, int rank) {
  /* Obtain local rank among nodes sharing the same host name */
  char zero = static_cast<char>(0);
  std::vector<char> name(MPI_MAX_PROCESSOR_NAME + 1, zero);
  int resultlen = 0;
  MPI_Get_processor_name(name.data(), &resultlen);
  std::string str(name.begin(), name.end());
  std::vector<char> allNames((MPI_MAX_PROCESSOR_NAME + 1) * size, zero);
  std::vector<size_t> aReceiveCount(size, MPI_MAX_PROCESSOR_NAME + 1);
  ccl::allgatherv((int8_t *)name.data(), name.size(), (int8_t *)allNames.data(),
                  aReceiveCount, comm)
      .wait();
  int localRank = 0;
  for (int i = 0; i < rank; i++) {
    auto nameBegin = allNames.begin() + i * (MPI_MAX_PROCESSOR_NAME + 1);
    std::string nbrName(nameBegin, nameBegin + (MPI_MAX_PROCESSOR_NAME + 1));
    if (nbrName == str)
      localRank++;
  }
  return localRank;
}

void set_gpu_by_rank_if_possible(ccl::communicator &comm, int size, int rank) {
  auto gpus = get_gpus();
  if (gpus.size() > 0) {
    auto local_rank = getLocalRank(comm, size, rank);
    auto rank_gpu = gpus[local_rank % gpus.size()];
    cl::sycl::queue queue(rank_gpu);
    daal::services::SyclExecutionContext ctx(queue);
    daal::services::Environment::getInstance()->setDefaultExecutionContext(ctx);
  } else {
    cl::sycl::cpu_selector cpu;
    cl::sycl::queue queue(cpu);
    daal::services::SyclExecutionContext ctx(queue);
    daal::services::Environment::getInstance()->setDefaultExecutionContext(ctx);
  }
}
#endif
