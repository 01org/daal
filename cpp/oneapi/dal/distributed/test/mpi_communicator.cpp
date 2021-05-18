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

#include "oneapi/dal/distributed/mpi_communicator.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test {

class mpi_test_setup {
public:
    mpi_test_setup() {
        MPI_Init(nullptr, nullptr);
    }

    ~mpi_test_setup() {
        MPI_Finalize();
    }
};

class mpi_test {
public:
    mpi_test() {
        [[maybe_unused]] static mpi_test_setup global_setup;
    }

    std::int64_t mpi_comm_rank() const {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }

    std::int64_t mpi_comm_size() const {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }
};

TEST_M(mpi_test, "can create default mpi_communicator") {
    preview::mpi_communicator{};
}

TEST_M(mpi_test, "mpi_communicator has expected rank") {
    preview::mpi_communicator comm;

    REQUIRE(comm.get_rank() == mpi_comm_rank());
    REQUIRE(comm.get_rank_count() == mpi_comm_size());
}

TEST_M(mpi_test, "mpi_communicator can broadcast") {
    preview::mpi_communicator comm;
    constexpr std::int64_t count = 10;
    std::array<byte_t, count> data_to_send = { 0, 8, 5, 3, 6, 7, 1, 0, 9 };
    std::array<byte_t, count> data_to_recv;

    if (mpi_comm_rank() == 0) {
        comm.bcast(data_to_send.data(), sizeof(byte_t) * count, 0).wait();
        data_to_recv = data_to_send;
    }
    else {
        comm.bcast(data_to_recv.data(), sizeof(byte_t) * count, 0).wait();
    }

    REQUIRE(data_to_recv == data_to_send);
}

} // namespace oneapi::dal::test
