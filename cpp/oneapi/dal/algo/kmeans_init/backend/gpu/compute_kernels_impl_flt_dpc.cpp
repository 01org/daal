/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernels_impl.hpp"

namespace oneapi::dal::kmeans_init::backend {

#ifdef ONEDAL_DATA_PARALLEL
template struct kmeans_init_kernel<float, kmeans_init::method::dense>;
template struct kmeans_init_kernel<float, kmeans_init::method::random_dense>;
template struct kmeans_init_kernel<float, kmeans_init::method::plus_plus_dense>;
template struct kmeans_init_kernel<float, kmeans_init::method::parallel_plus_dense>;
#endif

} // namespace oneapi::dal::kmeans_init::backend
