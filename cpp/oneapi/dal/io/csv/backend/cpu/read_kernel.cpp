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

#ifndef ONEDAL_DATA_CONVERSION

#define ONEDAL_DATA_CONVERSION
#include "daal/include/data_management/data_source/csv_feature_manager.h"
#include "daal/include/data_management/data_source/file_data_source.h"
#undef ONEDAL_DATA_CONVERSION

#endif

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/io/csv/backend/cpu/read_kernel.hpp"
#include "oneapi/dal/io/csv/backend/cpu/read_graph.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::csv::backend {

namespace interop = dal::backend::interop;
namespace daal_dm = daal::data_management;

template <>
table read_kernel_cpu<table>::operator()(const dal::backend::context_cpu& ctx,
                                         const detail::data_source_base& ds,
                                         const read_args<table>& args) const {
    daal_dm::CsvDataSourceOptions csv_options(daal_dm::operator|(
        daal_dm::operator|(daal_dm::CsvDataSourceOptions::allocateNumericTable,
                           daal_dm::CsvDataSourceOptions::createDictionaryFromContext),
        (ds.get_parse_header() ? daal_dm::CsvDataSourceOptions::parseHeader
                               : daal_dm::CsvDataSourceOptions::byDefault)));

    daal_dm::FileDataSource<daal_dm::CSVFeatureManager> daal_data_source(ds.get_file_name().c_str(),
                                                                         csv_options);
    interop::status_to_exception(daal_data_source.status());

    daal_data_source.getFeatureManager().setDelimiter(ds.get_delimiter());
    daal_data_source.loadDataBlock();
    interop::status_to_exception(daal_data_source.status());

    return oneapi::dal::backend::interop::convert_from_daal_homogen_table<DAAL_DATA_TYPE>(
        daal_data_source.getNumericTable());
}
} // namespace oneapi::dal::csv::backend
