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

#pragma once

#ifdef ONEDAL_DATA_PARALLEL
#include <daal/include/data_management/data/internal/numeric_table_sycl_homogen.h>
#endif

#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/backend/interop/sycl_table_adapter.hpp"
#include "oneapi/dal/table/backend/interop/host_homogen_table_adapter.hpp"
#include "oneapi/dal/table/backend/interop/host_soa_table_adapter.hpp"
#include "oneapi/dal/table/backend/interop/host_csr_table_adapter.hpp"
#include "oneapi/dal/backend/interop/csr_block_owner.hpp"

namespace oneapi::dal::backend::interop {

template <typename Data>
inline auto allocate_daal_homogen_table(std::int64_t row_count, std::int64_t column_count) {
    return daal::data_management::HomogenNumericTable<Data>::create(
        dal::detail::integral_cast<std::size_t>(column_count),
        dal::detail::integral_cast<std::size_t>(row_count),
        daal::data_management::NumericTable::doAllocate);
}

template <typename Data>
inline auto empty_daal_homogen_table(std::int64_t column_count) {
    return daal::data_management::HomogenNumericTable<Data>::create(
        dal::detail::integral_cast<std::size_t>(column_count),
        dal::detail::integral_cast<std::size_t>(0),
        daal::data_management::NumericTable::notAllocate);
}

template <typename Data>
inline auto convert_to_daal_homogen_table(array<Data>& data,
                                          std::int64_t row_count,
                                          std::int64_t column_count,
                                          bool allow_copy = false) {
    if (!data.get_count()) {
        return daal::services::SharedPtr<daal::data_management::HomogenNumericTable<Data>>();
    }

    if (allow_copy) {
        data.need_mutable_data();
    }

    ONEDAL_ASSERT(data.has_mutable_data());
    const auto daal_data =
        daal::services::SharedPtr<Data>(data.get_mutable_data(), daal_object_owner{ data });

    return daal::data_management::HomogenNumericTable<Data>::create(
        daal_data,
        dal::detail::integral_cast<std::size_t>(column_count),
        dal::detail::integral_cast<std::size_t>(row_count));
}

template <typename Data>
inline daal::data_management::NumericTablePtr copy_to_daal_homogen_table(const table& table) {
    // TODO: Preserve information about features
    const bool allow_copy = true;
    auto rows = row_accessor<const Data>{ table }.pull();
    return convert_to_daal_homogen_table(rows,
                                         table.get_row_count(),
                                         table.get_column_count(),
                                         allow_copy);
}

template <typename Data>
inline table convert_from_daal_homogen_table(const daal::data_management::NumericTablePtr& nt) {
    if (nt->getNumberOfRows() == 0) {
        return table{};
    }
    daal::data_management::BlockDescriptor<Data> block;
    const std::int64_t row_count = dal::detail::integral_cast<std::int64_t>(nt->getNumberOfRows());
    const std::int64_t column_count =
        dal::detail::integral_cast<std::int64_t>(nt->getNumberOfColumns());

    nt->getBlockOfRows(0, row_count, daal::data_management::readOnly, block);
    Data* data = block.getBlockPtr();
    array<Data> arr(data, row_count * column_count, [nt, block](Data* p) mutable {
        nt->releaseBlockOfRows(block);
    });
    return detail::homogen_table_builder{}.reset(arr, row_count, column_count).build();
}

inline daal::data_management::NumericTablePtr wrap_by_host_homogen_adapter(
    const homogen_table& table) {
    const auto& dtype = table.get_metadata().get_data_type(0);

    switch (dtype) {
        case data_type::float32: return host_homogen_table_adapter<float>::create(table);
        case data_type::float64: return host_homogen_table_adapter<double>::create(table);
        case data_type::int32: return host_homogen_table_adapter<std::int32_t>::create(table);
        default: return daal::data_management::NumericTablePtr();
    }
}

inline daal::data_management::NumericTablePtr wrap_by_host_soa_adapter(const homogen_table& table) {
    const auto& dtype = table.get_metadata().get_data_type(0);

    switch (dtype) {
        case data_type::float32: return host_soa_table_adapter::create<float>(table);
        case data_type::float64: return host_soa_table_adapter::create<double>(table);
        case data_type::int32: return host_soa_table_adapter::create<std::int32_t>(table);
        default: return daal::data_management::NumericTablePtr();
    }
}

template <typename Data>
inline daal::data_management::NumericTablePtr convert_to_daal_table(const homogen_table& table) {
    if (table.get_data_layout() == data_layout::row_major) {
        if (auto wrapper = wrap_by_host_homogen_adapter(table)) {
            return wrapper;
        }
    }
    else if (table.get_data_layout() == data_layout::column_major) {
        if (auto wrapper = wrap_by_host_soa_adapter(table)) {
            return wrapper;
        }
    }
    return copy_to_daal_homogen_table<Data>(table);
}

template <typename T>
inline auto convert_to_daal_csr_table(array<T>& data,
                                      array<std::int64_t>& column_indices,
                                      array<std::int64_t>& row_indices,
                                      std::int64_t row_count,
                                      std::int64_t column_count,
                                      bool allow_copy = false) {
    ONEDAL_ASSERT(data.get_count() == column_indices.get_count());
    ONEDAL_ASSERT(row_indices.get_count() == row_count + 1);

    if (!data.get_count() || !column_indices.get_count() || !row_indices.get_count()) {
        return daal::services::SharedPtr<daal::data_management::CSRNumericTable>();
    }

    if (allow_copy) {
        data.need_mutable_data();
        column_indices.need_mutable_data();
        row_indices.need_mutable_data();
    }

    ONEDAL_ASSERT(data.has_mutable_data());
    ONEDAL_ASSERT(column_indices.has_mutable_data());
    ONEDAL_ASSERT(row_indices.has_mutable_data());

    const auto daal_data =
        daal::services::SharedPtr<T>(data.get_mutable_data(), daal_object_owner{ data });
    ONEDAL_ASSERT(sizeof(std::size_t) == sizeof(std::int64_t));
    const auto daal_column_indices = daal::services::SharedPtr<std::size_t>(
        reinterpret_cast<std::size_t*>(column_indices.get_mutable_data()),
        daal_object_owner{ column_indices });
    const auto daal_row_indices = daal::services::SharedPtr<std::size_t>(
        reinterpret_cast<std::size_t*>(row_indices.get_mutable_data()),
        daal_object_owner{ row_indices });

    return daal::data_management::CSRNumericTable::create(
        daal_data,
        daal_column_indices,
        daal_row_indices,
        dal::detail::integral_cast<std::size_t>(column_count),
        dal::detail::integral_cast<std::size_t>(row_count));
}

template <typename Float>
inline daal::data_management::CSRNumericTablePtr copy_to_daal_csr_table(
    const detail::csr_table& table) {
    const bool allow_copy = true;
    auto block = detail::csr_accessor<const Float>{ table }.pull();
    return convert_to_daal_csr_table(block.data,
                                     block.column_indices,
                                     block.row_indices,
                                     table.get_row_count(),
                                     table.get_column_count(),
                                     allow_copy);
}

template <typename T>
inline table convert_from_daal_csr_table(const daal::data_management::NumericTablePtr& nt) {
    auto block_owner = std::make_shared<csr_block_owner<T>>(csr_block_owner<T>{ nt });

    ONEDAL_ASSERT(sizeof(std::size_t) == sizeof(std::int64_t));

    return detail::csr_table{
        array<T>{ block_owner->get_data(),
                  block_owner->get_element_count(),
                  [block_owner](const T* p) {} },
        array<std::int64_t>{ reinterpret_cast<std::int64_t*>(block_owner->get_column_indices()),
                             block_owner->get_element_count(),
                             [block_owner](const std::int64_t* p) {} },
        array<std::int64_t>{ reinterpret_cast<std::int64_t*>(block_owner->get_row_indices()),
                             block_owner->get_row_count() + 1,
                             [block_owner](const std::int64_t* p) {} },
        block_owner->get_row_count(),
        block_owner->get_column_count()
    };
}

inline daal::data_management::CSRNumericTablePtr wrap_by_host_csr_adapter(
    const detail::csr_table& table) {
    const auto& dtype = table.get_metadata().get_data_type(0);

    switch (dtype) {
        case data_type::float32: return host_csr_table_adapter<float>::create(table);
        case data_type::float64: return host_csr_table_adapter<double>::create(table);
        case data_type::int32: return host_csr_table_adapter<std::int32_t>::create(table);
        default: return daal::data_management::CSRNumericTablePtr();
    }
}

template <typename Float>
inline daal::data_management::CSRNumericTablePtr convert_to_daal_table(
    const detail::csr_table& table) {
    auto wrapper = wrap_by_host_csr_adapter(table);
    if (!wrapper) {
        return copy_to_daal_csr_table<Float>(table);
    }
    else {
        return wrapper;
    }
}

template <typename Data>
inline daal::data_management::NumericTablePtr convert_to_daal_table(const table& table) {
    if (table.get_kind() == homogen_table::kind()) {
        const auto& homogen = static_cast<const homogen_table&>(table);
        return convert_to_daal_table<Data>(homogen);
    }
    else if (table.get_kind() == detail::csr_table::kind()) {
        const auto& csr = static_cast<const detail::csr_table&>(table);
        return convert_to_daal_table<Data>(csr);
    }
    else {
        return copy_to_daal_homogen_table<Data>(table);
    }
}

template <typename Data>
inline table convert_from_daal_table(const daal::data_management::NumericTablePtr& nt) {
    if (nt->getDataLayout() == daal::data_management::NumericTableIface::StorageLayout::csrArray) {
        return convert_from_daal_csr_table<Data>(nt);
    }
    else {
        return convert_from_daal_homogen_table<Data>(nt);
    }
}

#ifdef ONEDAL_DATA_PARALLEL
inline daal::data_management::NumericTablePtr convert_to_daal_table(const sycl::queue& queue,
                                                                    const table& table) {
    return interop::sycl_table_adapter::create(queue, table);
}

template <typename Data>
inline daal::data_management::NumericTablePtr convert_to_daal_table(const sycl::queue& queue,
                                                                    const array<Data>& data,
                                                                    std::int64_t row_count,
                                                                    std::int64_t column_count) {
    using daal::services::Status;
    using daal::services::SharedPtr;
    using daal::services::internal::Buffer;
    using daal::data_management::internal::SyclHomogenNumericTable;
    using dal::detail::integral_cast;

    ONEDAL_ASSERT(data.get_count() == row_count * column_count);
    ONEDAL_ASSERT(data.has_mutable_data());
    ONEDAL_ASSERT(is_same_context(queue, data));

    const SharedPtr<Data> data_shared{ data.get_mutable_data(), daal_object_owner{ data } };

    Status status;
    const Buffer<Data> buffer{ data_shared,
                               integral_cast<std::size_t>(data.get_count()),
                               queue,
                               status };
    status_to_exception(status);

    const auto table =
        SyclHomogenNumericTable<Data>::create(buffer,
                                              integral_cast<std::size_t>(column_count),
                                              integral_cast<std::size_t>(row_count),
                                              &status);
    status_to_exception(status);

    return table;
}
#endif

} // namespace oneapi::dal::backend::interop
