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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/feature_type_helper.hpp"

namespace oneapi::dal::decision_forest::test {

namespace de = dal::detail;
namespace be = dal::decision_forest::backend;
namespace te = dal::test::engine;

template <typename TestType>
class indexed_feature_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Bin = std::tuple_element_t<1, TestType>;
    using Index = std::tuple_element_t<2, TestType>;

public:
    bool is_cpu() {
        return this->get_policy().is_cpu();
    }

    auto get_dataframe_base() {
        constexpr std::int64_t row_count = 6;
        constexpr std::int64_t column_count = 3;

        static const float data_arr[] = { -2.f, -9.f, 0.f, -1.f, -1.f, 0.f, -1.f, -2.f, 0.f,
                                          +1.f, +1.f, 1.f, +1.f, +2.f, 1.f, +2.f, +1.f, 1.f };

        static const Index bin_count_arr[] = { 4, 5, 2 };

        static const Float bin_border_arr[] = { -2.f, -1.f, 1.f, 2.f, -9.f, -2.f,
                                                -1.f, 1.f,  2.f, 0.f, 1.f };

        te::dataframe data{ array<float>::wrap(data_arr, row_count * column_count),
                            row_count,
                            column_count };

        return std::tuple(data, bin_count_arr, bin_border_arr);
    }

    auto base_checks(const te::dataframe& data,
                     const te::table_id& data_table_id,
                     const Index* bin_count_arr_ref,
                     const Float* bin_border_arr_ref) {
        auto& q = this->get_queue();
        const auto tbl = data.get_table(data_table_id, range(0, -1));

        be::indexed_features<Float, Bin, Index> iftrs(q, 5, 256);

        iftrs(tbl).wait_and_throw();

        Index total_bin_count_ref = 0;
        for (Index clmn_idx = 0; clmn_idx < tbl.get_column_count(); clmn_idx++) {
            Index bin_count_ref = bin_count_arr_ref[clmn_idx];
            total_bin_count_ref += bin_count_ref;

            if (iftrs.get_bin_count(clmn_idx) != bin_count_ref) {
                CAPTURE(clmn_idx, iftrs.get_bin_count(clmn_idx), bin_count_arr_ref[clmn_idx]);
                FAIL("incorrect number of bins for column");
            }
        }

        REQUIRE(iftrs.get_total_bin_count() == total_bin_count_ref);

        Index bin_offset = 0;
        for (Index clmn_idx = 0; clmn_idx < tbl.get_column_count(); clmn_idx++) {
            auto bin_borders_nd = iftrs.get_bin_borders(clmn_idx);
            auto bin_borders_nd_host = bin_borders_nd.to_host(q);
            const Float* bin_border_arr = bin_borders_nd_host.get_data();

            for (Index bin_idx = 0; bin_idx < iftrs.get_bin_count(clmn_idx); bin_idx++) {
                if (bin_border_arr[bin_idx] != bin_border_arr_ref[bin_offset + bin_idx]) {
                    CAPTURE(clmn_idx,
                            bin_idx,
                            bin_border_arr[bin_idx],
                            bin_border_arr_ref[bin_offset + bin_idx]);
                    FAIL("incorrect bin_borders for column");
                }
            }
            bin_offset += iftrs.get_bin_count(clmn_idx);
        }
    }
};

using indexed_features_types = _TE_COMBINE_TYPES_3((float, double),
                                                   (std::uint32_t, std::uint8_t),
                                                   (std::uint32_t));

#define INDEXED_FEATURES_TEST(name) \
    TEMPLATE_LIST_TEST_M(indexed_feature_test, name, "[df][unit]", indexed_features_types)

INDEXED_FEATURES_TEST("indexed features base checks") {
    SKIP_IF(this->is_cpu());

    const auto [data, bin_count_ref, bin_border_ref] = this->get_dataframe_base();

    this->base_checks(data, this->get_homogen_table_id(), bin_count_ref, bin_border_ref);
}

} // namespace oneapi::dal::decision_forest::test
