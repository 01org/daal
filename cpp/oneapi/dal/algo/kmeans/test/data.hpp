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

#pragma once

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

namespace oneapi::dal::kmeans::test {

namespace te = dal::test::engine;

class gold_dataset {
public:
    gold_dataset() = delete;

    static std::int64_t get_row_count() {
        return row_count;
    }

    static std::int64_t get_column_count() {
        return column_count;
    }

    static std::int64_t get_cluster_count() {
        return cluster_count;
    }

    static te::dataframe get_data() {
        static std::array<float, row_count* column_count> data = {
            -0.4211976,  -1.77513504, 1.27569859, //
            0.77215635,  0.83386471,  2.14067957, //
            0.73995455,  -0.27936096, 0.32836769, //
            0.9878956,   -1.76489722, -1.55065013, //
            -0.23259897, -3.25119741, 1.90446895, //
            1.40689361,  -0.31115323, -1.66140443, //
            1.1420697,   2.19824988,  3.87801049, //
            1.16309132,  -0.01096376, 2.07662207, //
            -1.27267835, -0.32793146, 0.37731112, //
            1.01918695,  0.58832042,  1.03438906, //
            -1.67515032, 0.56246549,  0.36486751, //
            -0.68945814, -2.37428525, 1.13609649, //
            1.07247479,  1.38376183,  -0.95088791, //
            -1.77092795, -1.02016895, -1.07451544, //
            1.24463038,  -0.29037568, -0.64246558, //
            0.57212026,  -0.89804204, 0.095685, //
            0.50918221,  -0.0821226,  -1.15073783, //
            -0.59723021, -1.80517875, 0.70360099, //
            0.38209732,  -1.276624,   0.99510949, //
            0.83370941,  2.11599648,  2.93158122, //
            0.3824402,   -2.19852439, 0.08723186, //
            0.65964977,  -1.09472161, -0.07429885, //
            1.26570172,  0.12698797,  -0.86244866, //
            1.50249537,  -0.13932538, -1.84652558, //
            1.15278953,  2.60733029,  1.04371592, //
            0.88069536,  -0.14010503, 0.42766178, //
            -0.76174957, -1.80284196, 1.8097555, //
            1.58933174,  0.64489407,  1.16527915, //
            0.84192179,  0.04067853,  -0.26056908, //
            -1.74762976, -1.09340904, 0.40319338, //
            -0.76731766, -0.98578116, 1.14794179, //
            0.65595061,  0.65688586,  1.26024014, //
            -0.15238779, -0.39842472, -0.08792889, //
            1.37645509,  0.60210913,  1.30110769, //
            -0.36163031, -3.03840135, 1.59898798, //
            0.67281735,  -3.09144198, -1.62539633, //
            0.15527585,  -0.2617361,  3.79020891, //
            0.87746671,  -1.10000304, -0.71070825, //
            -2.23680917, -0.47567914, 1.10073481, //
            -2.96471037, 0.42241915,  2.65339462, //
            0.9315062,   1.62132009,  1.02924194, //
            0.15601499,  -1.10261452, -0.23213162, //
            1.17970448,  -1.13192713, -2.17789545, //
            -1.28621184, -0.57017579, 1.33426628, //
            1.87437876,  0.990202,    -1.94993024, //
            1.63200008,  0.06698476,  -2.52710686, //
            1.34262114,  0.94681388,  2.78080064, //
            -1.61395326, -0.6503645,  0.55501739, //
            1.06344249,  -0.53785431, -0.82836749, //
            0.31822215,  -1.27804688, 1.31185101, //
            0.90394784,  1.76803287,  0.67041956, //
            0.58766195,  -1.10294262, 0.10232099, //
            0.37488412,  -2.12013662, -0.36183756, //
            -1.13788633, -0.97778709, 0.79764553, //
            -0.24496859, -1.0399562,  -0.10300635, //
            1.06667341,  -1.05407813, -1.11806412, //
            1.91078856,  1.28195572,  -0.01544957, //
            -0.03901089, -1.08121301, 2.13838797, //
            1.06196011,  1.01609094,  3.43781217, //
            1.50014415,  1.54477634,  1.25127517, //
            -0.99254206, 0.63330495,  2.60882098, //
            -2.19571245, 1.18419044,  2.5043204, //
            1.35847099,  -0.15581143, -1.62243044, //
            -1.81168441, -0.44494955, 2.01074606, //
            1.3058768,   -0.56829473, 0.02833572, //
            -0.23301173, -1.34779128, 1.52301631, //
            1.29824917,  -1.94374495, -2.98928684, //
            -1.9959795,  -0.65793058, 0.95630083, //
            0.56553457,  -2.58990424, -0.80840958, //
            1.1226466,   -0.16574495, 3.08252859, //
            1.56871812,  2.09986847,  1.07256432, //
            -1.25587664, 1.65667526,  4.15491654, //
            0.93876136,  -2.35808025, -1.87920405, //
            -0.8471835,  -0.69440113, 2.93681617, //
            1.59009779,  2.20871231,  -0.48965949, //
            1.06281801,  -0.0972685,  -0.42646198, //
            -1.30326865, -1.05482093, 0.70459856, //
            -0.09209033, -1.20780273, 1.89052171, //
            0.49382876,  0.1294951,   -0.20685572, //
            0.68082521,  -0.79019663, -0.52872095, //
            0.71753211,  -2.05470765, -0.83981589, //
            1.35769056,  1.51923224,  1.03336847, //
            -2.68472063, 1.02398527,  1.73875478, //
            -1.85761645, 1.1126523,   2.50932265, //
            0.97137302,  -0.45959616, -0.77741839, //
            1.28593955,  2.65851125,  1.13124138, //
            1.55515919,  0.97940204,  -0.82872543, //
            0.25941505,  -0.243583,   0.60783035, //
            0.22784927,  -3.06459701, -0.27873757, //
            1.4518669,   0.23012689,  -1.31553611, //
            -0.75287405, -1.33529655, -0.54345691, //
            0.61762095,  -0.57795275, 0.45454203, //
            -0.54526154, -1.88427543, -0.738393, //
            -0.45797315, -1.2040845,  -0.13653798, //
            0.21506145,  -3.44015965, -0.45534725, //
            0.9155692,   -1.39488343, -0.91673409, //
            1.43598828,  1.66680306,  1.22073332, //
            -1.46847185, 1.25844391,  2.864725, //
            0.64547726,  -2.26341401, -0.50330225, //
            1.90406703,  1.35132357,  1.83569703, //
        };
        return te::dataframe{ data.data(), row_count, column_count };
    }

    static te::dataframe get_initial_centroids() {
        static std::array<float, cluster_count* column_count> data = {
            0.67903983,  -1.48359774, -0.73226038, //
            0.7946199,   0.7242352,   1.23007028, //
            -1.32741404, -0.74386852, 0.92412268, //
        };
        return te::dataframe{ data.data(), cluster_count, column_count };
    }

private:
    static constexpr std::int64_t row_count = 100;
    static constexpr std::int64_t column_count = 3;
    static constexpr std::int64_t cluster_count = 3;
};

} // namespace oneapi::dal::kmeans::test
