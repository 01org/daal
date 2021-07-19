package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "decision_forest",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/algo:decision_tree",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:sort",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/classification:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/regression:kernel",
    ]
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":decision_forest",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
    ],
)
