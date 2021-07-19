load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_collect_modules",
    "dal_collect_test_suites",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "common",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal:table",
    ],
)

dal_collect_modules(
    name = "primitives",
    dal_deps = [
        ":common",
    ],
    modules = [
        "blas",
        "distance",
        "intersection",
        "lapack",
        "reduction",
        "rng",
        "selection",
        "sort",
        "stat",
        "super_accumulator",
    ],
    root = "@onedal//cpp/oneapi/dal/backend/primitives",
)

dal_test_suite(
    name = "dpc_compiler_tests",
    srcs = [
        "test/fill_dpc.cpp",
    ],
    compile_as = ["dpc++"],
    dal_deps = [
        ":common",
    ],
    framework = "catch2",
)

dal_test_suite(
    name = "common_tests",
    srcs = glob(
        [
            "test/*.cpp",
        ],
        exclude = [
            "test/*_dpc.cpp",
        ],
    ),
    dal_deps = [
        ":common",
    ],
    framework = "catch2",
    private = True,
)

dal_collect_test_suites(
    name = "tests",
    modules = [
        "blas",
        "lapack",
        "reduction",
        "selection",
        "sort",
        "stat",
    ],
    root = "@onedal//cpp/oneapi/dal/backend/primitives",
    tests = [
        ":common_tests",
        ":dpc_compiler_tests",
    ],
)
