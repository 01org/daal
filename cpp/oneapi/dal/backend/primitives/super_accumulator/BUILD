package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "super_accumulator",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
)

dal_test_suite(
    name = "cpp_tests",
    framework = "catch2",
    compile_as = [ "c++", ],
    private = True,
    srcs = glob([
        "test/*.cpp",
    ], exclude=[
        "test/*perf*.cpp",
        "test/*_dpc.cpp",
    ]),
    dal_deps = [
        ":super_accumulator",
    ],
)

dal_test_suite(
    name = "dpc_tests",
    framework = "catch2",
    compile_as = [ "dpc++", ],
    private = True,
    srcs = glob([
        "test/*_dpc.cpp",
    ], exclude=[
        "test/*perf*.cpp",
    ]),
    dal_deps = [
        ":super_accumulator",
    ],
)

dal_test_suite(
    name = "tests",
    test_deps = [
        ":dpc_tests",
        ":cpp_tests",
    ],
    dal_deps = [
        ":super_accumulator",
    ],
)

