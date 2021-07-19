package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_test_suite",
    "dal_collect_modules",
)

IOS = [
    "csv",
]

dal_collect_modules(
    name = "io",
    root = "@onedal//cpp/oneapi/dal/io",
    modules = IOS,
    dal_deps = [
    ],
)

dal_test_suite(
    name = "tests",
    tests = [],
)
