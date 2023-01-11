workspace(name = "w3c_patch_subset_incxfer")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Google Test
http_archive(
    name = "gtest",
    sha256 = "24564e3b712d3eb30ac9a85d92f7d720f60cc0173730ac166f27dda7fed76cb2",
    strip_prefix = "googletest-release-1.12.1",
    url = "https://github.com/google/googletest/archive/release-1.12.1.zip",
)

# Brotli Encoder/Decoder
http_archive(
    name = "brotli",
    build_file = "//third_party:brotli.BUILD",
    sha256 = "3b90c83489879701c05f39b68402ab9e5c880ec26142b5447e73afdda62ce525",
    strip_prefix = "brotli-71fe6cac061ac62c0241f410fbd43a04a6b4f303",
    url = "https://github.com/google/brotli/archive/71fe6cac061ac62c0241f410fbd43a04a6b4f303.zip",
)

# WOFF2 Encoder/Decoder
http_archive(
    name = "woff2",
    build_file = "//third_party:woff2.BUILD",
    sha256 = "db9ebe2aff6520e22ad9491863fc9e851b71fedbabefbb32508935d0f5cecf91",
    strip_prefix = "woff2-a0d0ed7da27b708c0a4e96ad7a998bddc933c06e",
    url = "https://github.com/google/woff2/archive/a0d0ed7da27b708c0a4e96ad7a998bddc933c06e.zip",
)

# harfbuzz
http_archive(
    name = "harfbuzz",
    build_file = "//third_party:harfbuzz.BUILD",
    sha256 = "f6a9083886cbe502d765163bc2e1babe3ce16adb75a68b5a54cae6b9eed9695c",
    strip_prefix = "harfbuzz-8f1bf23cc9a8912c452f7571e2a3f35a192a8120",
    urls = ["https://github.com/harfbuzz/harfbuzz/archive/8f1bf23cc9a8912c452f7571e2a3f35a192a8120.zip"],
)

# Fast Hash
http_archive(
    name = "fasthash",
    build_file = "//third_party:fasthash.BUILD",
    sha256 = "0f8fba20ea2b502c2aaec56d850367768535003ee0fc0e56043283db64e483ee",
    strip_prefix = "fast-hash-ae3bb53c199fe75619e940b5b6a3584ede99c5fc",
    urls = ["https://github.com/ztanml/fast-hash/archive/ae3bb53c199fe75619e940b5b6a3584ede99c5fc.zip"],
)

# abseil-cpp

http_archive(
    name = "com_google_absl",
    sha256 = "54707f411cb62a26a776dad5fd60829098c181700edcd022ea5c2ca49e9b7ef1",
    strip_prefix = "abseil-cpp-20220623.1",
    urls = ["https://github.com/abseil/abseil-cpp/archive/20220623.1.zip"],
)

http_archive(
  name = "bazel_skylib",
  urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz"],
  sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
)

# Proto buf generating rules
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_proto",
    sha256 = "dc3fb206a2cb3441b485eb1e423165b231235a1ea9b031b4433cf7bc1fa460dd",
    strip_prefix = "rules_proto-5.3.0-21.7",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.7.tar.gz",
    ],
)
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

### Emscripten ###

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "emsdk",
    sha256 = "506376d0d2a71fc3dd1a4dba6fb4cf18f0a2fa4e1936aa04ba4b59f2d435bf3f",
    strip_prefix = "emsdk-3.1.29/bazel",
    url = "https://github.com/emscripten-core/emsdk/archive/3.1.29.tar.gz",
)

load("@emsdk//:deps.bzl", emsdk_deps = "deps")
emsdk_deps()

load("@emsdk//:emscripten_deps.bzl", emsdk_emscripten_deps = "emscripten_deps")
emsdk_emscripten_deps(emscripten_version = "3.1.29")

load("@emsdk//:toolchains.bzl", "register_emscripten_toolchains")
register_emscripten_toolchains()


### End Emscripten ###

# libcbor
http_archive(
    name = "libcbor",
    build_file = "//third_party:libcbor.BUILD",
    sha256 = "dd04ea1a7df484217058d389e027e7a0143a4f245aa18a9f89a5dd3e1a4fcc9a",
    strip_prefix = "libcbor-0.8.0",
    urls = ["https://github.com/PJK/libcbor/archive/refs/tags/v0.8.0.zip"],
)

# open-vcdiff
http_archive(
    name = "open-vcdiff",
    build_file = "//third_party:open-vcdiff.BUILD",
    sha256 = "39ce3a95f72ba7b64e8054d95e741fc3c69abddccf9f83868a7f52f3ae2174c0",
    strip_prefix = "open-vcdiff-868f459a8d815125c2457f8c74b12493853100f9",
    urls = ["https://github.com/google/open-vcdiff/archive/868f459a8d815125c2457f8c74b12493853100f9.zip"],
)


