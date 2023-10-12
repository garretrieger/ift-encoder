workspace(name = "w3c_patch_subset_incxfer")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

## To create compile commands json for clangd integration:
#   Hedron's Compile Commands Extractor for Bazel
#   https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",

    # Replace the commit hash in both places (below) with the latest, rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    sha256 = "3cd0e49f0f4a6d406c1d74b53b7616f5e24f5fd319eafc1bf8eee6e14124d115",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/3dddf205a1f5cde20faf2444c1757abe0564ff4c.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-3dddf205a1f5cde20faf2444c1757abe0564ff4c",
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()

# Base64
http_archive(
    name = "base64",
    build_file = "//third_party:base64.BUILD",
    sha256 = "c101f3ff13ce48d6b80b4c931eb9759ee48812ba2e989c82322d8abd5c8bbcf7",
    strip_prefix = "cpp-base64-82147d6d89636217b870f54ec07ddd3e544d5f69",
    url = "https://github.com/ReneNyffenegger/cpp-base64/archive/82147d6d89636217b870f54ec07ddd3e544d5f69.zip",
)

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
    sha256 = "730b7f9de381c7b5b09c81841604fa10c5dd67628822fa377b776ab7929fe18c",
    strip_prefix = "woff2-c8c0d339131e8f1889ae8aac0075913d98d9a722",
    url = "https://github.com/google/woff2/archive/c8c0d339131e8f1889ae8aac0075913d98d9a722.zip",
)

# harfbuzz
http_archive(
    name = "harfbuzz",
    build_file = "//third_party:harfbuzz.BUILD",
    sha256 = "032c712d82d38ec6d3677b349f93acd12c6ed19bbdfcd116a356b73ce8ef6588",
    strip_prefix = "harfbuzz-f26fd69d858642d76413b8f4068eaf9b57c40a5f",
    urls = ["https://github.com/harfbuzz/harfbuzz/archive/f26fd69d858642d76413b8f4068eaf9b57c40a5f.zip"],
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
    sha256 = "b1cbb5265bbf328ea839c5190e011709f1dd2103219f386d88c764216390244d",
    strip_prefix = "abseil-cpp-3020b58f0d987073b8adab204426f82c3f60b283",
    urls = ["https://github.com/abseil/abseil-cpp/archive/3020b58f0d987073b8adab204426f82c3f60b283.zip"],
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
    strip_prefix = "emsdk-3.1.44/bazel",
    sha256 = "cb8cded78f6953283429d724556e89211e51ac4d871fcf38e0b32405ee248e91",
    url = "https://github.com/emscripten-core/emsdk/archive/3.1.44.tar.gz",
)

load("@emsdk//:deps.bzl", emsdk_deps = "deps")
emsdk_deps()

load("@emsdk//:emscripten_deps.bzl", emsdk_emscripten_deps = "emscripten_deps")
emsdk_emscripten_deps(emscripten_version = "3.1.44")

load("@emsdk//:toolchains.bzl", "register_emscripten_toolchains")
register_emscripten_toolchains()


### End Emscripten ###

# IFTB - Binned Incremental Font Transfer

http_archive(
    name = "iftb",
    build_file = "//third_party:iftb.BUILD",
    sha256 = "a24060e6a0c9c7c7f601ec076564f4198ce3a38328f92cc9a32d9d1a7828b7b7",
    strip_prefix = "binned-ift-reference-b2f144293ae0f8f3311c6a31bace15ac11ba39ea",
    urls = ["https://github.com/garretrieger/binned-ift-reference/archive/b2f144293ae0f8f3311c6a31bace15ac11ba39ea.zip"],
)

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


