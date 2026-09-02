#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
contract_cpp_dir="${CONTRACT_CPP_DIR:-/contracts/cpp}"

if [ "$#" -ne 0 ]; then
    echo "usage: bash ./test.sh" >&2
    exit 2
fi

if [ ! -f "${contract_cpp_dir}/training.pb.cc" ]; then
    echo "CONTRACT_CPP_DIR must point to generated rl-contracts C++ bindings" >&2
    echo "current value: ${contract_cpp_dir}" >&2
    exit 1
fi

test_build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rl-sample-pool-test.XXXXXX")"
trap 'rm -rf "${test_build_dir}"' EXIT

cmake -S "${repo_dir}" -B "${test_build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DCONTRACT_CPP_DIR="${contract_cpp_dir}"
cmake --build "${test_build_dir}" --parallel \
    --target sample_pool_development_test
ctest --test-dir "${test_build_dir}" \
    --output-on-failure \
    -R '^sample_pool_data_path$'
