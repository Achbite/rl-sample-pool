#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd -P)}"
version="$(tr -d '[:space:]' < "${repo_dir}/VERSION")"
contracts_version="$(tr -d '[:space:]' < "${workspace_root}/rl-contracts/VERSION")"
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
contract_dir="${workspace_root}/.workspace/artifacts/rl-contracts/${contracts_version}/training"
output_dir="${workspace_root}/.workspace/artifacts/rl-sample-pool/${version}/${platform_dir}"
builder_image="rl-training/sample-pool-builder:${version}"

test -f "${contract_dir}/cpp/common.pb.cc"
test -f "${contract_dir}/cpp/training.pb.cc"

mkdir -p "${workspace_root}/.workspace/artifacts" "$(dirname "${output_dir}")"
temp_dir="$(mktemp -d "${workspace_root}/.workspace/artifacts/.tmp-rl-sample-pool.XXXXXX")"
trap 'rm -rf "${temp_dir}"' EXIT
mkdir -p "${temp_dir}/bin" "${temp_dir}/config"

docker build \
    --file "${repo_dir}/Dockerfile.build" \
    --tag "${builder_image}" \
    "${repo_dir}"
docker run --rm \
    --volume "${repo_dir}:/source:ro" \
    --volume "${contract_dir}:/contracts:ro" \
    --volume "${temp_dir}:/output" \
    "${builder_image}" \
    bash -lc '
        set -euo pipefail
        cmake -S /source -B /tmp/sample-build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF \
            -DCONTRACT_CPP_DIR=/contracts/cpp
        cmake --build /tmp/sample-build --parallel --target maze_sample_pool
        cp /tmp/sample-build/maze_sample_pool /output/bin/
    '

cp "${repo_dir}/configs/pool_config.yaml" "${temp_dir}/config/pool_config.yaml"
test -x "${temp_dir}/bin/maze_sample_pool"
test -f "${temp_dir}/config/pool_config.yaml"

rm -rf "${output_dir}"
mv "${temp_dir}" "${output_dir}"
trap - EXIT
printf '%s\n' "${output_dir}"
