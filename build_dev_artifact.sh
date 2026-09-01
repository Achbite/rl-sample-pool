#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd -P)}"
contracts_repo="${workspace_root}/rl-contracts"
tool="${contracts_repo}/scripts/artifact_manifest.py"
version="$(tr -d '[:space:]' < "${repo_dir}/VERSION")"
source "${repo_dir}/artifact_versions.env"
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
contract_dir="${workspace_root}/.workspace/dev-artifacts/rl-contracts/${RL_TRAINING_CONTRACTS_VERSION}/training/current"
output_dir="${workspace_root}/.workspace/dev-artifacts/rl-sample-pool/${version}/${platform_dir}/current"
builder_image="rl-training/sample-pool-dev-builder:${version}"

python3 "${tool}" verify \
    --root "${contract_dir}" \
    --package rl-training-contracts \
    --version "${RL_TRAINING_CONTRACTS_VERSION}" \
    --channel development \
    --require-file cpp/common.pb.cc \
    --require-file cpp/training.pb.cc

mkdir -p "${workspace_root}/.workspace/dev-artifacts" "$(dirname "${output_dir}")"
temp_dir="$(mktemp -d "${workspace_root}/.workspace/dev-artifacts/.tmp-rl-sample-pool.XXXXXX")"
trap 'rm -rf "${temp_dir}"' EXIT
mkdir -p "${temp_dir}/bin" "${temp_dir}/config"

docker build \
    --file "${repo_dir}/Dockerfile.build" \
    --tag "${builder_image}" \
    "${repo_dir}" >&2
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
    ' >&2

cp "${repo_dir}/configs/pool_config.yaml" "${temp_dir}/config/pool_config.yaml"
python3 "${tool}" finalize-component \
    --output "${temp_dir}" \
    --package rl-sample-pool \
    --version "${version}" \
    --platform "${platform}" \
    --channel development \
    --contract-manifest "${contract_dir}/manifest.json" \
    --executable "bin/maze_sample_pool"
python3 "${tool}" verify \
    --root "${temp_dir}" \
    --package rl-sample-pool \
    --version "${version}" \
    --platform "${platform}" \
    --channel development \
    --contract-manifest "${contract_dir}/manifest.json" \
    --require-executable "bin/maze_sample_pool" \
    --require-file "config/pool_config.yaml"

rm -rf "${output_dir}"
mv "${temp_dir}" "${output_dir}"
trap - EXIT
printf '%s\n' "${output_dir}"
