#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
channel="production"
if [ "${1:-}" = "--development" ]; then
    channel="development"
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "usage: bash build_artifact.sh [--development]" >&2
    exit 2
fi
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
version="$(tr -d '[:space:]' < "${repo_dir}/VERSION")"
builder_image="rl-training/sample-pool-builder:${version}"
artifact_root="${RL_ARTIFACT_ROOT:-${repo_dir}/build/artifacts/${channel}}"
if [ -z "${CONTRACT_CPP_DIR:-}" ]; then
    echo "CONTRACT_CPP_DIR must point to an existing rl-contracts training artifact's cpp directory" >&2
    exit 2
fi
contract_cpp_dir="$(cd "${CONTRACT_CPP_DIR}" && pwd -P)"
contract_dir="$(cd "${contract_cpp_dir}/.." && pwd -P)"
output_dir="${artifact_root}/${platform_dir}"
if [ "${channel}" = "development" ]; then
    output_dir="${output_dir}/current"
fi

docker build --provenance=false --file "${repo_dir}/Dockerfile.build" \
    --tag "${builder_image}" "${repo_dir}" >&2
builder_id="$(docker image inspect --format '{{.Id}}' "${builder_image}")"
build_relative="${platform_dir}/${builder_id#sha256:}/${channel}"
build_dir="${repo_dir}/build/${build_relative}"
mkdir -p "${build_dir}" "${output_dir}/bin" "${output_dir}/config"
docker run --rm \
    --volume "${repo_dir}:/source:ro" \
    --volume "${contract_dir}:${contract_dir}:ro" \
    --volume "${build_dir}:/build" \
    --volume "rl-training-sample-pool-ccache:/var/cache/ccache" \
    --env CCACHE_DIR=/var/cache/ccache \
    --env "CONTRACT_CPP_DIR=${contract_cpp_dir}" \
    "${builder_id}" bash -lc '
        set -euo pipefail
        # Host edits can remain behind virtiofs attribute caches. Refresh only
        # build inputs before Ninja decides whether an object is up to date.
        find /source/CMakeLists.txt /source/main /source/src \
            "${CONTRACT_CPP_DIR}" "${CONTRACT_CPP_DIR}/../sdk" \
            -exec stat --cached=never --printf="" {} +
        cmake -S /source -B /build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DCONTRACT_CPP_DIR="${CONTRACT_CPP_DIR}"
        cmake --build /build --parallel --target maze_sample_pool
    ' >&2
# Both the local run entrypoint and artifact export select this successful build.
ln -sfn "${build_relative}/maze_sample_pool" "${repo_dir}/build/maze_sample_pool"
rsync -rcp "${build_dir}/maze_sample_pool" "${output_dir}/bin/"
rsync -rcp "${repo_dir}/configs/pool_config.yaml" "${output_dir}/config/"
printf '%s\n' "${output_dir}"
