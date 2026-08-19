#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd -P)}"
contracts_repo="${workspace_root}/rl-contracts"
tool="${contracts_repo}/scripts/dev_artifact.py"
version="$(tr -d '[:space:]' < "${repo_dir}/VERSION")"
source "${repo_dir}/artifact_versions.env"

if [ ! -f "${tool}" ]; then
    echo "development artifact tool is missing: ${tool}" >&2
    exit 1
fi
contract_dir="$(RL_TRAINING_WORKSPACE="${workspace_root}" bash "${contracts_repo}/build_dev_artifact.sh")"
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
source_digest="$(python3 "${tool}" source-meta --repo "${repo_dir}" --field source_digest)"
artifact_input_digest="$(
    python3 - "${source_digest}" "${contract_dir}/manifest.json" <<'PY'
import hashlib
import sys
from pathlib import Path

source_digest = sys.argv[1]
contract_manifest = Path(sys.argv[2]).read_bytes()
digest = hashlib.sha256()
digest.update(b"source\0")
digest.update(source_digest.encode("ascii"))
digest.update(b"\0contract-manifest\0")
digest.update(hashlib.sha256(contract_manifest).digest())
print(digest.hexdigest())
PY
)"
output_dir="${workspace_root}/.workspace/dev-artifacts/rl-sample-pool/${version}/${platform_dir}/${artifact_input_digest}"

if [ -d "${output_dir}" ]; then
    python3 "${tool}" verify \
        --root "${output_dir}" \
        --package rl-sample-pool \
        --version "${version}" \
        --platform "${platform}" \
        --source-digest "${source_digest}" \
        --contract-manifest "${contract_dir}/manifest.json"
    printf '%s\n' "${output_dir}"
    exit 0
fi

mkdir -p "$(dirname "${output_dir}")" "${workspace_root}/.workspace/dev-artifacts"
temp_dir="$(mktemp -d "${workspace_root}/.workspace/dev-artifacts/.tmp-sample-pool.XXXXXX")"
trap 'rm -rf "${temp_dir}"' EXIT
mkdir -p "${temp_dir}/bin" "${temp_dir}/config"
builder_image="rl-training/sample-pool-dev-builder:${version}-${source_digest:0:12}"

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
        cd /source
        bash ./test.sh
        cmake -S /source -B /tmp/sample-build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF \
            -DCONTRACT_CPP_DIR=/contracts/cpp
        cmake --build /tmp/sample-build --parallel
        cp /tmp/sample-build/maze_sample_pool /output/bin/
    ' >&2
cp "${repo_dir}/configs/pool_config.yaml" \
    "${temp_dir}/config/pool_config.yaml"

python3 "${tool}" finalize-component \
    --repo "${repo_dir}" \
    --output "${temp_dir}" \
    --package rl-sample-pool \
    --version "${version}" \
    --platform "${platform}" \
    --source-digest "${source_digest}" \
    --contract-manifest "${contract_dir}/manifest.json"
python3 "${tool}" verify \
    --root "${temp_dir}" \
    --package rl-sample-pool \
    --version "${version}" \
    --platform "${platform}" \
    --source-digest "${source_digest}" \
    --contract-manifest "${contract_dir}/manifest.json"

if [ -e "${output_dir}" ]; then
    echo "development artifact appeared concurrently: ${output_dir}" >&2
    exit 1
fi
mv "${temp_dir}" "${output_dir}"
trap - EXIT
printf '%s\n' "${output_dir}"
