#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
contract_root="${workspace_root}/.workspace/artifacts/rl-contracts"
artifact_root="${workspace_root}/.workspace/artifacts/rl-sample-pool"
version="$(tr -d '[:space:]' < "${repo_dir}/VERSION")"
source "${repo_dir}/artifact_versions.env"
source_commit="$(git -C "${repo_dir}" rev-parse --short=12 HEAD 2>/dev/null || printf 'unborn')"
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
contract_dir="${contract_root}/${RL_CONTRACTS_VERSION}/${platform_dir}"

if ! test -f "${contract_dir}/manifest.json" ||
   ! test -f "${contract_dir}/cpp/maze.pb.cc"; then
    echo "rl-contracts artifact is missing; run: (cd ../rl-contracts && bash build_artifact.sh)" >&2
    exit 1
fi

source_sha256="$(
    python3 - "${repo_dir}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(root.rglob("*")):
    if not path.is_file() or ".git" in path.parts or "build" in path.parts:
        continue
    digest.update(str(path.relative_to(root)).encode())
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
)"
source_id="${source_commit}"
if test -n "$(git -C "${repo_dir}" status --porcelain=v1)"; then
    source_id="${source_commit}-dirty-${source_sha256:0:12}"
fi

output_dir="${artifact_root}/${version}/${platform_dir}"
temp_dir="${workspace_root}/.workspace/artifacts/.tmp-sample-pool-$$"
builder_image="rl-training/sample-pool-builder:${version}"

if test -d "${output_dir}"; then
    if PACKAGE_VERSION="${version}" \
       SOURCE_ID="${source_id}" \
       SOURCE_SHA256="${source_sha256}" \
       PLATFORM="${platform}" \
       CONTRACT_MANIFEST="${contract_dir}/manifest.json" \
       python3 - "${output_dir}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
contract = json.loads(Path(os.environ["CONTRACT_MANIFEST"]).read_text(encoding="utf-8"))
expected = {
    "package": "rl-sample-pool",
    "version": os.environ["PACKAGE_VERSION"],
    "source_id": os.environ["SOURCE_ID"],
    "source_sha256": os.environ["SOURCE_SHA256"],
    "platform": os.environ["PLATFORM"],
}
if any(manifest.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
expected_contract = {
    "version": contract["version"],
    "source_id": contract["source_id"],
    "source_sha256": contract["source_sha256"],
}
if manifest.get("contract") != expected_contract:
    raise SystemExit(1)
for relative, checksum in manifest.get("files", {}).items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise SystemExit(1)
PY
    then
        printf '%s\n' "${output_dir}"
        exit 0
    fi
    echo "sample-pool artifact version already exists with different content: ${output_dir}" >&2
    echo "remove that generated artifact explicitly or increment VERSION" >&2
    exit 1
fi

mkdir -p "${temp_dir}/bin" "${temp_dir}/config" "$(dirname "${output_dir}")"
trap 'rm -rf "${temp_dir}"' EXIT

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
            -DBUILD_TESTING=ON \
            -DCONTRACT_CPP_DIR=/contracts/cpp
        cmake --build /tmp/sample-build --parallel
        ctest --test-dir /tmp/sample-build --output-on-failure
        cp /tmp/sample-build/maze_sample_distributor /output/bin/
    '

cp "${repo_dir}/configs/distributor_config.yaml" \
    "${temp_dir}/config/distributor_config.yaml"

PACKAGE_VERSION="${version}" \
SOURCE_COMMIT="${source_commit}" \
SOURCE_ID="${source_id}" \
SOURCE_SHA256="${source_sha256}" \
PLATFORM="${platform}" \
CONTRACT_MANIFEST="${contract_dir}/manifest.json" \
OUTPUT_DIR="${temp_dir}" \
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
contract = json.loads(Path(os.environ["CONTRACT_MANIFEST"]).read_text())
files = {}
for path in sorted(root.rglob("*")):
    if path.is_file():
        files[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

manifest = {
    "schema_version": 1,
    "package": "rl-sample-pool",
    "version": os.environ["PACKAGE_VERSION"],
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_id": os.environ["SOURCE_ID"],
    "source_sha256": os.environ["SOURCE_SHA256"],
    "platform": os.environ["PLATFORM"],
    "abi": "cxx17-grpc-debian13",
    "contract": {
        "version": contract["version"],
        "source_id": contract["source_id"],
        "source_sha256": contract["source_sha256"],
    },
    "files": files,
}
(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

mv "${temp_dir}" "${output_dir}"
trap - EXIT
printf '%s\n' "${output_dir}"
