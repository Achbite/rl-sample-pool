#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"
exec bash "${repo_dir}/build_artifact.sh"
