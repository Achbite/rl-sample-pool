#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/configs/distributor_config.yaml}"

if [ ! -x "${SCRIPT_DIR}/build/maze_sample_distributor" ]; then
    echo "[错误] 未找到可执行文件，请先运行: bash build.sh"
    exit 1
fi

exec "${SCRIPT_DIR}/build/maze_sample_distributor" "${CONFIG_PATH}"
