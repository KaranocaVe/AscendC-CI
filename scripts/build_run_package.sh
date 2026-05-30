#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build_run_package.sh \
    --json /abs/path/op.json \
    --host-dir /abs/path/op_host \
    --kernel-dir /abs/path/op_kernel \
    [--framework pytorch] \
    [--compute-unit ai_core-ascend910b4] \
    [--language cpp] \
    [--workspace /tmp/op-build]

This script assumes:
  1. The current shell can `source ~/.bashrc` to expose CANN tools.
  2. The container or host already has the required CANN runtime.
  3. Python dependencies are installed before `./build.sh` runs.
EOF
}

require_arg() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "missing required argument: $name" >&2
    usage >&2
    exit 1
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=source_cann_env.sh
source "${SCRIPT_DIR}/source_cann_env.sh"

JSON_PATH=""
HOST_DIR=""
KERNEL_DIR=""
FRAMEWORK="pytorch"
COMPUTE_UNIT="ai_core-ascend910b4"
LANGUAGE="cpp"
WORKSPACE="${REPO_ROOT}/.build/op-project"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_PATH="${2:-}"
      shift 2
      ;;
    --host-dir)
      HOST_DIR="${2:-}"
      shift 2
      ;;
    --kernel-dir)
      KERNEL_DIR="${2:-}"
      shift 2
      ;;
    --framework)
      FRAMEWORK="${2:-}"
      shift 2
      ;;
    --compute-unit)
      COMPUTE_UNIT="${2:-}"
      shift 2
      ;;
    --language)
      LANGUAGE="${2:-}"
      shift 2
      ;;
    --workspace)
      WORKSPACE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_arg "--json" "$JSON_PATH"
require_arg "--host-dir" "$HOST_DIR"
require_arg "--kernel-dir" "$KERNEL_DIR"

if [[ ! -f "$JSON_PATH" ]]; then
  echo "json file not found: $JSON_PATH" >&2
  exit 1
fi
if [[ ! -d "$HOST_DIR" ]]; then
  echo "host dir not found: $HOST_DIR" >&2
  exit 1
fi
if [[ ! -d "$KERNEL_DIR" ]]; then
  echo "kernel dir not found: $KERNEL_DIR" >&2
  exit 1
fi

rm -rf "$WORKSPACE"
mkdir -p "$WORKSPACE"

source_cann_env

MSOPGEN_ARGS=(
  -i "$JSON_PATH"
  -c "$COMPUTE_UNIT"
  -lan "$LANGUAGE"
  -out "$WORKSPACE"
)

if [[ -n "$FRAMEWORK" ]]; then
  MSOPGEN_ARGS+=(-f "$FRAMEWORK")
fi

msopgen gen "${MSOPGEN_ARGS[@]}"

cp -R "${HOST_DIR}/." "${WORKSPACE}/op_host/"
cp -R "${KERNEL_DIR}/." "${WORKSPACE}/op_kernel/"

(
  cd "$WORKSPACE"
  ./build.sh
)

echo "workspace=${WORKSPACE}"
echo "build_out=${WORKSPACE}/build_out"
find "${WORKSPACE}/build_out" -maxdepth 1 -type f -name '*.run' -print
