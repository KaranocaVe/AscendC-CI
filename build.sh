#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./build.sh \
    --json /abs/path/op.json \
    --host-dir /abs/path/op_host \
    --kernel-dir /abs/path/op_kernel \
    [--framework pytorch] \
    [--compute-unit ai_core-ascend910b4] \
    [--language cpp] \
    [--workspace /tmp/op-build] \
    [--image swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8] \
    [--skip-python-deps]

This is the public entrypoint for local/manual Docker builds.

By default it:
  1. runs the build inside the fixed CANN Docker image
  2. installs the required Python dependencies in the container
  3. sources the available CANN bashrc in the container
  4. runs msopgen and ./build.sh in the generated operator project
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_PWD="$(pwd)"
IMAGE="swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8"
INSTALL_PYTHON_DEPS="true"
JSON_PATH=""
HOST_DIR=""
KERNEL_DIR=""
FRAMEWORK="pytorch"
COMPUTE_UNIT="ai_core-ascend910b4"
LANGUAGE="cpp"
WORKSPACE="/tmp/op-build"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
      fi
      JSON_PATH="$2"
      shift 2
      ;;
    --host-dir)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
      fi
      HOST_DIR="$2"
      shift 2
      ;;
    --kernel-dir)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
      fi
      KERNEL_DIR="$2"
      shift 2
      ;;
    --framework)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
      fi
      FRAMEWORK="$2"
      shift 2
      ;;
    --compute-unit)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
      fi
      COMPUTE_UNIT="$2"
      shift 2
      ;;
    --language)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
      fi
      LANGUAGE="$2"
      shift 2
      ;;
    --workspace)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
      fi
      WORKSPACE="$2"
      shift 2
      ;;
    --image)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
      fi
      IMAGE="$2"
      shift 2
      ;;
    --skip-python-deps)
      INSTALL_PYTHON_DEPS="false"
      shift
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

if [[ -z "${JSON_PATH}" || -z "${HOST_DIR}" || -z "${KERNEL_DIR}" ]]; then
  echo "--json, --host-dir, and --kernel-dir are required" >&2
  usage >&2
  exit 1
fi

if [[ "${JSON_PATH}" != /* ]]; then
  JSON_PATH="${HOST_PWD}/${JSON_PATH}"
fi
if [[ "${HOST_DIR}" != /* ]]; then
  HOST_DIR="${HOST_PWD}/${HOST_DIR}"
fi
if [[ "${KERNEL_DIR}" != /* ]]; then
  KERNEL_DIR="${HOST_PWD}/${KERNEL_DIR}"
fi
if [[ "${WORKSPACE}" != /* ]]; then
  WORKSPACE="${HOST_PWD}/${WORKSPACE}"
fi

mkdir -p "${WORKSPACE}"

MOUNTS=()
SEEN_MOUNTS=()

add_mount() {
  local src="$1"
  local dst="$2"
  local key="${src}:${dst}"
  local seen=""
  for seen in "${SEEN_MOUNTS[@]-}"; do
    if [[ "${seen}" == "${key}" ]]; then
      return 0
    fi
  done
  SEEN_MOUNTS+=("${key}")
  MOUNTS+=(-v "${src}:${dst}")
}

add_mount "${SCRIPT_DIR}" /workspace
add_mount "${SCRIPT_DIR}" "${SCRIPT_DIR}"
add_mount "$(dirname "${JSON_PATH}")" "$(dirname "${JSON_PATH}")"
add_mount "${HOST_DIR}" "${HOST_DIR}"
add_mount "${KERNEL_DIR}" "${KERNEL_DIR}"
add_mount "${WORKSPACE}" /host-workspace

docker run --rm \
  "${MOUNTS[@]}" \
  -w /workspace \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "
    set -euo pipefail
    container_workspace='/tmp/codex-op-build'
    rm -rf \"\${container_workspace}\"
    if [[ \"${INSTALL_PYTHON_DEPS}\" == \"true\" ]]; then
      /workspace/scripts/install_cann_python_deps.sh
    fi
    /workspace/scripts/build_run_package.sh \
      --json '${JSON_PATH}' \
      --host-dir '${HOST_DIR}' \
      --kernel-dir '${KERNEL_DIR}' \
      --framework '${FRAMEWORK}' \
      --compute-unit '${COMPUTE_UNIT}' \
      --language '${LANGUAGE}' \
      --workspace \"\${container_workspace}\"
    rm -rf /host-workspace/*
    cp -R \"\${container_workspace}\"/. /host-workspace/
  "
