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
    [--skip-python-deps]

This is the public entrypoint for both:
  1. local/manual CLI builds
  2. GitHub Actions builds

By default it:
  1. installs the required Python dependencies
  2. sources the available CANN bashrc
  3. runs msopgen and ./build.sh in the generated operator project
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PYTHON_DEPS="true"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-python-deps)
      INSTALL_PYTHON_DEPS="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --json|--host-dir|--kernel-dir|--framework|--compute-unit|--language|--workspace)
      if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
      fi
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${INSTALL_PYTHON_DEPS}" == "true" ]]; then
  "${SCRIPT_DIR}/scripts/install_cann_python_deps.sh"
fi

"${SCRIPT_DIR}/scripts/build_run_package.sh" "${FORWARD_ARGS[@]}"
