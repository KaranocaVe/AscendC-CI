#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=source_cann_env.sh
source "${SCRIPT_DIR}/source_cann_env.sh"

source_cann_env

ensure_python_module() {
  local import_name="$1"
  shift
  if python3 -c "import ${import_name}" >/dev/null 2>&1; then
    return 0
  fi
  pip3 install "$@"
}

ensure_python_module wheel wheel==0.38.4
ensure_python_module yaml pyyaml==6.0.1
ensure_python_module setuptools setuptools==67.4.0
ensure_python_module numpy numpy==1.26.4
ensure_python_module attrs attrs
ensure_python_module decorator decorator
ensure_python_module sympy sympy
ensure_python_module psutil psutil
ensure_python_module scipy scipy
ensure_python_module google.protobuf protobuf
