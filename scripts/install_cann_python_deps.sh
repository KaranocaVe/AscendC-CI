#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=source_cann_env.sh
source "${SCRIPT_DIR}/source_cann_env.sh"

source_cann_env

pip3 install wheel==0.38.4
pip3 install pyyaml==6.0.1
pip3 install setuptools==67.4.0
pip3 install numpy==1.26.4
pip3 install attrs
pip3 install decorator
pip3 install sympy
pip3 install psutil
pip3 install scipy
pip3 install protobuf
