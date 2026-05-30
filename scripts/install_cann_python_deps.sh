#!/usr/bin/env bash

set -euo pipefail

set +u
source ~/.bashrc
set -u

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
