# AscendC-CI

This repository contains a GitHub Actions workflow for building an Ascend custom
operator project from:

- operator prototype json
- `op_host/`
- `op_kernel/`

The workflow currently mirrors the verified local sequence:

```bash
./scripts/install_cann_python_deps.sh
./scripts/build_run_package.sh \
  --json /abs/path/op.json \
  --host-dir /abs/path/op_host \
  --kernel-dir /abs/path/op_kernel \
  --framework pytorch \
  --compute-unit ai_core-ascend910b4
```

The build path is the verified:

```bash
source ~/.bashrc
msopgen gen -f pytorch -c ai_core-ascend910b4 -lan cpp -out <output_dir>
./build.sh
```

The GitHub Actions job currently runs in:

```text
ascendai/cann:8.5.2-910b-openeuler24.03-py3.11
```

Local validation in this thread used:

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.2-910b-ubuntu22.04-py3.11
```

and uploads:

- `build_out/custom_opp_*.run`
- the full `build_out/` directory

Manual trigger inputs:

- `operator_json`
- `operator_host_dir`
- `operator_kernel_dir`
- `framework`
- `compute_unit`
- `language`

Current validated example:

- `operator_json`: `DoNotSubmitThisFolder/PdistGrad/pdist_grad.json`
- `operator_host_dir`: `DoNotSubmitThisFolder/PdistGrad/op_host`
- `operator_kernel_dir`: `DoNotSubmitThisFolder/PdistGrad/op_kernel`
- `framework`: `pytorch`
- `compute_unit`: `ai_core-ascend910b4`
