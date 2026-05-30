# AscendC-CI

This repository contains a GitHub Actions workflow and a manual CLI entrypoint
for building an Ascend custom operator project from:

- operator prototype json
- `op_host/`
- `op_kernel/`

Manual CLI entrypoint:

```bash
./build.sh \
  --json /abs/path/op.json \
  --host-dir /abs/path/op_host \
  --kernel-dir /abs/path/op_kernel \
  --framework pytorch \
  --compute-unit ai_core-ascend910b4
```

If the current environment already has the required Python packages such as
`numpy`, `scipy`, and `protobuf`, you can skip that step:

```bash
./build.sh \
  --json /abs/path/op.json \
  --host-dir /abs/path/op_host \
  --kernel-dir /abs/path/op_kernel \
  --framework pytorch \
  --compute-unit ai_core-ascend910b4 \
  --skip-python-deps
```

Shared internal scripts:

- `scripts/install_cann_python_deps.sh`
- `scripts/build_run_package.sh`

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

- `operator_json`: `demo/pdist_grad/pdist_grad.json`
- `operator_host_dir`: `demo/pdist_grad/op_host`
- `operator_kernel_dir`: `demo/pdist_grad/op_kernel`
- `framework`: `pytorch`
- `compute_unit`: `ai_core-ascend910b4`

Validated manual example:

```bash
./build.sh \
  --json "$(pwd)/demo/pdist_grad/pdist_grad.json" \
  --host-dir "$(pwd)/demo/pdist_grad/op_host" \
  --kernel-dir "$(pwd)/demo/pdist_grad/op_kernel" \
  --framework pytorch \
  --compute-unit ai_core-ascend910b4
```
