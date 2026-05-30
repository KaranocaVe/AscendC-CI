# AscendC-CI

This repository contains a Docker-based CLI entrypoint for building an Ascend
custom operator project from:

- operator prototype json
- `op_host/`
- `op_kernel/`

Default build image:

```text
swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8
```

Manual CLI entrypoint:

```bash
./build.sh \
  --json /abs/path/op.json \
  --host-dir /abs/path/op_host \
  --kernel-dir /abs/path/op_kernel \
  --framework pytorch \
  --compute-unit ai_core-ascend910b4
```

If the selected Docker image already has the required Python packages such as
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

Container-internal build path:

```bash
source ~/.bashrc
msopgen gen -f pytorch -c ai_core-ascend910b4 -lan cpp -out <output_dir>
./build.sh
```

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

If you need to override the image:

```bash
./build.sh \
  --json "$(pwd)/demo/pdist_grad/pdist_grad.json" \
  --host-dir "$(pwd)/demo/pdist_grad/op_host" \
  --kernel-dir "$(pwd)/demo/pdist_grad/op_kernel" \
  --framework pytorch \
  --compute-unit ai_core-ascend910b4 \
  --image your.registry.example/cann:tag
```
