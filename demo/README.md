# Demo

This directory contains public sample inputs for manual CLI usage and GitHub
Actions validation.

Current sample:

- `demo/pdist_grad/pdist_grad.json`
- `demo/pdist_grad/op_host/`
- `demo/pdist_grad/op_kernel/`

Manual build example:

```bash
./build.sh \
  --json "$(pwd)/demo/pdist_grad/pdist_grad.json" \
  --host-dir "$(pwd)/demo/pdist_grad/op_host" \
  --kernel-dir "$(pwd)/demo/pdist_grad/op_kernel" \
  --framework pytorch \
  --compute-unit ai_core-ascend910b4
```
