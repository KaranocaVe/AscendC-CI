#!/usr/bin/env bash

set -euo pipefail

source_cann_env() {
  local bashrc=""
  local candidates=()

  if [[ -n "${HOME:-}" ]]; then
    candidates+=("${HOME}/.bashrc")
  fi
  candidates+=(
    "/root/.bashrc"
    "/home/HwHiAiUser/.bashrc"
  )

  set +u
  for bashrc in "${candidates[@]}"; do
    if [[ -f "${bashrc}" ]]; then
      # shellcheck disable=SC1090
      source "${bashrc}"
      set -u
      return 0
    fi
  done
  set -u

  if command -v msopgen >/dev/null 2>&1; then
    return 0
  fi

  echo "failed to source a CANN bashrc from: ${candidates[*]}" >&2
  exit 1
}
