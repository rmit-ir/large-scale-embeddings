#!/bin/sh
set -eu

log() {
  printf '%s\n' "torch-cuda-entrypoint: $*"
}

detect_cuda_version() {
  ver=""

  if command -v nvidia-smi >/dev/null 2>&1; then
    ver="$(nvidia-smi 2>/dev/null | awk -F 'CUDA Version: ' '/CUDA Version/ {print $2}' | awk '{print $1}' | head -n1)"
  fi

  if [ -z "${ver}" ] && [ -f /usr/local/cuda/version.json ]; then
    ver="$(python3 - <<'PY'
import json
try:
    with open("/usr/local/cuda/version.json", "r", encoding="utf-8") as handle:
        data = json.load(handle)
    cuda = data.get("cuda", {})
    print(cuda.get("version") or data.get("version") or "")
except Exception:
    pass
PY
)"
  fi

  if [ -z "${ver}" ] && [ -f /usr/local/cuda/version.txt ]; then
    ver="$(sed -n 's/.*CUDA Version *//p' /usr/local/cuda/version.txt | awk '{print $1}' | head -n1)"
  fi

  printf '%s' "${ver}"
}

cuda_ver="$(detect_cuda_version || true)"
if [ -n "${cuda_ver}" ]; then
  case "${cuda_ver}" in
    *.*)
      cuda_ver="$(printf '%s' "${cuda_ver}" | awk -F. '{print $1"."$2}')"
      ;;
    *)
      cuda_ver=""
      ;;
  esac
fi

if [ -z "${cuda_ver}" ]; then
  log "No CUDA runtime detected; skipping torch CUDA matching."
  exec "$@"
fi

cu_tag="$(printf '%s' "${cuda_ver}" | tr -d '.')"

torch_info="$(python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("torch")
if spec is None:
    print("NOT_INSTALLED")
    raise SystemExit(0)

import torch

print(torch.version.cuda or "")
PY
)" || true

torch_cuda="$(printf '%s\n' "${torch_info}" | sed -n '1p')"

if [ "${torch_info}" = "NOT_INSTALLED" ] || [ -z "${torch_info}" ]; then
  log "Torch not installed; installing for CUDA ${cuda_ver} (cu${cu_tag})."
  if ! python3 -m pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/cu${cu_tag}" torch; then
    log "Torch install failed for cu${cu_tag}; continuing without changes."
  fi
  exec "$@"
fi

if [ "${torch_cuda}" = "${cuda_ver}" ]; then
  log "Torch CUDA ${torch_cuda} matches detected CUDA ${cuda_ver}."
else
  log "Torch CUDA ${torch_cuda:-none} does not match detected CUDA ${cuda_ver}; reinstalling latest torch for cu${cu_tag}."
  if ! python3 -m pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/cu${cu_tag}" torch; then
    log "Torch reinstall failed for cu${cu_tag}; keeping existing torch."
  fi
fi

exec "$@"
