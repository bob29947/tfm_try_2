#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
RAY_VERSION="${RAY_VERSION:-3.0.0.dev0}"

"$PYTHON_BIN" -m venv "$VENV_DIR"

PY_TAG="$("$VENV_DIR/bin/python" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)
    PLATFORM_TAG="manylinux2014_x86_64"
    ;;
  aarch64|arm64)
    PLATFORM_TAG="manylinux2014_aarch64"
    ;;
  *)
    echo "Unsupported architecture for Ray wheel URL: $ARCH" >&2
    exit 1
    ;;
esac

RAY_WHEEL_URL="${RAY_WHEEL_URL:-https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-${RAY_VERSION}-${PY_TAG}-${PY_TAG}-${PLATFORM_TAG}.whl}"

"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel
"$VENV_DIR/bin/python" -m pip install "ray[data,train] @ ${RAY_WHEEL_URL}"
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/tfm_repo/requirements.txt"

(
  cd "$ROOT_DIR/ray"
  "$VENV_DIR/bin/python" python/ray/setup-dev.py -y
)

"$VENV_DIR/bin/python" -m ipykernel install --user \
  --name tfm-ray-dev \
  --display-name "TFM Ray Dev (.venv)"

"$VENV_DIR/bin/python" - <<'PY'
import ray

print("Ray environment ready")
print("version:", ray.__version__)
print("module :", ray.__file__)
PY
