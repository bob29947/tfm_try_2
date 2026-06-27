#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
RAY_VERSION="${RAY_VERSION:-3.0.0.dev0}"
RAY_WHEEL_COMMIT="${RAY_WHEEL_COMMIT:-85bdbaa879fc6a78c5d80256bb677fe2fb297430}"

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

RAY_WHEEL_NAME="ray-${RAY_VERSION}-${PY_TAG}-${PY_TAG}-${PLATFORM_TAG}.whl"
DEFAULT_RAY_WHEEL_URL="https://s3-us-west-2.amazonaws.com/ray-wheels/master/${RAY_WHEEL_COMMIT}/${RAY_WHEEL_NAME}"
LATEST_RAY_WHEEL_URL="https://s3-us-west-2.amazonaws.com/ray-wheels/latest/${RAY_WHEEL_NAME}"
RAY_WHEEL_URL="${RAY_WHEEL_URL:-$DEFAULT_RAY_WHEEL_URL}"

"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel
if ! "$VENV_DIR/bin/python" -m pip install "ray[data,train] @ ${RAY_WHEEL_URL}"; then
  if [[ "$RAY_WHEEL_URL" == "$DEFAULT_RAY_WHEEL_URL" ]]; then
    echo "Commit-specific Ray wheel was unavailable; falling back to latest nightly." >&2
    "$VENV_DIR/bin/python" -m pip install "ray[data,train] @ ${LATEST_RAY_WHEEL_URL}"
  else
    exit 1
  fi
fi
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
