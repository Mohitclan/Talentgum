#!/usr/bin/env bash
# One-step local setup. Handles the quirks found on macOS: a Homebrew Python whose venv cannot
# install pip, the missing Intel-Mac wheel for the newest llvmlite, and Basic Pitch's old pins.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || { echo "Python 3.12 is required (PyTorch/Transkun do not support 3.13+ yet)."; exit 1; }
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg is not on PATH; install it (macOS: brew install ffmpeg)."

if "$PY" -m venv .venv 2>/dev/null && [ -x .venv/bin/pip ]; then
  PIP=(.venv/bin/python -m pip)
else
  echo "venv could not bootstrap pip; using the interpreter's own pip to install into .venv"
  rm -rf .venv
  "$PY" -m venv --without-pip .venv
  PIP=("$PY" -m pip --python .venv/bin/python)
fi

if [ "$(uname -s)-$(uname -m)" = "Darwin-x86_64" ]; then
  "${PIP[@]}" install -q --only-binary=llvmlite "llvmlite<0.45" "numba<0.62"
fi
"${PIP[@]}" install -q -e ".[dev,transkun,service]"

# optional lower-bound baseline: Basic Pitch with its bundled ONNX model, without TensorFlow
if "${PIP[@]}" install -q --no-deps basic-pitch && \
   "${PIP[@]}" install -q --only-binary=:all: "numpy<2" onnxruntime resampy; then
  echo "Basic Pitch installed (ONNX runtime)"
else
  echo "Basic Pitch skipped (optional)"
fi

# some macOS setups hide the editable-install .pth file; add a plain one if `import ras` fails
if ! (cd /tmp && "$PWD/.venv/bin/python" -c "import ras" 2>/dev/null); then
  SITE="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
  echo "$PWD" > "$SITE/ras_dev.pth"
fi

.venv/bin/python -m pytest -p no:warnings 2>&1 | tail -1
echo "Done. Try: .venv/bin/ras serve   (then open http://127.0.0.1:8000/ui/)"
