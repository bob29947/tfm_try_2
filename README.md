# TFM Ray Development Workspace

This repo vendors the two moving pieces together:

- `ray/`: the editable Ray source tree we will modify.
- `tfm_repo/`: the Transaction Foundation Model notebooks that exercise that Ray tree.

There are no submodules required for normal use. Generated data, model checkpoints, notebook outputs, Ray build products, and the local virtualenv stay ignored.

## Local Setup

Create and wire the local environment:

```bash
git clone https://github.com/bob29947/tfm_try_2.git
cd tfm_try_2
scripts/setup_venv.sh
source .venv/bin/activate
```

The setup script installs the `ray-3.0.0.dev0` wheel for the vendored Ray commit, installs the TFM head-node notebook dependencies, then runs Ray's `python/ray/setup-dev.py` so Ray Python package directories resolve to this workspace's `ray/python/ray` source tree.

To check the wiring:

```bash
python -c "import ray; print(ray.__version__); print(ray.__file__)"
```

Use the `TFM Ray Dev (.venv)` Jupyter kernel when running the notebooks.

## Development Notes

Push changes to this repo, not to the upstream Ray or TFM repositories. A fresh clone of `tfm_try_2` contains the Ray and notebook source files directly; `scripts/setup_venv.sh` links the installed Ray wheel's Python packages back to the local `ray/python/ray` tree.

If you edit Ray Python code under `ray/python/ray`, rerun the notebooks with the `TFM Ray Dev (.venv)` kernel and the changes should be picked up immediately. If you edit Ray's compiled C++ core, rebuild/install a matching Ray wheel before expecting notebook runs to use those compiled changes.
