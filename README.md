# TFM Ray Development Workspace

This wrapper repo keeps the two moving pieces together:

- `ray/`: the editable Ray checkout we will modify.
- `tfm_repo/`: the Transaction Foundation Model notebooks that exercise that Ray checkout.

The top-level Git repo tracks wrapper files plus the two nested repos as submodule pointers. Generated data, model checkpoints, notebook outputs, Ray build products, and the local virtualenv stay ignored.

## Local Setup

Create and wire the local environment:

```bash
scripts/setup_venv.sh
source .venv/bin/activate
```

The setup script installs the latest `ray-3.0.0.dev0` wheel for the active Python, installs the TFM head-node notebook dependencies, then runs Ray's `python/ray/setup-dev.py` so Ray Python package directories resolve to this workspace's `ray/python/ray` source tree.

To check the wiring:

```bash
python -c "import ray; print(ray.__version__); print(ray.__file__)"
```

Use the `TFM Ray Dev (.venv)` Jupyter kernel when running the notebooks.
