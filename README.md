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

## Preprocess Raw Parquet

Notebook 01 and notebook 02 preprocessing can be run as two scripts. First,
create temporal splits from a raw TabFormer parquet file:

```bash
source .venv/bin/activate
tfm_repo/scripts/create_temporal_splits.py \
  tfm_repo/data/raw/parquet/card_transaction.v1.parquet \
  --overwrite
```

By default, `card_transaction.v1.parquet` writes:

```text
tfm_repo/data/temporal_split_v1/{train,val,test}
```

Then tokenize those splits for NB03:

```bash
export TFM_SPLIT_DIR=$PWD/tfm_repo/data/temporal_split_v1
tfm_repo/scripts/tokenize_splits.py "$TFM_SPLIT_DIR" --overwrite
export TFM_TOKENIZED_DIR=$PWD/tfm_repo/data/tokenized_v1
```

NB03 trains from `tfm_repo/data/tokenized_v1/{train,val,test}`.

For v3, use `card_transaction.v3.parquet`; the default outputs become
`temporal_split_v3` and `tokenized_v3`.

## Development Notes

Push changes to this repo, not to the upstream Ray or TFM repositories. A fresh clone of `tfm_try_2` contains the Ray and notebook source files directly; `scripts/setup_venv.sh` links the installed Ray wheel's Python packages back to the local `ray/python/ray` tree.

If you edit Ray Python code under `ray/python/ray`, rerun the notebooks with the `TFM Ray Dev (.venv)` kernel and the changes should be picked up immediately. If you edit Ray's compiled C++ core, rebuild/install a matching Ray wheel before expecting notebook runs to use those compiled changes.
