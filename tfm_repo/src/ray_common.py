import os
from pathlib import Path

from .tokenization.contract import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MERCHANT_HASH_MODE,
    MERCHANT_HASH_SIZE,
    PAD_TOKEN_ID,
    SEQ_CHUNK_SIZE,
    SEQ_LENGTH,
    SEP_TOKEN_ID,
    UNK_TOKEN_ID,
)

# --------------------------------------------------------------------------- #
# Shared, cross-node storage (NFS on Anyscale). Falls back to a local dir when
# running off-Anyscale so the code is still importable anywhere.
# --------------------------------------------------------------------------- #
_CLUSTER_STORAGE = Path("/mnt/cluster_storage")
_DEFAULT_DATA_ROOT = (_CLUSTER_STORAGE if _CLUSTER_STORAGE.is_dir() else Path.home()) / "tfm_ray"
DATA_ROOT = Path(os.environ.get("TFM_DATA_ROOT", _DEFAULT_DATA_ROOT)).expanduser().resolve()


def _dir_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


RAW_DIR = _dir_from_env("TFM_RAW_DIR", DATA_ROOT / "raw")                       # synthetic / real raw parquet
SPLIT_DIR = _dir_from_env("TFM_SPLIT_DIR", DATA_ROOT / "temporal_split")        # train/val/test parquet
TOKENIZED_DIR = _dir_from_env("TFM_TOKENIZED_DIR", DATA_ROOT / "tokenized")     # NB02 output (parquet, replaces corpus.txt)
MODEL_DIR = _dir_from_env("TFM_MODEL_DIR", DATA_ROOT / "models")               # NB03 checkpoints (HF format)
EMBED_DIR = _dir_from_env("TFM_EMBED_DIR", DATA_ROOT / "embeddings")           # NB04 output (parquet)
OUTPUT_DIR = _dir_from_env("TFM_OUTPUT_DIR", DATA_ROOT / "outputs")            # NB05 plots / metrics

for _d in (RAW_DIR, SPLIT_DIR, TOKENIZED_DIR, MODEL_DIR, EMBED_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Single shared GPU runtime_env. Installed once per worker node, reused by all
# notebooks. Pinned to the exact versions validated on this cluster.
# --------------------------------------------------------------------------- #
GPU_PIP = [
    "torch==2.12.0",
    "transformers==5.9.0",
    "cudf-cu12==25.6.*",
    "cupy-cuda12x",
    "xgboost==3.0.*",
    "scikit-learn",
    "pandas",
    "pyarrow",
]

# Absolute path to the `src` package dir.
_SRC_RAY_DIR = str(Path(__file__).resolve().parent)

# JOB-level env (pass to `ray.init`): ships the src package to every node so
# `import src.tokenizer` works on workers. Code only — the CPU head installs
# nothing here. `py_modules` (local-dir upload) is ONLY valid at the job level.
JOB_RUNTIME_ENV = {
    "py_modules": [_SRC_RAY_DIR],
    "env_vars": {"PIP_EXTRA_INDEX_URL": "https://pypi.nvidia.com"},
}

# PER-OP env (pass to map_batches): installs the GPU wheels on the worker node.
# Merged with the job env, so workers get both the pip deps and src code.
GPU_RUNTIME_ENV = {
    "pip": GPU_PIP,
    "env_vars": {"PIP_EXTRA_INDEX_URL": "https://pypi.nvidia.com"},
}

# Tokenization only needs RAPIDS/cuDF and CuPy.  Keeping this runtime smaller
# avoids pulling training/inference wheels into the preprocessing benchmark.
TOKENIZE_GPU_RUNTIME_ENV = {
    "pip": [
        "cudf-cu12==25.6.*",
        "cupy-cuda12x[ctk]",
        "pyarrow",
    ],
    "env_vars": {"PIP_EXTRA_INDEX_URL": "https://pypi.nvidia.com"},
}

# Ray Train workers inherit the JOB runtime_env, so for training notebooks we
# pass this combined env to `ray.init` (code + pip). The CPU head driver does not
# import torch, so its (cached) install is harmless.
TRAIN_JOB_ENV = {
    "py_modules": [_SRC_RAY_DIR],
    "pip": GPU_PIP,
    "env_vars": {"PIP_EXTRA_INDEX_URL": "https://pypi.nvidia.com"},
}

# Non-tokenization model constants shared across notebooks.
EMB_MAX_LENGTH = 128          # per-transaction encode length for embeddings (NB04)

# Llama decoder config (the ~29M-param foundation model from the original NB03).
MODEL_CONFIG = dict(
    vocab_size=6251,
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2,
    intermediate_size=1408,
    max_position_embeddings=8192,
    rope_theta=500000.0,
    hidden_act="silu",
    rms_norm_eps=1.0e-5,
    attention_dropout=0.0,
    tie_word_embeddings=False,
    bos_token_id=BOS_TOKEN_ID,
    eos_token_id=EOS_TOKEN_ID,
    pad_token_id=PAD_TOKEN_ID,
    merchant_hash_mode=MERCHANT_HASH_MODE,
    merchant_hash_size=MERCHANT_HASH_SIZE,
)
