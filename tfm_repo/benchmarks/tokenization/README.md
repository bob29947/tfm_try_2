# Tokenization benchmark profiles

Benchmark profiles keep host-specific throughput settings separate from the
normal pipeline defaults. They supply command-line defaults only; explicit
flags always win, and the fully normalized settings are recorded in the output
manifest.

The audited four-V100 v3 result uses `v3-4x-v100`:

```bash
tfm_repo/scripts/tokenize_splits.py \
  tfm_repo/data/temporal_split_v3 \
  --output-dir tfm_repo/data/tokenized_v3_bench \
  --profile v3-4x-v100 \
  --ray-address local \
  --overwrite
```

Run this profile only on a host with at least four visible GPUs and 64 Ray
CPUs. The CLI checks total cluster resources before creating long-lived actors.
This fail-fast check applies to fixed local Ray runtimes; remote clusters remain
free to autoscale after actor demand is submitted.
The measured approximately 3.03-second data path uses prewarmed actors and a
page-cached source, and excludes actor/Ray startup, output cleanup, provenance
hashing, and durable `fsync`. See `../../TOKENIZATION_VALIDATION.md` for the
full timing and quality record.
