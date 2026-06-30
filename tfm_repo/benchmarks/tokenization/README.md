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

## Four-node AWS S3-to-S3 benchmark

The cloud launcher runs one warmup and one measured pipeline for `original`,
`tuned`, then `fast` on exactly four `g6.4xlarge` nodes in `us-west-2d`:

```bash
./tfm_repo/benchmarks/tokenization/run_aws_benchmark.sh
```

It idempotently copies and verifies the v3 inputs under
`s3://bobbwang-tfm-gpu-e2e/tfm_try_2/input/temporal_split_v3`, launches a
run-named cluster, requires four L4s and a working KvikIO S3 range read on each
node, retains only validated measured outputs, downloads all reports and logs,
and verifies that no tagged EC2 instances remain.

By default, local artifacts are written to
`benchmarks/tokenization/results/aws/<run-id>/` and measured objects to
`s3://bobbwang-tfm-gpu-e2e/tfm_try_2/benchmarks/<run-id>/outputs/`.
`TFM_BENCH_RUN_ID`, `TFM_BENCH_INPUT_URI`, and `TFM_BENCH_OUTPUT_ROOT` can
override those defaults. Do not reuse a run ID; the harness refuses an existing
measured-output prefix.

### Focused fast-path cloud sweep

To tune the direct S3 path and immediately run the winning end-to-end arm on a
fresh four-node cluster:

```bash
./tfm_repo/benchmarks/tokenization/run_aws_fast_sweep.sh
```

The default screen compares a focused twelve-configuration matrix covering
16/32/48 row-group batches, 8/16/32/64 KvikIO threads, 0.5/1/2/4/16 MiB
KvikIO tasks, and an eight-writer variant. Each configuration gets fresh actors,
a separately timed one-row-group warmup, and a disposable train trial capped at
32 row groups per key range. The fastest four safe configurations are then run
on complete train with fresh actors before the winner is used for the validated
train/val/test arm. The final arm enables host-write/GPU-preparation overlap by
default; set `TFM_SWEEP_FINAL_OVERLAP=0` to retain split barriers.

Screen, confirmation, and failed outputs under `<run-id>/tuning/` are always
deleted. The final `outputs/fast/` is retained only if its full validation and
three commit markers pass. Local `fast-sweep.json` and `fast-sweep.md` record
runner/process/read/write critical paths, actor aggregates, GPU memory
high-water, realized KvikIO settings, the selection evidence, and the final
pipeline result. `TFM_SWEEP_SCREEN_ROW_GROUPS` and
`TFM_SWEEP_CONFIRMATION_TOP` control the bounded screen and confirmation count.

### Fresh-actor fast measurement

After freezing a fast configuration, run it on a fresh four-node cluster
without a preceding actor warmup or tuning sweep:

```bash
TFM_BENCH_COLD_FAST=1 ./tfm_repo/benchmarks/tokenization/run_aws_benchmark.sh
```

This focused mode uses the selected 32-row-group, 32-connection, 16 MiB KvikIO
task configuration. Its pipeline clock starts before S3 footer planning and
creation/readiness of four fresh GPU actors, and ends after all train/val/test
writes, exact inventories, and durable commit markers. The result records
`actor_create_and_ready_s`, `fresh_actors_created=true`, and
`actor_lifecycle=fresh_actor_inclusive`. Cluster provisioning, the mandatory
per-node smoke gate, actor teardown, and exhaustive post-run validation remain
separately timed.
