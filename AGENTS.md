# Agent Notes

- For local Ray/GPU tokenization benchmarks, avoid `/tmp` on this machine. It is
  often close to full and can introduce object spilling or runtime-env
  bottlenecks.
- Prefer `/dev/shm` for Ray session/temp and spill state. The tokenizer script
  defaults local Ray to `--ray-temp-dir /dev/shm/tfm-ray` and
  `--ray-spill-dir /dev/shm/tfm-ray/spill`; keep those defaults unless shared
  memory is constrained.
- When benchmarking `tfm_repo/scripts/tokenize_splits.py`, run outside the
  sandbox so Ray can see the GPUs, and use `--local-num-cpus 64
  --local-num-gpus 4` for the v3 target run.
- Best known v3 GPU-parquet benchmark on this host uses 4 actors at 1 GPU
  each, 16 CPUs each, 64 row groups per actor batch, 12 parquet writer threads
  per actor, 128 MiB output shards, fixed-size binary tensor output, `uint16`
  token IDs, no parquet compression, and no dictionary encoding. With actor
  and CUDA-kernel prewarm plus output cleanup excluded from the data stopwatch,
  the best measured v3 data-path time was 4.54s for train/val/test on normal
  RAID output; repeated runs with the same defaults measured 4.54-5.00s. This
  stopwatch is prewarmed and excludes Ray startup, output cleanup, provenance
  hashing, and durable fsync.
