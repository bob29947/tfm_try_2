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
  each, 16 CPUs each, 64 row groups per actor batch, fixed-size binary tensor
  output, `uint16` token IDs, no parquet compression, and no dictionary
  encoding. With actor prewarm and output cleanup excluded from the data
  stopwatch, the measured v3 wall time was 6.49s for train/val/test when
  writing output to `/dev/shm`; normal RAID output measured about 7.07s.
