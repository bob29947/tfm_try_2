# Agent Notes

- For local Ray/GPU tokenization benchmarks, avoid `/tmp` on this machine. It is
  often close to full and can introduce object spilling or runtime-env
  bottlenecks.
- Prefer `/dev/shm` for Ray session/temp and spill state. The tokenizer script
  defaults local Ray to `--ray-temp-dir /dev/shm/tfm-ray` and
  `--ray-spill-dir /dev/shm/tfm-ray/spill`; keep those defaults unless shared
  memory is constrained.
- When benchmarking `tfm_repo/scripts/tokenize_splits.py`, run outside the
  sandbox so Ray can see the GPUs. Use `--profile v3-4x-v100` for the audited
  four-GPU target; the profile contains the local CPU/GPU and writer settings.
- Benchmark configuration and timing scope live in
  `tfm_repo/benchmarks/tokenization/`. Keep the normal two-actor pipeline
  defaults deployment-safe rather than copying host-specific settings back
  into the CLI.
