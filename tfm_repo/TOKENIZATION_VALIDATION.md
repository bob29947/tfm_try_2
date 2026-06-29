# Tokenization performance and quality validation

This note records the local V100 validation for branch
`codex/tokenize-v3-5s`. Generated corpora, checkpoints, and result JSON files
remain under the ignored `data/` directory; source temporal splits were read
only and were not modified.

## V3 performance and output identity

The final normal-RAID run used Ray with four one-GPU actors, 16 CPUs per actor,
64 row groups per actor batch, 12 writer threads, 128 MiB output shards,
fixed-size-binary `uint16` tensors, no compression, and no dictionary encoding.

- Original normal-RAID time: **7.07 s**
- Previous optimized normal-RAID time: **4.54 s**
- Final normal-RAID time: **3.03 s median** (**2.98–3.08 s**)
- Improvement from the original: **2.33x** (**57.1% less time**)
- Output counts: train 614,874; validation 81,993; test 82,794

The final actor initializes an 8 GiB RMM device-memory pool before importing
cuDF and routes CuPy allocations through it. This replaces repeated serialized
CUDA allocation/free calls with standard allocator reuse; every read,
tokenization, sort, sequence construction, host transfer, and normal-RAID
write remains in the timed path.

The key-range planner and long-lived actor runner are application code in
`src/gpu_parquet.py`. Ray supplies scheduling and GPU isolation, while the TFM
pipeline owns its shared-filesystem, integral-key filtering, output commit, and
RMM policies. Fixed-size-binary outputs are read through Ray's existing
`tensor_column_schema` option rather than a custom Parquet datasource hook.

The 3.03 s median is from two final-code runs using the same prewarmed,
page-cached, buffered data-path stopwatch used for the 7.07 s baseline. It
excludes Ray/actor startup, output cleanup, post-run provenance hashing, and
durable `fsync`. The final audited run measured 2.98 s for that data path,
22.98 s for SHA-256 provenance work, and 34.56 s for the process including Ray
startup. A same-day run immediately before the allocator change measured
4.68 s with otherwise identical arguments.

After moving the runner out of Ray and switching consumers to Ray's public
tensor schema option, a fresh normal-RAID audit measured **3.02 s** for the
same data path, 22.71 s for provenance, and 33.92 s end to end. The source
manifest, sequence counts, and all 56 output files (paths, rows, bytes, and
SHA-256) remained identical.

Logical tensor bytes match the previously verified fast v3 corpus for every
sequence:

| Split | Rows | SHA-256 |
| --- | ---: | --- |
| train | 614,874 | `3abe70bb0be54b279af65ce29b4bb594e7af6eb80ba7a9aea41cde7b26259032` |
| val | 81,993 | `a7e422d6e72995d624248d7f51a32da48624b9ec4cbc737eeef08536590e857f` |
| test | 82,794 | `3f08c2562dbd7e1a3e20d6df77999b455cdc2fdce8e915eca06985c88e8841a1` |

The run manifest is
`data/tokenized_v3_sub4_bench/_tokenization_manifest.json`. Its source-file
records and every output file's path, row count, byte count, and SHA-256 match
the previous verified manifest exactly.

## Merchant mapping

The fast path maps the dataset's opaque signed integer merchant ID to
`abs(id) % 2000`. NB02 and NB04 now use and record the same mapping. The old
cleaned-string cuDF hash remains available as `string_hash` for legacy
comparison.

A direct GPU check on 12,288 real v1 transactions found zero merchant-token
mismatches between `GPUTokenizer(integer_mod)` and
`FastParquetSplitTokenizer`; both produced `12 + abs(id) % 2000`. A broader v3
distribution check occupied all 2,000 buckets with normalized entropy 0.9973.

## Paired language-model proxy

Two 400-step, fixed-seed models were trained with Ray Train from the same model
initialization and the same paired v1 sequences. Pairing masked only merchant
positions. There were 64,334 exact training pairs and all 9,739 validation
pairs. One of 64,335 training sequences was excluded because two transactions
with the same timestamp appeared in opposite order between engines.

| Metric | Legacy string hash | Fast integer modulo | Fast - legacy |
| --- | ---: | ---: | ---: |
| Median last-25 train CE | 1.06921 | 1.05895 | -0.01026 |
| Validation CE | 1.13341 | 1.11919 | -0.01422 |
| Merchant-position validation CE | 4.85859 | 4.90414 | +0.04555 |

The predeclared proxy limits were +0.03 validation CE and +0.10 merchant CE;
the fast mapping passed both. Both arms used FP16 on V100, skipped zero
optimizer steps, and had the same initial-parameter BLAKE2b hash. Full results
are in `data/downstream_validation/paired_v1_400.json`.

## Paired fraud-model sensitivity check

The checkpoint comparison used 250,000 fixed-seed training transactions,
102,109 validation transactions, and 102,724 test transactions. It retained
all 2,109 validation and 2,724 test fraud cases, sampled normals by fixed row
ordinal, and applied population-prevalence weights to evaluation metrics.
Each arm used its matching checkpoint and merchant mapping. Row-ID hashes were
identical through raw sampling, tokenization, and embedding extraction.

With the notebook's early stopping, the very-low-learning-rate combined models
stopped after only four to five trees and produced an unstable, contradictory
AUPRC result. That result is retained in
`data/downstream_validation/fraud_ab_v1.json`; it is not used as evidence of
equivalence.

The fixed-estimator-budget sensitivity run removed that premature-stop
confound while retaining the notebook's tree parameters:

| Test metric | Legacy | Fast | Fast - legacy |
| --- | ---: | ---: | ---: |
| Embedding-only AUROC | 0.93392 | 0.94985 | +0.01593 |
| Embedding-only AUPRC | 0.01938 | 0.02943 | +0.01005 |
| Combined AUROC | 0.98544 | 0.98574 | +0.00031 |
| Combined AUPRC | 0.06389 | 0.07087 | +0.00698 |

Across 500 paired user-cluster bootstrap samples, the 95% interval for the
combined AUROC difference was `[-0.00020, +0.00080]`, and the interval for the
fast/legacy AUPRC ratio was `[0.965, 1.230]`. These pass the predeclared
noninferiority limits of -0.005 AUROC and 0.90x AUPRC. Full results are in
`data/downstream_validation/fraud_ab_v1_fixed_budget.json`.

These downstream values are a paired sensitivity test, not a reproduction of
the notebook's historical absolute metrics: the local sample, prevalence
weighting, V100 precision, and fixed estimator budget differ.

## Remaining performance headroom

The slowest actor in the final run took 2.95 s and reported about 0.51 s
reading, 0.72 s tokenizing, 0.14 s sorting, 0.83 s constructing sequences, and
0.72 s writing; CPU writes overlap other work, so stage values are not
additive. Profiling shows the next largest exact-semantics opportunity is the
pageable GPU-to-host sequence copy. Pinned staging plus split/write overlap
could plausibly remove another 0.3–0.6 s, but it adds memory-lifetime and
resource-management complexity that is not needed for the sub-4-second goal.
