import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.tokenization.cloud_benchmark import (
    EXPECTED_ROWS,
    EXPECTED_SEQUENCES,
    S3Store,
    arm_configuration,
    assert_unique_output_keys,
    benchmark_uris,
    cleanup_benchmark_prefixes,
    commit_split,
    derive_override_num_blocks,
    generate_comparison_summary,
    new_result,
    parse_s3_uri,
    release_fast_actors,
    realized_arm_configuration,
    parse_args,
    s3_join,
    validate_result_schema,
    validate_fast_output_files,
    validate_parquet_footer_contract,
    run_fast_stage,
    run_fast_measured,
    validate_uri_isolation,
)


class _FakeStore:
    def __init__(self, counts):
        self.counts = dict(counts)
        self.deleted = []

    def delete_prefix(self, uri):
        self.deleted.append(uri)
        return self.counts.get(uri, 0)


class _InventoryStore:
    def __init__(self, inventory):
        self.inventory = inventory

    def list_files(self, uri):
        return list(self.inventory.get(uri, ()))

    def write_json(self, uri, document):
        self.written_uri = uri
        self.written_document = document


class _WarmupDeleteFailureStore(_FakeStore):
    def delete_prefix(self, uri):
        self.deleted.append(uri)
        if uri.endswith("warmup"):
            raise OSError("transient warmup delete failure")
        return self.counts.get(uri, 0)


class CloudBenchmarkTest(unittest.TestCase):
    def test_footer_validation_checks_fast_and_legacy_tensor_metadata(self):
        fast_values = np.zeros((2, 4096), dtype=np.uint16)
        storage = pa.Array.from_buffers(
            pa.binary(8192),
            len(fast_values),
            [None, pa.py_buffer(fast_values.view("uint8"))],
        )
        fast_field = pa.field(
            "input_ids",
            storage.type,
            metadata={
                b"ray.data.fixed_size_binary_tensor.shape": b"[4096]",
                b"ray.data.fixed_size_binary_tensor.dtype": b"uint16",
            },
        )
        fast_table = pa.Table.from_arrays(
            [storage], schema=pa.schema([fast_field])
        )

        from ray.data.extensions.tensor_extension import ArrowTensorArray

        legacy_table = pa.Table.from_arrays(
            [ArrowTensorArray.from_numpy(np.zeros((2, 4096), dtype=np.int64))],
            names=["input_ids"],
        )

        buffers = {}
        for key, table in (("bucket/fast.parquet", fast_table), ("bucket/legacy.parquet", legacy_table)):
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink)
            buffers[key] = sink.getvalue()

        class Filesystem:
            def open_input_file(self, path):
                return pa.BufferReader(buffers[path])

        store = SimpleNamespace(filesystem=Filesystem())
        self.assertEqual(
            validate_parquet_footer_contract(
                store, ["s3://bucket/fast.parquet"], arm="fast"
            )["rows"],
            2,
        )
        self.assertEqual(
            validate_parquet_footer_contract(
                store, ["s3://bucket/legacy.parquet"], arm="original"
            )["rows"],
            2,
        )

    def test_s3_store_aborts_every_multipart_upload_under_prefix(self):
        class Paginator:
            def paginate(self, **kwargs):
                self.kwargs = kwargs
                return [
                    {
                        "Uploads": [
                            {"Key": "run/warmup/a", "UploadId": "u1"},
                            {"Key": "run/warmup/b", "UploadId": "u2"},
                        ]
                    }
                ]

        class Client:
            def __init__(self):
                self.paginator = Paginator()
                self.aborted = []

            def get_paginator(self, name):
                self.name = name
                return self.paginator

            def abort_multipart_upload(self, **kwargs):
                self.aborted.append(kwargs)

        client = Client()
        store = S3Store(filesystem=object(), s3_client=client)

        count = store.abort_multipart_uploads("s3://bucket/run/warmup")

        self.assertEqual(count, 2)
        self.assertEqual(client.name, "list_multipart_uploads")
        self.assertEqual(
            client.paginator.kwargs,
            {"Bucket": "bucket", "Prefix": "run/warmup/"},
        )
        self.assertEqual(client.aborted[0]["UploadId"], "u1")

    def test_fast_actor_release_frees_validation_resources(self):
        class FakeRay:
            def __init__(self):
                self.killed = []

            def kill(self, actor, no_restart):
                self.killed.append((actor, no_restart))

            def cluster_resources(self):
                return {"CPU": 64.0, "GPU": 4.0}

            def available_resources(self):
                return {"CPU": 64.0, "GPU": 4.0}

        ray = FakeRay()
        actors = [object() for _ in range(4)]

        result = release_fast_actors(ray, actors)

        self.assertEqual(result["actors_released"], 4)
        self.assertEqual(ray.killed, [(actor, True) for actor in actors])

    def test_parse_and_join_s3_uri(self):
        parsed = parse_s3_uri("s3://bucket/a/b/")

        self.assertEqual(parsed.bucket, "bucket")
        self.assertEqual(parsed.key, "a/b")
        self.assertEqual(parsed.arrow_path, "bucket/a/b")
        self.assertEqual(s3_join(parsed.uri, "c", "train.parquet"), "s3://bucket/a/b/c/train.parquet")

    def test_rejects_non_s3_and_unsafe_uris(self):
        for uri in ("/tmp/data", "https://bucket/key", "s3://bucket/a/../b", "s3:///key"):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                parse_s3_uri(uri)

    def test_tuned_block_derivation_and_config(self):
        self.assertEqual(derive_override_num_blocks(EXPECTED_ROWS["train"]), 92)
        self.assertEqual(derive_override_num_blocks(EXPECTED_ROWS["val"]), 12)
        config = arm_configuration("tuned")

        self.assertEqual(config["actors"], 8)
        self.assertEqual(config["gpus_per_actor"], 0.5)
        self.assertEqual(config["batch_size"], 16_384)
        self.assertEqual(config["per_split"]["train"]["override_num_blocks"], 92)

    def test_original_and_fast_configs_lock_benchmark_contract(self):
        original = arm_configuration("original")
        fast = arm_configuration("fast")

        self.assertEqual(original["actors"], 4)
        self.assertEqual(original["merchant_hash_mode"], "string_hash")
        self.assertIsNone(original["per_split"]["train"]["override_num_blocks"])
        self.assertEqual(fast["actors"], 4)
        self.assertEqual(fast["row_groups_per_batch"], 16)
        self.assertEqual(fast["gpus_per_actor"], 1.0)
        self.assertEqual(fast["write_threads_per_actor"], 4)
        self.assertEqual(fast["output_shard_size_bytes"], 128 << 20)
        self.assertEqual(fast["kvikio_remote_connections"], 8)
        self.assertEqual(fast["kvikio_task_size_bytes"], 4 << 20)

    def test_output_uris_are_arm_scoped_and_unique(self):
        original = benchmark_uris("s3://bucket/project/benchmarks", "run-1", "original")
        fast = benchmark_uris("s3://bucket/project/benchmarks", "run-1", "fast")

        self.assertNotEqual(original["output"], fast["output"])
        self.assertEqual(
            original["output"],
            "s3://bucket/project/benchmarks/run-1/outputs/original",
        )
        assert_unique_output_keys(
            [{"uri": s3_join(original["output"], "train", "part-0.parquet")},
             {"uri": s3_join(original["output"], "train", "part-1.parquet")}]
        )
        with self.assertRaisesRegex(ValueError, "Duplicate output"):
            assert_unique_output_keys([{"key": "same"}, {"key": "same"}])

    def test_input_must_not_overlap_any_generated_prefix(self):
        uris = benchmark_uris("s3://bucket/project", "run-1", "original")
        validate_uri_isolation("s3://bucket/project/input", uris)

        with self.assertRaisesRegex(ValueError, "overlaps generated warmup"):
            validate_uri_isolation(uris["warmup"], uris)

    def test_result_schema_requires_stable_top_level_fields(self):
        document = new_result(
            run_id="run-1",
            arm="fast",
            input_uri="s3://bucket/input",
            output_uri="s3://bucket/output",
            config=arm_configuration("fast"),
        )
        validate_result_schema(document)

        del document["pipeline"]
        with self.assertRaisesRegex(ValueError, "pipeline"):
            validate_result_schema(document)

    def test_fast_reported_output_files_must_match_s3_inventory(self):
        root = "s3://bucket/run/outputs/fast"
        train = s3_join(root, "train")
        store = _InventoryStore(
            {train: [{"uri": s3_join(train, "part-00000.parquet"), "bytes": 10}]}
        )
        validate_fast_output_files(
            store,
            output_uri=root,
            output_files={"train": [s3_join(train, "part-00000.parquet")]},
            splits=("train",),
        )
        with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
            validate_fast_output_files(
                store,
                output_uri=root,
                output_files={"train": [s3_join(train, "different.parquet")]},
                splits=("train",),
            )

    def test_fast_adapter_passes_every_audited_setting_and_bounded_warmup(self):
        captured = {}

        def fake_api(
            ray_module,
            input_uris,
            output_uri,
            *,
            actors,
            cpus_per_actor,
            gpus_per_actor,
            row_groups_per_batch,
            write_threads,
            output_shard_size_bytes,
            actor_handles,
            splits,
            processor_kwargs,
            max_row_groups_per_partition,
            aws_region,
            s3_connections,
            kvikio_task_size_bytes,
            overlap_split_writes,
        ):
            captured.update(locals())
            return {
                "actors": [object() for _ in range(actors)],
                "sequence_counts": {split: 1 for split in splits},
                "raw_rows": {split: 1 for split in splits},
                "stage_timings": {},
                "actor_stats": [],
                "output_files": {
                    split: [f"{output_uri}/{split}/part-00000.parquet"]
                    for split in splits
                },
                "backend_info": [],
                "config": {},
                "writes_success_markers": False,
            }

        config = arm_configuration("fast")
        config["aws_region"] = "us-west-2"
        with patch(
            "benchmarks.tokenization.cloud_benchmark._fast_api",
            return_value=fake_api,
        ):
            run_fast_stage(
                object(),
                input_uris={"train": "s3://bucket/input/train.parquet"},
                output_uri="s3://bucket/output",
                config=config,
                actor_handles=None,
                warmup=True,
            )

        self.assertEqual(captured["actors"], 4)
        self.assertEqual(captured["cpus_per_actor"], 16)
        self.assertEqual(captured["gpus_per_actor"], 1.0)
        self.assertEqual(captured["row_groups_per_batch"], 16)
        self.assertEqual(captured["write_threads"], 4)
        self.assertEqual(captured["output_shard_size_bytes"], 128 << 20)
        self.assertEqual(captured["s3_connections"], 8)
        self.assertEqual(captured["kvikio_task_size_bytes"], 4 << 20)
        self.assertFalse(captured["overlap_split_writes"])
        self.assertEqual(captured["max_row_groups_per_partition"], 1)

    def test_fast_cli_knobs_override_only_the_realized_fast_config(self):
        args = parse_args(
            [
                "--arm",
                "fast",
                "--run-id",
                "run-1",
                "--fast-row-groups-per-batch",
                "32",
                "--fast-write-threads-per-actor",
                "8",
                "--fast-kvikio-remote-connections",
                "16",
                "--fast-kvikio-task-size-bytes",
                str(1 << 20),
                "--fast-output-shard-size-bytes",
                str(64 << 20),
                "--fast-overlap-split-writes",
            ]
        )

        self.assertEqual(args.fast_row_groups_per_batch, 32)
        self.assertEqual(args.fast_write_threads_per_actor, 8)
        self.assertEqual(args.fast_kvikio_remote_connections, 16)
        self.assertEqual(args.fast_kvikio_task_size_bytes, 1 << 20)
        self.assertEqual(args.fast_output_shard_size_bytes, 64 << 20)
        self.assertTrue(args.fast_overlap_split_writes)

        config = realized_arm_configuration(args)
        self.assertEqual(config["row_groups_per_batch"], 32)
        self.assertEqual(config["write_threads_per_actor"], 8)
        self.assertEqual(config["kvikio_remote_connections"], 16)
        self.assertEqual(config["kvikio_task_size_bytes"], 1 << 20)
        self.assertEqual(config["output_shard_size_bytes"], 64 << 20)
        self.assertTrue(config["actors_reused_after_warmup"])
        self.assertEqual(config["actor_lifecycle"], "prewarmed_actor_reuse")

    def test_skip_warmup_records_fresh_actor_inclusive_lifecycle(self):
        args = parse_args(
            [
                "--arm",
                "fast",
                "--run-id",
                "run-1",
                "--skip-warmup",
            ]
        )

        config = realized_arm_configuration(args)

        self.assertFalse(config["actors_reused_after_warmup"])
        self.assertEqual(config["actor_lifecycle"], "fresh_actor_inclusive")

    def test_fast_cli_knobs_must_be_positive(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--arm",
                    "fast",
                    "--run-id",
                    "run-1",
                    "--fast-kvikio-task-size-bytes",
                    "0",
                ]
            )

    def test_fast_adapter_rejects_api_signature_drift(self):
        def incompatible_api(ray_module, input_uris, output_uri):
            raise AssertionError("must fail before invocation")

        with patch(
            "benchmarks.tokenization.cloud_benchmark._fast_api",
            return_value=incompatible_api,
        ), self.assertRaisesRegex(RuntimeError, "incompatible"):
            run_fast_stage(
                object(),
                input_uris={"train": "s3://bucket/input/train.parquet"},
                output_uri="s3://bucket/output",
                config=arm_configuration("fast"),
                actor_handles=None,
                warmup=True,
            )

    def test_fast_count_mismatch_fails_before_inventory_or_commit(self):
        core = {
            "actors": [object() for _ in range(4)],
            "raw_rows": dict(EXPECTED_ROWS),
            "sequence_counts": {**EXPECTED_SEQUENCES, "train": 1},
        }
        store = _InventoryStore({})
        with patch(
            "benchmarks.tokenization.cloud_benchmark.run_fast_stage",
            return_value=core,
        ), self.assertRaisesRegex(RuntimeError, "sequence-count mismatch"):
            run_fast_measured(
                object(),
                store,
                run_id="run-1",
                input_uris={
                    split: f"s3://bucket/input/{split}.parquet"
                    for split in ("train", "val", "test")
                },
                output_uri="s3://bucket/output",
                config=arm_configuration("fast"),
                actor_handles=(),
            )

        self.assertFalse(hasattr(store, "written_uri"))

    def test_success_marker_contains_exact_nonempty_inventory(self):
        split = "s3://bucket/run/outputs/fast/train"
        inventory = [
            {"uri": s3_join(split, "part-00000.parquet"), "key": "part-00000.parquet", "bytes": 10},
            {"uri": s3_join(split, "part-00001.parquet"), "key": "part-00001.parquet", "bytes": 20},
        ]
        store = _InventoryStore({split: inventory})

        marker = commit_split(
            store, split, arm="fast", split="train", run_id="run-1"
        )

        self.assertEqual(marker["objects"], inventory)
        self.assertEqual(marker["output_bytes"], 30)
        self.assertEqual(store.written_uri, s3_join(split, "_SUCCESS.json"))

    def test_orchestrator_cli_aliases_are_accepted(self):
        args = parse_args(
            [
                "--arm",
                "tuned",
                "--run-id",
                "run-1",
                "--warmup",
                "--validate",
                "--retain-output",
                "--skip-smoke",
            ]
        )

        self.assertFalse(args.skip_warmup)
        self.assertFalse(args.skip_validation)
        self.assertTrue(args.retain_output)
        self.assertTrue(args.skip_smoke_test)

    def test_cleanup_always_removes_warmup_and_removes_failed_output(self):
        store = _FakeStore({"s3://b/warmup": 3, "s3://b/output": 5})
        actions = cleanup_benchmark_prefixes(
            store,
            warmup_uri="s3://b/warmup",
            output_uri="s3://b/output",
            succeeded=False,
            retain_output=True,
        )

        self.assertEqual(store.deleted, ["s3://b/warmup", "s3://b/output"])
        self.assertEqual(actions["warmup_deleted_objects"], 3)
        self.assertEqual(actions["output_deleted_objects"], 5)

    def test_cleanup_retains_successful_measured_output(self):
        store = _FakeStore({"s3://b/warmup": 1, "s3://b/output": 2})
        cleanup_benchmark_prefixes(
            store,
            warmup_uri="s3://b/warmup",
            output_uri="s3://b/output",
            succeeded=True,
            retain_output=True,
        )

        self.assertEqual(store.deleted, ["s3://b/warmup"])

    def test_cleanup_attempts_failed_output_after_warmup_delete_error(self):
        store = _WarmupDeleteFailureStore({"s3://b/output": 2})

        with self.assertRaisesRegex(RuntimeError, "warmup"):
            cleanup_benchmark_prefixes(
                store,
                warmup_uri="s3://b/warmup",
                output_uri="s3://b/output",
                succeeded=False,
                retain_output=True,
            )

        self.assertEqual(store.deleted, ["s3://b/warmup", "s3://b/output"])

    def test_summary_supports_partial_arm_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = new_result(
                run_id="run-1",
                arm="original",
                input_uri="s3://bucket/input",
                output_uri="s3://bucket/output",
                config=arm_configuration("original"),
            )
            result.update(status="succeeded", finished_at="2026-01-01T00:00:00Z")
            result["warmup"] = {"elapsed_s": 2.0}
            result["pipeline"] = {"elapsed_s": 10.0, "output_bytes": 1024}
            result["validation"] = {"status": "passed"}
            (root / "original.json").write_text(__import__("json").dumps(result))

            summary = generate_comparison_summary(root)

            self.assertEqual([row["arm"] for row in summary["rows"]], ["original"])
            self.assertTrue((root / "summary.json").exists())
            self.assertIn("10.00", (root / "summary.md").read_text())

    def test_summary_finds_per_arm_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm_dir = root / "fast"
            arm_dir.mkdir()
            result = new_result(
                run_id="run-1",
                arm="fast",
                input_uri="s3://bucket/input",
                output_uri="s3://bucket/output",
                config=arm_configuration("fast"),
            )
            result.update(status="succeeded", finished_at="2026-01-01T00:00:00Z")
            result["pipeline"] = {"elapsed_s": 1.0, "output_bytes": 1024}
            result["validation"] = {"status": "passed"}
            (arm_dir / "fast.json").write_text(__import__("json").dumps(result))

            summary = generate_comparison_summary(root)

            self.assertEqual([row["arm"] for row in summary["rows"]], ["fast"])

    def test_strict_summary_rejects_mixed_or_unvalidated_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ("original", "tuned", "fast"):
                result = new_result(
                    run_id=("other-run" if arm == "fast" else "run-1"),
                    arm=arm,
                    input_uri="s3://bucket/input",
                    output_uri=f"s3://bucket/output/{arm}",
                    config=arm_configuration(arm),
                )
                result.update(status="succeeded", finished_at="2026-01-01T00:00:00Z")
                result["pipeline"] = {
                    "elapsed_s": 1.0,
                    "sequences_per_s": 1.0,
                    "output_bytes": 1,
                }
                result["validation"] = {"status": "passed"}
                (root / f"{arm}.json").write_text(__import__("json").dumps(result))

            with self.assertRaisesRegex(RuntimeError, "run IDs do not match"):
                generate_comparison_summary(root, require_complete=True)

            fast_path = root / "fast.json"
            fast = __import__("json").loads(fast_path.read_text())
            fast["run_id"] = "run-1"
            fast["validation"] = {"status": "skipped"}
            fast_path.write_text(__import__("json").dumps(fast))
            with self.assertRaisesRegex(RuntimeError, "unvalidated"):
                generate_comparison_summary(root, require_complete=True)


if __name__ == "__main__":
    unittest.main()
