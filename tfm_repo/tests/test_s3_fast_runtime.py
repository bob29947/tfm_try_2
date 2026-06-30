import types
import threading
import time
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.tokenization.fast_actor import (
    FastParquetSplitTokenizer,
    _configure_kvikio_runtime,
)
from src.tokenization.parquet_runner import (
    _limit_partition_row_groups,
    plan_parquet_key_range_partitions,
    run_s3_tokenization,
)


class S3FooterPlanningTest(unittest.TestCase):
    def test_s3_footer_plan_preserves_uri_and_exact_row_groups(self):
        sink = pa.BufferOutputStream()
        pq.write_table(
            pa.table({"User": list(range(8)), "value": list(range(8))}),
            sink,
            row_group_size=2,
        )
        parquet_bytes = sink.getvalue()

        class FakeS3:
            def open_input_file(self, path):
                self.opened = path
                return pa.BufferReader(parquet_bytes)

        filesystem = FakeS3()
        uri = "s3://example-bucket/project/train.parquet"
        with mock.patch(
            "src.tokenization.parquet_runner._arrow_input_files",
            return_value=[(uri, filesystem, "example-bucket/project/train.parquet")],
        ):
            planned = plan_parquet_key_range_partitions(
                {"train": uri}, key_column="User", num_partitions=4
            )["train"]

        self.assertEqual(filesystem.opened, "example-bucket/project/train.parquet")
        self.assertEqual(
            [part["fragments"][0]["path"] for part in planned], [uri] * 4
        )
        self.assertEqual(
            [part["fragments"][0]["row_groups"] for part in planned],
            [[0], [1], [2], [3]],
        )
        self.assertEqual(
            [part["fragments"][0]["row_group_num_rows"] for part in planned],
            [[2], [2], [2], [2]],
        )

    def test_warmup_limit_keeps_first_row_group_per_partition(self):
        plan = {
            "train": [
                {
                    "partition_id": 0,
                    "estimated_input_rows": 60,
                    "fragments": [
                        {
                            "path": "s3://bucket/a.parquet",
                            "row_groups": [1, 2],
                            "row_group_num_rows": [10, 20],
                        },
                        {
                            "path": "s3://bucket/b.parquet",
                            "row_groups": [3],
                            "row_group_num_rows": [30],
                        },
                    ],
                }
            ]
        }

        limited = _limit_partition_row_groups(plan, 1)["train"][0]

        self.assertEqual(limited["estimated_input_rows"], 10)
        self.assertEqual(len(limited["fragments"]), 1)
        self.assertEqual(limited["fragments"][0]["row_groups"], [1])
        # The reusable full plan must not be mutated by warmup trimming.
        self.assertEqual(plan["train"][0]["fragments"][0]["row_groups"], [1, 2])


class FastActorS3BehaviorTest(unittest.TestCase):
    def test_kvikio_runtime_settings_are_set_queried_and_verified(self):
        class FakeDefaults:
            def __init__(self):
                self.values = {"num_threads": 1, "task_size": 4 << 20}
                self.set_calls = []

            def set(self, values):
                self.set_calls.append(dict(values))
                self.values.update(values)

            def get(self, name):
                return self.values[name]

        defaults = FakeDefaults()

        realized = _configure_kvikio_runtime(
            defaults, num_threads=32, task_size_bytes=1 << 20
        )

        self.assertEqual(
            defaults.set_calls,
            [{"num_threads": 32, "task_size": 1 << 20}],
        )
        self.assertEqual(
            realized,
            {"num_threads": 32, "task_size_bytes": 1 << 20},
        )

    def test_kvikio_runtime_setting_mismatch_is_fatal(self):
        defaults = types.SimpleNamespace(
            set=lambda values: None,
            get=lambda name: 7 if name == "num_threads" else 4 << 20,
        )

        with self.assertRaisesRegex(RuntimeError, "settings mismatch"):
            _configure_kvikio_runtime(
                defaults, num_threads=8, task_size_bytes=4 << 20
            )

    def test_ready_reports_actor_placement_and_backend(self):
        actor = object.__new__(FastParquetSplitTokenizer)
        actor.s3_mode = True
        actor._s3_backend = "cudf-kvikio"
        actor.aws_region = "us-west-2"
        actor.s3_connections = 8
        actor.kvikio_task_size_bytes = 4 << 20
        actor._kvikio_realized = {
            "num_threads": 8,
            "task_size_bytes": 4 << 20,
        }
        actor.row_groups_per_batch = 16
        context = types.SimpleNamespace(
            get_node_id=lambda: "node-123",
            get_accelerator_ids=lambda: {"GPU": ["0"]},
        )

        with mock.patch("ray.get_runtime_context", return_value=context), mock.patch(
            "src.tokenization.fast_actor.socket.gethostname", return_value="worker-1"
        ):
            info = actor.ready()

        self.assertEqual(info["node_id"], "node-123")
        self.assertEqual(info["hostname"], "worker-1")
        self.assertEqual(info["accelerator_ids"], {"GPU": ["0"]})
        self.assertEqual(info["read_backend"], "cudf-kvikio")
        self.assertEqual(info["kvikio_num_threads"], 8)
        self.assertEqual(info["kvikio_task_size_bytes"], 4 << 20)

    def test_tokenize_finishes_splits_one_at_a_time(self):
        actor = object.__new__(FastParquetSplitTokenizer)
        actor._tokenize_one_split = mock.Mock(
            side_effect=lambda work: {"split": work["split"]}
        )

        result = actor.tokenize([{"split": "train"}, {"split": "val"}])

        self.assertEqual(result, [{"split": "train"}, {"split": "val"}])
        self.assertEqual(
            [call.args[0]["split"] for call in actor._tokenize_one_split.call_args_list],
            ["train", "val"],
        )

    def test_opt_in_write_overlap_prepares_next_split_before_write_finishes(self):
        actor = object.__new__(FastParquetSplitTokenizer)
        actor.overlap_split_writes = True
        actor.write_threads = 1
        train_write_started = threading.Event()
        val_prepared = threading.Event()
        writes = []

        def prepare(work):
            started = time.perf_counter()
            if work["split"] == "val":
                self.assertTrue(train_write_started.wait(timeout=1))
                # Keep preparation active long enough for overlap timing to be
                # robust even on a very fast test host.
                time.sleep(0.02)
                val_prepared.set()
            path = f"s3://bucket/{work['split']}.parquet"
            stat = {
                "split": work["split"],
                "count": 1,
                "rows": 1,
                "compute_s": time.perf_counter() - started,
                "write_s": 0.0,
                "write_wait_s": 0.0,
                "write_overlap_s": 0.0,
                "output_paths": [path],
                "output_path": path,
            }
            return stat, [(path, np.zeros((1, 4), dtype=np.uint16))], started

        def write(_writer, path, _shard):
            if path.endswith("train.parquet"):
                train_write_started.set()
                self.assertTrue(val_prepared.wait(timeout=1))
            writes.append(path)

        actor._prepare_one_split = mock.Mock(side_effect=prepare)
        actor._write_sequences = mock.Mock(side_effect=write)
        actor._cleanup_output_paths = mock.Mock()

        result = actor.tokenize([{"split": "train"}, {"split": "val"}])

        self.assertEqual([stat["split"] for stat in result], ["train", "val"])
        self.assertEqual(
            writes,
            ["s3://bucket/train.parquet", "s3://bucket/val.parquet"],
        )
        self.assertGreater(result[0]["write_overlap_s"], 0.0)
        self.assertEqual(actor._cleanup_output_paths.call_count, 0)

    def test_opt_in_write_overlap_cleans_completed_outputs_on_later_failure(self):
        actor = object.__new__(FastParquetSplitTokenizer)
        actor.overlap_split_writes = True
        actor.write_threads = 1

        def prepare(work):
            if work["split"] == "test":
                raise RuntimeError("injected test preparation failure")
            started = time.perf_counter()
            path = f"s3://bucket/{work['split']}.parquet"
            stat = {
                "split": work["split"],
                "count": 1,
                "rows": 1,
                "compute_s": 0.0,
                "write_s": 0.0,
                "write_wait_s": 0.0,
                "write_overlap_s": 0.0,
                "output_paths": [path],
                "output_path": path,
            }
            return stat, [(path, np.zeros((1, 4), dtype=np.uint16))], started

        actor._prepare_one_split = mock.Mock(side_effect=prepare)
        actor._write_sequences = mock.Mock(return_value=None)
        actor._cleanup_output_paths = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "test preparation failure"):
            actor.tokenize(
                [{"split": "train"}, {"split": "val"}, {"split": "test"}]
            )

        cleaned = {
            path
            for call in actor._cleanup_output_paths.call_args_list
            for path in call.args[0]
        }
        self.assertEqual(
            cleaned,
            {"s3://bucket/train.parquet", "s3://bucket/val.parquet"},
        )

    def test_s3_runner_dispatches_all_splits_once_when_overlap_is_enabled(self):
        plan = {
            split: [
                {
                    "partition_id": 0,
                    "key_min": 0,
                    "key_max": 9,
                    "fragments": [
                        {
                            "path": f"s3://bucket/{split}.parquet",
                            "row_groups": [0],
                            "row_group_num_rows": [1],
                        }
                    ],
                }
            ]
            for split in ("train", "val")
        }
        calls = []

        def map_partitions(partitions, *_args, **_kwargs):
            calls.append(list(partitions))
            return [
                [
                    {
                        "split": split,
                        "count": 1,
                        "rows": 1,
                        "elapsed_s": 0.1,
                        "compute_s": 0.08,
                        "write_s": 0.03,
                        "write_wait_s": 0.01,
                        "write_overlap_s": 0.02,
                        "output_paths": [
                            partitions[split][0]["output_path"]
                        ],
                    }
                    for split in partitions
                ]
            ]

        backend = {
            "hostname": "worker-1",
            "kvikio_num_threads": 8,
            "kvikio_task_size_bytes": 4 << 20,
            "overlap_split_writes": True,
        }
        actor = types.SimpleNamespace(
            ready=types.SimpleNamespace(remote=lambda: backend)
        )
        fake_ray = types.SimpleNamespace(get=lambda refs: refs)

        with mock.patch(
            "src.tokenization.parquet_runner.plan_parquet_key_range_partitions",
            return_value=plan,
        ), mock.patch(
            "src.tokenization.parquet_runner.map_gpu_parquet_partitions",
            side_effect=map_partitions,
        ):
            result = run_s3_tokenization(
                fake_ray,
                {
                    "train": "s3://bucket/train.parquet",
                    "val": "s3://bucket/val.parquet",
                },
                "s3://bucket/output",
                actors=1,
                actor_handles=[actor],
                overlap_split_writes=True,
            )

        self.assertEqual(calls, [["train", "val"]])
        self.assertTrue(result["config"]["overlap_split_writes"])
        self.assertTrue(result["config"]["actors_reused"])
        self.assertFalse(result["config"]["fresh_actors_created"])
        self.assertEqual(result["sequence_counts"], {"train": 1, "val": 1})
        self.assertGreaterEqual(
            result["stage_timings"]["actor_create_and_ready_s"], 0.0
        )
        self.assertGreater(result["stage_timings"]["write_overlap_s"], 0.0)

    def test_s3_runner_records_fresh_actor_creation_inside_stage_timing(self):
        plan = {
            "train": [
                {
                    "partition_id": 0,
                    "key_min": 0,
                    "key_max": 9,
                    "fragments": [
                        {
                            "path": "s3://bucket/train.parquet",
                            "row_groups": [0],
                            "row_group_num_rows": [1],
                        }
                    ],
                }
            ]
        }
        backend = {
            "hostname": "worker-1",
            "kvikio_num_threads": 8,
            "kvikio_task_size_bytes": 4 << 20,
            "overlap_split_writes": False,
        }
        actor = types.SimpleNamespace(
            ready=types.SimpleNamespace(remote=lambda: backend)
        )
        fake_ray = types.SimpleNamespace(get=lambda refs: refs)

        def map_partitions(partitions, *_args, **_kwargs):
            output_path = partitions["train"][0]["output_path"]
            return [
                [
                    {
                        "split": "train",
                        "count": 1,
                        "rows": 1,
                        "elapsed_s": 0.1,
                        "compute_s": 0.05,
                        "write_s": 0.02,
                        "write_wait_s": 0.02,
                        "write_overlap_s": 0.0,
                        "output_paths": [output_path],
                    }
                ]
            ]

        with mock.patch(
            "src.tokenization.parquet_runner.plan_parquet_key_range_partitions",
            return_value=plan,
        ), mock.patch(
            "src.tokenization.parquet_runner.create_gpu_parquet_actors",
            return_value=[actor],
        ) as create_actors, mock.patch(
            "src.tokenization.parquet_runner.map_gpu_parquet_partitions",
            side_effect=map_partitions,
        ):
            result = run_s3_tokenization(
                fake_ray,
                {"train": "s3://bucket/train.parquet"},
                "s3://bucket/output",
                actors=1,
            )

        self.assertFalse(result["config"]["actors_reused"])
        self.assertTrue(result["config"]["fresh_actors_created"])
        self.assertGreaterEqual(
            result["stage_timings"]["actor_create_and_ready_s"], 0.0
        )
        self.assertFalse(create_actors.call_args.kwargs["wait_until_ready"])

    def test_s3_read_fails_if_kvikio_option_is_disabled(self):
        actor = object.__new__(FastParquetSplitTokenizer)
        actor.s3_mode = True
        actor._s3_backend = "cudf-kvikio"
        actor.cudf = types.SimpleNamespace(get_option=lambda _: False)

        with self.assertRaisesRegex(RuntimeError, "disabled"):
            actor._read_row_groups("s3://bucket/train.parquet", [0])

    def test_s3_shard_paths_are_unique(self):
        actor = object.__new__(FastParquetSplitTokenizer)
        actor.output_shard_size_bytes = 16
        sequences = np.zeros((10, 4), dtype=np.uint16)

        shards = actor._sequence_output_shards(
            "s3://bucket/run/train/part-00003.parquet", sequences
        )
        paths = [path for path, _ in shards]

        self.assertEqual(len(paths), 5)
        self.assertEqual(len(set(paths)), len(paths))
        self.assertEqual(paths[0], "s3://bucket/run/train/part-00003-00.parquet")

    def test_s3_writer_reuses_arrow_client_and_closes_stream(self):
        class FakeStream:
            def __init__(self):
                self.closed = False

            def write(self, value):
                return len(value)

            def close(self):
                self.closed = True

        class FakeS3:
            def __init__(self):
                self.paths = []
                self.streams = []

            def open_output_stream(self, path, metadata=None):
                self.paths.append(path)
                stream = FakeStream()
                self.streams.append(stream)
                return stream

        class FakeParquet:
            @staticmethod
            def write_table(table, stream, **kwargs):
                stream.write(b"parquet")

        actor = object.__new__(FastParquetSplitTokenizer)
        actor.output_format = "binary-tensor"
        actor.seq_length = 4
        actor.pa = pa
        actor.pq = FakeParquet()
        actor.compression = "zstd"
        actor.compression_level = 1
        actor.use_dictionary = False
        actor.s3_mode = True
        actor.arrow_s3_fs = FakeS3()
        sequences = np.zeros((2, 4), dtype=np.uint16)

        actor._write_sequences(None, "s3://bucket/a.parquet", sequences)
        actor._write_sequences(None, "s3://bucket/b.parquet", sequences)

        self.assertEqual(actor.arrow_s3_fs.paths, ["bucket/a.parquet", "bucket/b.parquet"])
        self.assertTrue(all(stream.closed for stream in actor.arrow_s3_fs.streams))

    def test_s3_writer_aborts_stream_after_write_failure(self):
        class FakeStream:
            def __init__(self):
                self.aborted = False

            def abort(self):
                self.aborted = True

        class FakeS3:
            def __init__(self):
                self.stream = FakeStream()

            def open_output_stream(self, path, metadata=None):
                return self.stream

            def get_file_info(self, path):
                return types.SimpleNamespace(type=types.SimpleNamespace(name="NotFound"))

        class FailingParquet:
            @staticmethod
            def write_table(table, stream, **kwargs):
                raise OSError("injected multipart failure")

        actor = object.__new__(FastParquetSplitTokenizer)
        actor.output_format = "binary-tensor"
        actor.seq_length = 4
        actor.pa = pa
        actor.pq = FailingParquet()
        actor.compression = "zstd"
        actor.compression_level = 1
        actor.use_dictionary = False
        actor.s3_mode = True
        actor.arrow_s3_fs = FakeS3()

        with self.assertRaisesRegex(OSError, "multipart failure"):
            actor._write_sequences(
                None,
                "s3://bucket/fail.parquet",
                np.zeros((2, 4), dtype=np.uint16),
            )

        self.assertTrue(actor.arrow_s3_fs.stream.aborted)


if __name__ == "__main__":
    unittest.main()
