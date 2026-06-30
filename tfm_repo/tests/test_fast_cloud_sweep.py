import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmarks.tokenization.fast_cloud_sweep import (
    MIB,
    TrialConfig,
    choose_candidates,
    choose_winner,
    cleanup_trial_prefix,
    default_trial_configs,
    final_benchmark_argv,
    parse_args,
    parse_trial_config,
    summarize_trial_core,
    trial_prefix,
    validate_trial_configs,
)


def _trial(name, seconds, *, phase="screen", safe=True):
    return {
        "phase": phase,
        "config": {"name": name},
        "status": "succeeded" if safe else "failed",
        "safe": safe,
        "timings": {"runner_total_s": seconds, "process_wall_s": seconds - 0.1},
    }


class FastCloudSweepTest(unittest.TestCase):
    def test_aws_wrapper_has_sync_cleanup_and_zero_instance_backstops(self):
        wrapper = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "tokenization"
            / "run_aws_fast_sweep.sh"
        ).read_text()

        self.assertIn('trap cleanup EXIT INT TERM', wrapper)
        self.assertIn('rsync-down', wrapper)
        self.assertIn('fast_cloud_sweep.py', wrapper)
        self.assertIn('--run-final-full', wrapper)
        self.assertIn('aws ec2 wait instance-terminated', wrapper)
        self.assertIn('Teardown postcondition failed', wrapper)
        self.assertIn('cleanup_s3_run.py', wrapper)

    def test_default_matrix_is_focused_and_unique(self):
        configs = validate_trial_configs(default_trial_configs())

        self.assertEqual(len(configs), 12)
        self.assertEqual(len({config.name for config in configs}), len(configs))
        self.assertEqual(configs[0].row_groups_per_batch, 16)
        self.assertEqual(
            {8, 16, 32, 64},
            {config.s3_connections for config in configs},
        )
        self.assertEqual(
            {config.kvikio_task_size_bytes for config in configs},
            {MIB // 2, 1 * MIB, 2 * MIB, 4 * MIB, 16 * MIB},
        )

    def test_parse_custom_trial_config(self):
        config = parse_trial_config(
            "name=trial-a,row_groups=24,connections=16,task_mib=2,"
            "write_threads=6,shard_mib=96"
        )

        self.assertEqual(config.name, "trial-a")
        self.assertEqual(config.row_groups_per_batch, 24)
        self.assertEqual(config.s3_connections, 16)
        self.assertEqual(config.kvikio_task_size_bytes, 2 * MIB)
        self.assertEqual(config.output_shard_size_bytes, 96 * MIB)

        with self.assertRaises(argparse.ArgumentTypeError):
            parse_trial_config("name=x,unknown=1")
        with self.assertRaises(ValueError):
            validate_trial_configs([config, config])

    def test_trial_prefix_is_scoped_below_disposable_tuning_root(self):
        self.assertEqual(
            trial_prefix("s3://bucket/bench", "run-1", "confirmation", "cfg-1"),
            "s3://bucket/bench/run-1/tuning/confirmation/cfg-1",
        )
        with self.assertRaises(ValueError):
            trial_prefix("s3://bucket/bench", "run-1", "bad", "cfg-1")

    def test_candidate_selection_ignores_failed_and_confirms_forced(self):
        trials = [_trial("slow", 5.0), _trial("failed", 1.0, safe=False), _trial("fast", 3.0)]

        self.assertEqual(choose_candidates(trials, count=2), ["fast", "slow"])
        self.assertEqual(
            choose_candidates(trials, count=2, forced_name="slow"), ["slow"]
        )
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            choose_candidates(trials, count=2, forced_name="failed")

    def test_winner_prefers_full_train_evidence(self):
        screens = [_trial("a", 2.0), _trial("b", 3.0)]
        confirmations = [
            _trial("a", 12.0, phase="confirmation"),
            _trial("b", 10.0, phase="confirmation"),
        ]

        self.assertEqual(
            choose_winner(screens, confirmations)["config"]["name"], "b"
        )
        self.assertEqual(
            choose_winner(screens, [], forced_name="a")["config"]["name"], "a"
        )

    def test_trial_summary_reports_wall_critical_aggregate_and_peak(self):
        core = {
            "raw_rows": {"train": 100},
            "sequence_counts": {"train": 5},
            "stage_timings": {
                "total_s": 4.0,
                "plan_s": 0.2,
                "process_s": 3.8,
                "split_wall_s": {"train": 3.7},
                "read_s": 5.0,
                "write_s": 1.0,
            },
            "actor_stats": [
                [{"elapsed_s": 3.0, "read_s": 2.0, "write_s": 0.5, "peak_gpu_memory_bytes": 10}],
                [{"elapsed_s": 3.5, "read_s": 3.0, "write_s": 0.4, "peak_gpu_memory_bytes": 20}],
            ],
        }

        result = summarize_trial_core(core, output_bytes=123)

        self.assertEqual(result["actor_critical_path_s"]["read_s"], 3.0)
        self.assertEqual(result["actor_aggregate_s"]["read_s"], 5.0)
        self.assertEqual(result["peak_gpu_memory_bytes"], 20)
        self.assertEqual(result["rows_per_s"], 25.0)

    def test_cleanup_attempts_delete_abort_and_empty_verification(self):
        class Store:
            def __init__(self):
                self.calls = []

            def delete_prefix(self, prefix):
                self.calls.append(("delete", prefix))
                return 3

            def abort_multipart_uploads(self, prefix):
                self.calls.append(("abort", prefix))
                return 1

            def list_files(self, prefix):
                self.calls.append(("list", prefix))
                return []

        store = Store()
        result = cleanup_trial_prefix(store, "s3://bucket/run/tuning")

        self.assertEqual(result["deleted_objects"], 3)
        self.assertEqual(result["aborted_multipart_uploads"], 1)
        self.assertEqual([name for name, _ in store.calls], ["delete", "abort", "list"])

    def test_final_argv_forwards_selected_knobs_and_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                input_uri="s3://bucket/input",
                output_root="s3://bucket/benchmarks",
                run_id="run-1",
                results_dir=Path(temporary),
                ray_address="auto",
                aws_region="us-west-2",
                final_overlap_split_writes=True,
                overwrite_final_output=False,
                cluster_bootstrap_seconds=12.5,
            )
            config = TrialConfig("winner", 32, 16, 1 * MIB, 8, 64 * MIB)

            argv = final_benchmark_argv(args, config)

            self.assertIn("--fast-overlap-split-writes", argv)
            self.assertEqual(argv[argv.index("--fast-row-groups-per-batch") + 1], "32")
            self.assertEqual(argv[argv.index("--fast-output-shard-size-bytes") + 1], str(64 * MIB))

    def test_cli_rejects_overlap_without_final_run(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--run-id",
                    "run-1",
                    "--final-overlap-split-writes",
                ]
            )


if __name__ == "__main__":
    unittest.main()
