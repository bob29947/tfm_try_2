import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.gpu_parquet import (
    _group_partitions_by_id,
    plan_parquet_key_range_partitions,
)
from src.ray_tokenize import tokenized_parquet_read_kwargs


class ParquetPartitionRunnerTest(unittest.TestCase):
    def _write(self, root: Path, values: list[int], *, row_group_size: int) -> Path:
        path = root / "data.parquet"
        pq.write_table(
            pa.table({"user_id": values, "value": values}),
            path,
            row_group_size=row_group_size,
        )
        return path

    def test_plans_aligned_row_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), list(range(12)), row_group_size=3)
            planned = plan_parquet_key_range_partitions(
                {"train": path}, key_column="user_id", num_partitions=4
            )["train"]

        self.assertEqual(
            [(part["key_min"], part["key_max"]) for part in planned],
            [(0, 2), (3, 5), (6, 8), (9, 11)],
        )
        self.assertEqual(
            [part["fragments"][0]["row_groups"] for part in planned],
            [[0], [1], [2], [3]],
        )

    def test_boundary_row_group_is_repeated_for_processor_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), list(range(12)), row_group_size=12)
            planned = plan_parquet_key_range_partitions(
                path, key_column="user_id", num_partitions=4
            )["default"]

        self.assertEqual(
            [part["fragments"][0]["row_groups"] for part in planned],
            [[0], [0], [0], [0]],
        )
        self.assertEqual(
            [part["estimated_input_rows"] for part in planned], [12, 12, 12, 12]
        )

    def test_rejects_more_partitions_than_integral_key_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), [0, 1], row_group_size=2)
            with self.assertRaisesRegex(ValueError, "exceeds the integral key"):
                plan_parquet_key_range_partitions(
                    path, key_column="user_id", num_partitions=4
                )

    def test_groups_named_inputs_by_partition_id(self):
        partitions = {
            "train": [
                {"partition_id": 0, "fragments": [{"path": "train", "row_groups": [0]}]},
                {"partition_id": 1, "fragments": []},
            ],
            "test": [
                {"partition_id": 0, "fragments": [{"path": "test", "row_groups": [0]}]},
            ],
        }

        grouped = _group_partitions_by_id(partitions)

        self.assertEqual(list(grouped), [0])
        self.assertEqual(
            [item["fragments"][0]["path"] for item in grouped[0]],
            ["train", "test"],
        )

    def test_binary_tensor_manifest_uses_public_ray_schema_option(self):
        kwargs = tokenized_parquet_read_kwargs(
            {
                "config": {
                    "sequence_length": 4096,
                    "output": {"format": "binary-tensor", "dtype": "uint16"},
                }
            }
        )

        dtype, shape = kwargs["tensor_column_schema"]["input_ids"]
        self.assertEqual(str(dtype), "uint16")
        self.assertEqual(shape, (4096,))
        self.assertEqual(tokenized_parquet_read_kwargs({"output": {}}), {})


if __name__ == "__main__":
    unittest.main()
