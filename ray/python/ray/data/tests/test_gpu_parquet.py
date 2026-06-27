import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ray.data._internal.datasource.parquet_datasource import (
    _decode_fixed_size_binary_tensor_columns,
)
from ray.data.experimental.gpu_parquet import plan_parquet_key_range_partitions


def test_plan_parquet_key_range_partitions(tmp_path):
    path = tmp_path / "data.parquet"
    table = pa.table(
        {
            "user_id": pa.array(list(range(12)), type=pa.int64()),
            "value": pa.array(list(range(12)), type=pa.int64()),
        }
    )
    pq.write_table(table, path, row_group_size=3)

    partitions = plan_parquet_key_range_partitions(
        {"train": path}, key_column="user_id", num_partitions=4
    )

    planned = partitions["train"]
    assert [(p["key_min"], p["key_max"]) for p in planned] == [
        (0, 2),
        (3, 5),
        (6, 8),
        (9, 11),
    ]
    assert [p["fragments"][0]["row_groups"] for p in planned] == [
        [0],
        [1],
        [2],
        [3],
    ]
    assert [p["estimated_input_rows"] for p in planned] == [3, 3, 3, 3]


def test_decode_fixed_size_binary_tensor_columns():
    values = np.arange(12, dtype=np.int32).reshape(3, 4)
    storage = pa.Array.from_buffers(
        pa.binary(values.shape[1] * values.dtype.itemsize),
        len(values),
        [None, pa.py_buffer(np.ascontiguousarray(values).view("uint8"))],
    )
    field = pa.field(
        "input_ids",
        storage.type,
        metadata={
            b"ray.data.fixed_size_binary_tensor.shape": b"[4]",
            b"ray.data.fixed_size_binary_tensor.dtype": b"int32",
        },
    )
    table = pa.Table.from_arrays([storage], schema=pa.schema([field]))

    decoded = _decode_fixed_size_binary_tensor_columns(table)
    actual = decoded.column("input_ids").chunk(0).to_numpy_ndarray()

    np.testing.assert_array_equal(actual, values)
