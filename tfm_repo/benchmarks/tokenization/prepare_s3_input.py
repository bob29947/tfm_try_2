#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Copy and verify the canonical v3 benchmark input under the project prefix.

The copy is server-side and idempotent.  Source objects are never deleted.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_SOURCE = "s3://bobbwang-tfm-gpu-e2e/temporal_split_v3"
DEFAULT_DESTINATION = (
    "s3://bobbwang-tfm-gpu-e2e/tfm_try_2/input/temporal_split_v3"
)
EXPECTED = {
    "train.parquet": {
        "bytes": 2_332_745_196,
        "rows": 191_957_576,
        "row_groups": 184,
        "sha256": "015c791659e6d4fb6b696f5355d20a2006e6725431a31f6b401b51dcfd669775",
    },
    "val.parquet": {
        "bytes": 296_393_446,
        "rows": 24_031_181,
        "row_groups": 23,
        "sha256": "415a5cc41e2d7ac8d56b3bca870475808c0565d309ea81506d9a7bd5f0b6bd02",
    },
    "test.parquet": {
        "bytes": 296_457_841,
        "rows": 24_011_243,
        "row_groups": 23,
        "sha256": "89e3ba6dbacd7926b9e69c4fa6df6433ea259bacfb75da0cfcca98517884cad2",
    },
}


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/").rstrip("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or any(part in {".", ".."} for part in key.split("/"))
    ):
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    return parsed.netloc, key


def aws_json(
    *args: str,
    allow_missing: bool = False,
    region: str | None = None,
) -> dict | None:
    command = ["aws", *args]
    if region:
        command.extend(["--region", region])
    command.extend(["--output", "json"])
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        if allow_missing and any(
            marker in result.stderr for marker in ("Not Found", "NoSuchKey", "(404)")
        ):
            return None
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return json.loads(result.stdout or "{}")


def head(
    bucket: str,
    key: str,
    *,
    region: str,
    allow_missing: bool = False,
) -> dict | None:
    return aws_json(
        "s3api",
        "head-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--checksum-mode",
        "ENABLED",
        region=region,
        allow_missing=allow_missing,
    )


def normalized_etag(value: str | None) -> str:
    return (value or "").strip('"')


def checksum_sha256_b64(hex_digest: str) -> str:
    return base64.b64encode(bytes.fromhex(hex_digest)).decode("ascii")


def copy_one(
    source_bucket: str,
    source_key: str,
    destination_bucket: str,
    destination_key: str,
    expected: dict,
    region: str,
) -> tuple[dict, bool]:
    source = head(source_bucket, source_key, region=region)
    if source is None:
        raise RuntimeError(f"Missing source object s3://{source_bucket}/{source_key}")
    if int(source["ContentLength"]) != expected["bytes"]:
        raise RuntimeError(
            f"Unexpected source size for {source_key}: "
            f"{source['ContentLength']} != {expected['bytes']}"
        )

    source_etag = normalized_etag(source.get("ETag"))
    expected_checksum = checksum_sha256_b64(expected["sha256"])
    destination = head(
        destination_bucket,
        destination_key,
        region=region,
        allow_missing=True,
    )
    metadata = (destination or {}).get("Metadata", {})
    reusable = bool(
        destination
        and int(destination.get("ContentLength", -1)) == expected["bytes"]
        and metadata.get("source-etag") == source_etag
        and metadata.get("sha256") == expected["sha256"]
        and destination.get("ChecksumSHA256") == expected_checksum
    )
    if not reusable:
        aws_json(
            "s3api",
            "copy-object",
            "--bucket",
            destination_bucket,
            "--key",
            destination_key,
            "--copy-source",
            f"{source_bucket}/{source_key}",
            "--metadata-directive",
            "REPLACE",
            "--metadata",
            f"source-etag={source_etag},sha256={expected['sha256']}",
            "--content-type",
            "application/vnd.apache.parquet",
            "--server-side-encryption",
            "AES256",
            "--checksum-algorithm",
            "SHA256",
            region=region,
        )
        destination = head(destination_bucket, destination_key, region=region)

    assert destination is not None
    if int(destination["ContentLength"]) != expected["bytes"]:
        raise RuntimeError(f"Destination size mismatch for {destination_key}")
    if destination.get("ChecksumSHA256") != expected_checksum:
        raise RuntimeError(
            f"Destination SHA-256 mismatch for {destination_key}: "
            f"{destination.get('ChecksumSHA256')!r} != {expected_checksum!r}"
        )
    return {
        "source_uri": f"s3://{source_bucket}/{source_key}",
        "destination_uri": f"s3://{destination_bucket}/{destination_key}",
        "bytes": int(destination["ContentLength"]),
        "source_etag": source_etag,
        "destination_etag": normalized_etag(destination.get("ETag")),
        "checksum_sha256_b64": destination["ChecksumSHA256"],
        "sha256": expected["sha256"],
        "server_side_encryption": destination.get("ServerSideEncryption"),
    }, not reusable


def parquet_metadata(bucket: str, key: str, region: str) -> dict:
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    filesystem = pafs.S3FileSystem(region=region)
    with filesystem.open_input_file(f"{bucket}/{key}") as stream:
        parquet = pq.ParquetFile(stream)
        return {
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "columns": parquet.schema.names,
        }


def upload_manifest(path: Path, bucket: str, key: str, region: str) -> None:
    result = subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(path),
            f"s3://{bucket}/{key}",
            "--only-show-errors",
            "--sse",
            "AES256",
            "--content-type",
            "application/json",
            "--region",
            region,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-uri", default=DEFAULT_SOURCE)
    parser.add_argument("--destination-uri", default=DEFAULT_DESTINATION)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--local-manifest", type=Path)
    args = parser.parse_args()

    source_bucket, source_prefix = parse_s3_uri(args.source_uri)
    destination_bucket, destination_prefix = parse_s3_uri(args.destination_uri)
    objects = {}
    copied = []
    for filename, expected in EXPECTED.items():
        source_key = f"{source_prefix}/{filename}"
        destination_key = f"{destination_prefix}/{filename}"
        record, changed = copy_one(
            source_bucket,
            source_key,
            destination_bucket,
            destination_key,
            expected,
            args.region,
        )
        metadata = parquet_metadata(destination_bucket, destination_key, args.region)
        for field in ("rows", "row_groups"):
            if metadata[field] != expected[field]:
                raise RuntimeError(
                    f"{filename} {field} mismatch: "
                    f"{metadata[field]} != {expected[field]}"
                )
        if "User" not in metadata["columns"]:
            raise RuntimeError(f"{filename} does not contain the User column")
        objects[filename] = {**record, **metadata}
        if changed:
            copied.append(filename)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "region": args.region,
        "source_uri": args.source_uri.rstrip("/"),
        "destination_uri": args.destination_uri.rstrip("/"),
        "source_deleted": False,
        "copied_this_run": copied,
        "objects": objects,
    }
    manifest_key = f"{destination_prefix}/_input_manifest.json"
    if args.local_manifest:
        local_manifest = args.local_manifest.expanduser().resolve()
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        upload_manifest(local_manifest, destination_bucket, manifest_key, args.region)
    else:
        with tempfile.TemporaryDirectory(prefix="tfm-s3-input-") as directory:
            local_manifest = Path(directory) / "input_manifest.json"
            local_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            upload_manifest(local_manifest, destination_bucket, manifest_key, args.region)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Verified manifest: s3://{destination_bucket}/{manifest_key}")


if __name__ == "__main__":
    main()
