#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Final local cleanup backstop for one cloud benchmark run.

The remote harness owns normal cleanup.  This command runs after EC2 teardown
so a killed driver or worker cannot leave warmup objects, failed output objects,
or incomplete multipart uploads behind.  Only arms with a downloaded,
validated successful result and all three commit markers are retained.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


ARMS = ("original", "tuned", "fast")
SPLITS = ("train", "val", "test")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    key = parsed.path.strip("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or parsed.query
        or parsed.fragment
        or any(part in {".", ".."} for part in key.split("/"))
    ):
        raise ValueError(f"Unsafe S3 URI: {uri!r}")
    return parsed.netloc, key


def delete_prefix(client, bucket: str, prefix: str) -> int:
    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", ())]
        for start in range(0, len(objects), 1000):
            batch = objects[start : start + 1000]
            if batch:
                response = client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": batch, "Quiet": True},
                )
                errors = response.get("Errors", ())
                if errors:
                    raise RuntimeError(f"S3 delete errors: {errors!r}")
                deleted += len(batch)
    return deleted


def abort_multipart_uploads(client, bucket: str, prefix: str) -> int:
    aborted = 0
    paginator = client.get_paginator("list_multipart_uploads")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for upload in page.get("Uploads", ()):
            client.abort_multipart_upload(
                Bucket=bucket,
                Key=upload["Key"],
                UploadId=upload["UploadId"],
            )
            aborted += 1
    return aborted


def marker_exists(client, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def result_allows_retention(
    results_dir: Path,
    *,
    run_id: str,
    arm: str,
    expected_output_uri: str,
) -> bool:
    candidates = (
        results_dir / f"{arm}.json",
        results_dir / arm / f"{arm}.json",
        *sorted(results_dir.rglob(f"{arm}.json")),
    )
    for path in candidates:
        if not path.exists():
            continue
        document = json.loads(path.read_text())
        return bool(
            document.get("run_id") == run_id
            and document.get("arm") == arm
            and document.get("status") == "succeeded"
            and document.get("validation", {}).get("status") == "passed"
            and document.get("uris", {}).get("output") == expected_output_uri
        )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--region", default="us-west-2")
    args = parser.parse_args()

    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError(f"Unsafe run ID: {args.run_id!r}")
    bucket, root_key = parse_s3_uri(args.output_root)
    run_prefix = f"{root_key}/{args.run_id}"

    import boto3

    client = boto3.client("s3", region_name=args.region)
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deleted": {},
        "retained_arms": [],
        "deleted_arms": [],
        "aborted_multipart_uploads": 0,
    }

    report["deleted"]["warmup"] = delete_prefix(
        client, bucket, f"{run_prefix}/warmup"
    )
    report["deleted"]["smoke"] = delete_prefix(
        client, bucket, f"{run_prefix}/smoke"
    )
    # Tuning screens and full-train confirmations are deliberately disposable.
    # This is also the local post-EC2 backstop for a driver killed between a
    # trial write and its in-cluster finally block.
    report["deleted"]["tuning"] = delete_prefix(
        client, bucket, f"{run_prefix}/tuning"
    )

    for arm in ARMS:
        output_prefix = f"{run_prefix}/outputs/{arm}"
        output_uri = f"s3://{bucket}/{output_prefix}"
        result_ok = result_allows_retention(
            args.results_dir,
            run_id=args.run_id,
            arm=arm,
            expected_output_uri=output_uri,
        )
        markers_ok = all(
            marker_exists(
                client,
                bucket,
                f"{output_prefix}/{split}/_SUCCESS.json",
            )
            for split in SPLITS
        )
        if result_ok and markers_ok:
            report["retained_arms"].append(arm)
        else:
            report["deleted_arms"].append(arm)
            report["deleted"][f"output_{arm}"] = delete_prefix(
                client, bucket, output_prefix
            )

    report["aborted_multipart_uploads"] = abort_multipart_uploads(
        client, bucket, run_prefix
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
