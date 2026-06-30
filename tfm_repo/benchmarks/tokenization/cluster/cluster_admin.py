#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Readiness, S3/KvikIO smoke, and cache administration for the AWS cluster."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import time
from pathlib import Path
from urllib.parse import urlparse


EXPECTED_VERSIONS = {
    "ray": "3.0.0.dev0",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "cudf-cu12": "25.10.0",
    "kvikio-cu12": "25.10.0",
    "cupy-cuda12x": "14.1.1",
    "pyarrow": "19.0.1",
    "s3fs": "2026.6.0",
    "boto3": "1.42.90",
    "botocore": "1.42.90",
}


def alive_nodes(ray) -> list[dict]:
    return [node for node in ray.nodes() if node.get("Alive")]


def one_per_node(ray, fn, *args, **options):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    refs = []
    for index, node in enumerate(alive_nodes(ray)):
        per_node = dict(options)
        per_node["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node["NodeID"], soft=False
        )
        refs.append(fn.options(**per_node).remote(index, *args))
    return ray.get(refs)


def instance_type() -> str | None:
    import urllib.request

    try:
        token_request = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_request, timeout=2).read().decode()
        request = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(request, timeout=2).read().decode()
    except Exception:
        return None


def wait_for_cluster(ray, timeout_s: int) -> dict:
    deadline = time.monotonic() + timeout_s
    last = {}
    while time.monotonic() < deadline:
        nodes = alive_nodes(ray)
        resources = ray.cluster_resources()
        last = {
            "nodes": len(nodes),
            "cpus": int(resources.get("CPU", 0)),
            "gpus": int(resources.get("GPU", 0)),
        }
        if last == {"nodes": 4, "cpus": 64, "gpus": 4}:
            return last
        print(f"Waiting for four-node cluster: {last}", flush=True)
        time.sleep(10)
    raise TimeoutError(f"Cluster did not reach 4 nodes/64 CPUs/4 GPUs: {last}")


def inventory(ray) -> dict:
    @ray.remote(num_cpus=0)
    def probe(_: int) -> dict:
        from importlib import metadata
        import ray as ray_module

        gpu_name = None
        gpu_memory = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            pynvml.nvmlShutdown()
        except Exception:
            pass
        versions = {}
        for distribution in EXPECTED_VERSIONS:
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                versions[distribution] = None
        nvme = shutil.disk_usage("/mnt/nvme")
        return {
            "hostname": socket.gethostname(),
            "instance_type": instance_type(),
            "gpu": gpu_name,
            "gpu_memory_bytes": gpu_memory,
            "versions": versions,
            "ray_commit": getattr(ray_module, "__commit__", None),
            "nvme_total_bytes": nvme.total,
            "nvme_free_bytes": nvme.free,
        }

    per_node = one_per_node(ray, probe)
    result = {
        "resources": ray.cluster_resources(),
        "nodes": per_node,
    }
    if len(per_node) != 4:
        raise RuntimeError(f"Expected four nodes, found {len(per_node)}")
    for node in per_node:
        if node["instance_type"] != "g6.4xlarge":
            raise RuntimeError(f"Unexpected instance type: {node}")
        if "L4" not in (node["gpu"] or ""):
            raise RuntimeError(f"Expected NVIDIA L4: {node}")
        if node["nvme_total_bytes"] < 500_000_000_000:
            raise RuntimeError(f"Expected the 600 GB instance NVMe mount: {node}")
        if node["versions"] != EXPECTED_VERSIONS:
            raise RuntimeError(
                f"Pinned package inventory mismatch: expected={EXPECTED_VERSIONS}, "
                f"actual={node['versions']}"
            )
        if node["ray_commit"] != "85bdbaa879fc6a78c5d80256bb677fe2fb297430":
            raise RuntimeError(f"Unexpected Ray wheel commit: {node}")
    return result


def smoke(ray, input_uri: str, output_root: str, run_id: str) -> list[dict]:
    parsed_input = urlparse(input_uri.rstrip("/") + "/train.parquet")
    parsed_output = urlparse(output_root)
    if parsed_input.scheme != "s3" or parsed_output.scheme != "s3":
        raise ValueError("Smoke paths must use s3://")

    @ray.remote(num_cpus=1, num_gpus=1)
    def probe(node_index: int, input_value: str, output_value: str, rid: str) -> dict:
        import boto3
        from botocore.exceptions import ClientError

        os.environ["CUDF_KVIKIO_REMOTE_IO"] = "1"
        os.environ["KVIKIO_NTHREADS"] = "8"
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise RuntimeError("EC2 role credentials are unavailable")
        frozen = credentials.get_frozen_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
        if frozen.token:
            os.environ["AWS_SESSION_TOKEN"] = frozen.token

        input_parsed = urlparse(input_value.rstrip("/") + "/train.parquet")
        output_parsed = urlparse(output_value)
        key = (
            f"{output_parsed.path.lstrip('/').rstrip('/')}/{rid}/smoke/"
            f"{socket.gethostname()}-{node_index}.json"
        )
        client = boto3.client("s3", region_name="us-west-2")
        body = json.dumps({"host": socket.gethostname()}).encode()
        try:
            client.put_object(
                Bucket=output_parsed.netloc,
                Key=key,
                Body=body,
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )
            loaded = client.get_object(
                Bucket=output_parsed.netloc, Key=key
            )["Body"].read()
            if loaded != body:
                raise RuntimeError("S3 smoke round trip changed the payload")

            # Match the measured fast path: direct s3:// URI, no fsspec
            # filesystem argument, and a hard check that the KvikIO remote
            # extension exists. Passing s3fs here would bypass the backend we
            # are trying to validate.
            from kvikio.remote_file import is_remote_file_available

            if not is_remote_file_available():
                raise RuntimeError("KvikIO remote-file support is unavailable")
            import cudf

            cudf.set_option("kvikio_remote_io", True)
            if not cudf.get_option("kvikio_remote_io"):
                raise RuntimeError("cuDF KvikIO remote I/O is inactive")
            frame = cudf.read_parquet(
                input_value.rstrip("/") + "/train.parquet",
                engine="cudf",
                columns=["User"],
                row_groups=[node_index],
                dataset_kwargs={"partitioning": None},
                use_pandas_metadata=False,
                categorical_partitions=False,
            )
            rows = len(frame)
            del frame
        finally:
            # The smoke prefix must remain empty even when the read fails.
            client.delete_object(Bucket=output_parsed.netloc, Key=key)
            try:
                client.head_object(Bucket=output_parsed.netloc, Key=key)
            except ClientError as exc:
                error_code = str(exc.response.get("Error", {}).get("Code", ""))
                if error_code not in {"404", "NoSuchKey", "NotFound"}:
                    raise
            else:
                raise RuntimeError(f"S3 smoke object still exists after delete: {key}")
        if rows <= 0:
            raise RuntimeError("KvikIO smoke read returned no rows")
        return {
            "host": socket.gethostname(),
            "row_group": node_index,
            "rows": rows,
            "kvikio_remote_io": True,
            "kvikio_remote_file_available": True,
        }

    return one_per_node(
        ray,
        probe,
        input_uri,
        output_root,
        run_id,
    )


def clear_caches(ray) -> list[dict]:
    @ray.remote(num_cpus=0)
    def clear(_: int) -> dict:
        removed = []
        for path in (
            os.path.expanduser("~/.cupy/kernel_cache"),
            os.path.expanduser("~/.cache/cupy"),
        ):
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path)
        return {"host": socket.gethostname(), "removed": removed}

    return one_per_node(ray, clear)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    wait_parser = subparsers.add_parser("wait")
    wait_parser.add_argument("--timeout-seconds", type=int, default=1200)
    wait_parser.add_argument("--output-json", type=Path)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--input-uri", required=True)
    smoke_parser.add_argument("--output-root", required=True)
    smoke_parser.add_argument("--run-id", required=True)
    smoke_parser.add_argument("--output-json", type=Path)
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--output-json", type=Path)
    clear_parser = subparsers.add_parser("clear-caches")
    clear_parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    import ray

    ray.init(address="auto", logging_level="ERROR")
    try:
        if args.command == "wait":
            result = wait_for_cluster(ray, args.timeout_seconds)
        elif args.command == "inventory":
            result = inventory(ray)
        elif args.command == "smoke":
            wait_for_cluster(ray, 1200)
            inventory(ray)
            result = smoke(ray, args.input_uri, args.output_root, args.run_id)
        else:
            result = clear_caches(ray)
        rendered = json.dumps(result, indent=2, sort_keys=True, default=str)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n")
        print(rendered)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
