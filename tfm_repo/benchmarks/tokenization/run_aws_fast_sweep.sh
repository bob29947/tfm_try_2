#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CLUSTER_CONFIG_TEMPLATE="$SCRIPT_DIR/cluster/ray-tokenization-s3-4xl4.yaml"
CLUSTER_NAME_BASE="ray-tfm-fast-sweep-4xl4-bobbwang"
CLUSTER_ADMIN="benchmarks/tokenization/cluster/cluster_admin.py"
SWEEP="benchmarks/tokenization/fast_cloud_sweep.py"
INPUT_URI="${TFM_BENCH_INPUT_URI:-s3://bobbwang-tfm-gpu-e2e/tfm_try_2/input/temporal_split_v3}"
OUTPUT_ROOT="${TFM_BENCH_OUTPUT_ROOT:-s3://bobbwang-tfm-gpu-e2e/tfm_try_2/benchmarks}"
GIT_SHA="$(git -C "$WORKSPACE_ROOT" rev-parse HEAD)"
RUN_ID="${TFM_SWEEP_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-fast-${GIT_SHA:0:10}}"
SCREEN_ROW_GROUPS="${TFM_SWEEP_SCREEN_ROW_GROUPS:-32}"
CONFIRMATION_TOP="${TFM_SWEEP_CONFIRMATION_TOP:-4}"
FINAL_OVERLAP="${TFM_SWEEP_FINAL_OVERLAP:-1}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || ${#RUN_ID} -gt 64 ]]; then
    echo "Unsafe or overly long sweep run ID: $RUN_ID" >&2
    exit 2
fi
if [[ ! "$SCREEN_ROW_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TFM_SWEEP_SCREEN_ROW_GROUPS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$CONFIRMATION_TOP" =~ ^[0-9]+$ ]]; then
    echo "TFM_SWEEP_CONFIRMATION_TOP must be a nonnegative integer" >&2
    exit 2
fi
if [[ "$FINAL_OVERLAP" != "0" && "$FINAL_OVERLAP" != "1" ]]; then
    echo "TFM_SWEEP_FINAL_OVERLAP must be 0 or 1" >&2
    exit 2
fi

CLUSTER_NAME="${CLUSTER_NAME_BASE}-${RUN_ID}"
LOCAL_RESULTS="$SCRIPT_DIR/results/aws/$RUN_ID"
CLUSTER_CONFIG="$LOCAL_RESULTS/cluster.yaml"
REMOTE_RESULTS="/home/ray/benchmark_results/$RUN_ID"
ARTIFACT_DIR="$WORKSPACE_ROOT/.artifacts"
CODE_BUNDLE="$ARTIFACT_DIR/tfm-fast-sweep-bundle-$RUN_ID"
WHEEL_BUNDLE="$ARTIFACT_DIR/ray-wheel-bundle-$RUN_ID"
RAY_WHEEL="$ARTIFACT_DIR/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"
RAY_WHEEL_URL="https://s3-us-west-2.amazonaws.com/ray-wheels/master/85bdbaa879fc6a78c5d80256bb677fe2fb297430/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"
RAY_WHEEL_SHA256="a029e177155d08c4c1a85d5996a87b90f3be3bb31972f533595b36babc1965c3"
LAUNCHER_VENV="$WORKSPACE_ROOT/.venv-aws"
RAY_CLI="$LAUNCHER_VENV/bin/ray"
CLUSTER_STARTED=0

cd "$WORKSPACE_ROOT"
mkdir -p "$LOCAL_RESULTS" "$ARTIFACT_DIR"
cp "$0" "$LOCAL_RESULTS/run_aws_fast_sweep.sh"

# Stage only executable source and the exact benchmark modules. The generated
# path is unique to this run, so a later local edit cannot mutate a live mount.
CODE_BUNDLE="$CODE_BUNDLE" \
"$WORKSPACE_ROOT/.venv/bin/python" - <<'PY'
import os
import shutil
from pathlib import Path

root = Path.cwd()
bundle = Path(os.environ["CODE_BUNDLE"])
shutil.rmtree(bundle, ignore_errors=True)
shutil.copytree(
    root / "tfm_repo" / "src",
    bundle / "src",
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
)
files = (
    "benchmarks/__init__.py",
    "benchmarks/tokenization/__init__.py",
    "benchmarks/tokenization/cloud_benchmark.py",
    "benchmarks/tokenization/run_cloud_benchmark.py",
    "benchmarks/tokenization/fast_cloud_sweep.py",
    "benchmarks/tokenization/cluster/cluster_admin.py",
)
for relative in files:
    destination = bundle / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "tfm_repo" / relative, destination)
PY

CLUSTER_CONFIG_TEMPLATE="$CLUSTER_CONFIG_TEMPLATE" \
CLUSTER_CONFIG="$CLUSTER_CONFIG" \
CLUSTER_NAME="$CLUSTER_NAME" \
CODE_BUNDLE="$CODE_BUNDLE" \
WHEEL_BUNDLE="$WHEEL_BUNDLE" \
"$WORKSPACE_ROOT/.venv/bin/python" - <<'PY'
import os
from pathlib import Path

import yaml

source = Path(os.environ["CLUSTER_CONFIG_TEMPLATE"])
destination = Path(os.environ["CLUSTER_CONFIG"])
cluster_name = os.environ["CLUSTER_NAME"]
document = yaml.safe_load(source.read_text())
document["cluster_name"] = cluster_name
document["file_mounts"]["/home/ray/tfm"] = os.environ["CODE_BUNDLE"]
document["file_mounts"]["/home/ray/ray-wheel"] = os.environ["WHEEL_BUNDLE"]
for node_type in document["available_node_types"].values():
    for specification in node_type["node_config"].get("TagSpecifications", []):
        for tag in specification.get("Tags", []):
            if tag.get("Key") == "Cluster":
                tag["Value"] = cluster_name
destination.write_text(yaml.safe_dump(document, sort_keys=False))
PY

git -C "$WORKSPACE_ROOT" status --porcelain=v1 >"$LOCAL_RESULTS/source-status.txt"
git -C "$WORKSPACE_ROOT" diff --binary --no-ext-diff >"$LOCAL_RESULTS/source.patch"
tar -czf "$LOCAL_RESULTS/source-bundle.tar.gz" \
    tfm_repo/src \
    tfm_repo/benchmarks/__init__.py \
    tfm_repo/benchmarks/tokenization/__init__.py \
    tfm_repo/benchmarks/tokenization/cloud_benchmark.py \
    tfm_repo/benchmarks/tokenization/run_cloud_benchmark.py \
    tfm_repo/benchmarks/tokenization/fast_cloud_sweep.py \
    tfm_repo/benchmarks/tokenization/prepare_s3_input.py \
    tfm_repo/benchmarks/tokenization/cleanup_s3_run.py \
    tfm_repo/benchmarks/tokenization/cluster/ray-tokenization-s3-4xl4.yaml \
    tfm_repo/benchmarks/tokenization/cluster/cluster_admin.py

write_run_metadata() {
    local status="$1"
    STATUS="$status" RUN_ID_VALUE="$RUN_ID" \
    INPUT_URI_VALUE="$INPUT_URI" OUTPUT_ROOT_VALUE="$OUTPUT_ROOT" \
    CLUSTER_NAME_VALUE="$CLUSTER_NAME" GIT_SHA_VALUE="$GIT_SHA" \
    "$WORKSPACE_ROOT/.venv/bin/python" - "$LOCAL_RESULTS/orchestration.json" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "schema_version": 1,
    "kind": "aws-fast-cloud-sweep",
    "run_id": os.environ["RUN_ID_VALUE"],
    "status": os.environ["STATUS"],
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "git_sha": os.environ["GIT_SHA_VALUE"],
    "input_uri": os.environ["INPUT_URI_VALUE"],
    "output_root": os.environ["OUTPUT_ROOT_VALUE"],
    "cluster_name": os.environ["CLUSTER_NAME_VALUE"],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

sync_results() {
    if [[ "$CLUSTER_STARTED" -eq 1 ]]; then
        "$RAY_CLI" rsync-down "$CLUSTER_CONFIG" "$REMOTE_RESULTS/" "$LOCAL_RESULTS/" \
            >>"$LOCAL_RESULTS/rsync.log" 2>&1
    fi
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    sync_results || true
    local teardown_failed=0
    if [[ "$CLUSTER_STARTED" -eq 1 ]]; then
        "$RAY_CLI" down "$CLUSTER_CONFIG" -y \
            >"$LOCAL_RESULTS/ray_down.log" 2>&1

        # Ray teardown can fail if the head/container is unhealthy. The exact,
        # run-unique cluster tag is a narrow and deterministic final backstop.
        remaining_ids="$(aws ec2 describe-instances \
            --region us-west-2 \
            --filters \
                "Name=tag:ray-cluster-name,Values=$CLUSTER_NAME" \
                "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
            --query 'Reservations[].Instances[].InstanceId' \
            --output text 2>>"$LOCAL_RESULTS/ray_down.log")"
        if [[ "$?" -ne 0 ]]; then
            teardown_failed=1
            remaining_ids=""
        fi
        if [[ -n "$remaining_ids" && "$remaining_ids" != "None" ]]; then
            read -r -a remaining_array <<<"$remaining_ids"
            aws ec2 terminate-instances --region us-west-2 \
                --instance-ids "${remaining_array[@]}" \
                >>"$LOCAL_RESULTS/ray_down.log" 2>&1
            [[ "$?" -eq 0 ]] || teardown_failed=1
            aws ec2 wait instance-terminated --region us-west-2 \
                --instance-ids "${remaining_array[@]}" \
                >>"$LOCAL_RESULTS/ray_down.log" 2>&1
            [[ "$?" -eq 0 ]] || teardown_failed=1
        fi

        final_ids="$(aws ec2 describe-instances \
            --region us-west-2 \
            --filters \
                "Name=tag:ray-cluster-name,Values=$CLUSTER_NAME" \
                "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
            --query 'Reservations[].Instances[].InstanceId' \
            --output text 2>>"$LOCAL_RESULTS/ray_down.log")"
        if [[ "$?" -ne 0 || ( -n "$final_ids" && "$final_ids" != "None" ) ]]; then
            echo "Teardown postcondition failed; remaining instances: $final_ids" \
                >>"$LOCAL_RESULTS/ray_down.log"
            teardown_failed=1
        fi

        # Runs after EC2 termination so no worker can race this local backstop.
        # It always deletes tuning/warmup/smoke and retains final fast output
        # only when the downloaded result is successful and fully validated.
        "$LAUNCHER_VENV/bin/python" "$SCRIPT_DIR/cleanup_s3_run.py" \
            --output-root "$OUTPUT_ROOT" \
            --run-id "$RUN_ID" \
            --results-dir "$LOCAL_RESULTS" \
            --report "$LOCAL_RESULTS/s3_cleanup.json" \
            >>"$LOCAL_RESULTS/s3_cleanup.log" 2>&1
        [[ "$?" -eq 0 ]] || teardown_failed=1
    fi
    if [[ "$teardown_failed" -ne 0 ]]; then
        exit_code=1
    fi
    if [[ "$exit_code" -eq 0 ]]; then
        write_run_metadata complete
    else
        write_run_metadata failed
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

write_run_metadata preparing

if [[ ! -s "$RAY_WHEEL" ]]; then
    echo "Downloading exact Ray wheel..."
    curl --fail --location --retry 5 --retry-delay 2 \
        --output "$RAY_WHEEL.partial" "$RAY_WHEEL_URL"
    mv "$RAY_WHEEL.partial" "$RAY_WHEEL"
fi
actual_wheel_sha256="$(sha256sum "$RAY_WHEEL" | awk '{print $1}')"
if [[ "$actual_wheel_sha256" != "$RAY_WHEEL_SHA256" ]]; then
    echo "Ray wheel SHA-256 mismatch: $actual_wheel_sha256" >&2
    exit 1
fi
printf '%s  %s\n' "$actual_wheel_sha256" "$RAY_WHEEL" \
    >"$LOCAL_RESULTS/ray-wheel.sha256"
rm -rf "$WHEEL_BUNDLE"
mkdir -p "$WHEEL_BUNDLE"
cp "$RAY_WHEEL" "$WHEEL_BUNDLE/"

if [[ ! -x "$LAUNCHER_VENV/bin/python" ]]; then
    python3 -m venv "$LAUNCHER_VENV"
    "$LAUNCHER_VENV/bin/python" -m pip install --upgrade pip wheel
fi
"$LAUNCHER_VENV/bin/python" -m pip install --upgrade \
    "ray[default] @ file://$RAY_WHEEL" \
    "boto3==1.42.90" "botocore==1.42.90"
"$LAUNCHER_VENV/bin/python" - <<'PY'
import importlib.metadata
import ray

assert ray.__version__ == "3.0.0.dev0", ray.__version__
assert getattr(ray, "__commit__", None) == "85bdbaa879fc6a78c5d80256bb677fe2fb297430"
assert importlib.metadata.version("boto3") == "1.42.90"
assert importlib.metadata.version("botocore") == "1.42.90"
PY

echo "Copying/verifying project S3 input..."
"$WORKSPACE_ROOT/.venv/bin/python" \
    "$SCRIPT_DIR/prepare_s3_input.py" \
    --destination-uri "$INPUT_URI" \
    --local-manifest "$LOCAL_RESULTS/input_manifest.json" \
    2>&1 | tee "$LOCAL_RESULTS/prepare_s3_input.log"

echo "Launching four g6.4xlarge nodes for focused tuning..."
launch_started="$(date +%s)"
CLUSTER_STARTED=1
"$RAY_CLI" up "$CLUSTER_CONFIG" -y \
    2>&1 | tee "$LOCAL_RESULTS/ray_up.console.log"
"$RAY_CLI" exec "$CLUSTER_CONFIG" "mkdir -p '$REMOTE_RESULTS'"
"$RAY_CLI" exec "$CLUSTER_CONFIG" \
    "cd /home/ray/tfm && python -u '$CLUSTER_ADMIN' wait --timeout-seconds 1800 --output-json '$REMOTE_RESULTS/cluster_wait.json'" \
    2>&1 | tee "$LOCAL_RESULTS/cluster_wait.console.log"
"$RAY_CLI" exec "$CLUSTER_CONFIG" \
    "cd /home/ray/tfm && python -u '$CLUSTER_ADMIN' inventory --output-json '$REMOTE_RESULTS/cluster_inventory.json'" \
    2>&1 | tee "$LOCAL_RESULTS/cluster_inventory.console.log"
launch_finished="$(date +%s)"
CLUSTER_BOOTSTRAP_SECONDS="$((launch_finished - launch_started))"
echo "$CLUSTER_BOOTSTRAP_SECONDS" >"$LOCAL_RESULTS/cluster_launch_seconds.txt"

echo "Running exact four-node S3/KvikIO smoke gate..."
"$RAY_CLI" exec "$CLUSTER_CONFIG" \
    "cd /home/ray/tfm && python -u '$CLUSTER_ADMIN' smoke --input-uri '$INPUT_URI' --output-root '$OUTPUT_ROOT' --run-id '$RUN_ID' --output-json '$REMOTE_RESULTS/smoke.json'" \
    2>&1 | tee "$LOCAL_RESULTS/smoke.console.log"
sync_results

"$RAY_CLI" exec "$CLUSTER_CONFIG" \
    "cd /home/ray/tfm && python -u '$CLUSTER_ADMIN' clear-caches --output-json '$REMOTE_RESULTS/clear_caches.json'" \
    2>&1 | tee "$LOCAL_RESULTS/clear_caches.console.log"

overlap_flag=""
if [[ "$FINAL_OVERLAP" == "1" ]]; then
    overlap_flag="--final-overlap-split-writes"
fi

echo "Running bounded screen, full-train confirmation, and final full arm..."
"$RAY_CLI" exec "$CLUSTER_CONFIG" \
    "cd /home/ray/tfm && TFM_BENCH_GIT_SHA='$GIT_SHA' python -u '$SWEEP' --ray-address auto --input-uri '$INPUT_URI' --output-root '$OUTPUT_ROOT' --run-id '$RUN_ID' --results-dir '$REMOTE_RESULTS' --screen-row-groups-per-partition '$SCREEN_ROW_GROUPS' --confirmation-top '$CONFIRMATION_TOP' --cluster-bootstrap-seconds '$CLUSTER_BOOTSTRAP_SECONDS' --run-final-full $overlap_flag" \
    2>&1 | tee "$LOCAL_RESULTS/fast_sweep.console.log"
sync_results

echo "Fast sweep complete: $LOCAL_RESULTS"
