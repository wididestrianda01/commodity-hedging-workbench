"""Lambda handler: run the refresh job, upload the gated bundle to S3.

Kept in the pipeline layer — the package's refresh job stays upload-free.
Bucket comes from env REFRESH_BUCKET (set by Terraform). Prints progress so
runs are observable in CloudWatch logs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import boto3

from hedging_workbench.refresh import refresh


def _upload(bucket: str, out: Path) -> list[str]:
    s3 = boto3.client("s3")
    keys = []
    for path in sorted(out.iterdir()):
        if path.is_file():
            key = path.name
            s3.upload_file(str(path), bucket, key)
            keys.append(key)
    return keys


def handler(event: dict, context) -> dict:
    bucket = os.environ["REFRESH_BUCKET"]
    print(f"refresh: start (bucket={bucket})")
    out = Path("/tmp/refresh-out")
    # refresh() is fail-closed: a gate failure raises before ANY file is
    # written, the invocation fails, and the schedule surfaces it in logs.
    # A gate_report.json in `out` therefore implies ok=True.
    refresh(start=event.get("start", "2024-01-01"), out_dir=out)
    report = json.loads((out / "gate_report.json").read_text())
    print(f"refresh: gates {report['summary']}")
    keys = _upload(bucket, out)
    print(f"refresh: ok universe={report['universe']} keys={len(keys)}")
    return {"bucket": bucket, "keys": keys, "universe": report["universe"]}
