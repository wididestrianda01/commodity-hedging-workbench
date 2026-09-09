"""Lambda handler: run the refresh job, upload the gated bundle to S3.

Kept in the pipeline layer — the package's refresh job stays upload-free.
Bucket comes from env REFRESH_BUCKET (set by Terraform).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import boto3

from hedging_workbench.refresh import refresh

BUNDLE_KEYS = ("gate_report.json", "manifest_rates.json")


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
    out = Path("/tmp/refresh-out")
    refresh(start=event.get("start", "2024-01-01"), out_dir=out)
    report = json.loads((out / "gate_report.json").read_text())
    if not report["ok"]:
        # Fail-closed: the report is still uploaded for observability, but
        # the invocation fails so the schedule surfaces the failure.
        _upload(bucket, out)
        raise RuntimeError(f"gates failed: {report['summary']}")
    keys = _upload(bucket, out)
    return {"bucket": bucket, "keys": keys, "universe": report["universe"]}
