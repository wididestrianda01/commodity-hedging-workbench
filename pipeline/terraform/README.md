# Phase 8 refresh pipeline — apply / invoke / teardown (floci emulator)

## One-time

    docker build -f pipeline/Dockerfile.lambda -t workbench-refresh-lambda:latest .

## Apply

    cd pipeline/terraform
    tofu init
    tofu apply          # deploys S3 bucket + container Lambda against floci :4566

## Invoke (manual, 8-02)

    curl -s -X POST "http://localhost:4566/2015-03-31/functions/workbench-refresh/invocations" \
      -d '{}'

Then list the artifacts:

    curl -s "http://localhost:4566/workbench-gated-snapshots" | head -c 2000


## Scheduled run + logs (8-03)

`tofu apply` also creates the EventBridge Scheduler schedule
(`workbench-refresh-nightly`, `rate(1 day)`) targeting the Lambda. Read the
Lambda's CloudWatch logs (last 20 events):

    /tmp/awsv/bin/python -c "import boto3; c=boto3.client('logs', endpoint_url='http://localhost:4566', region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test'); [print(e['message'][:150].rstrip()) for e in c.filter_log_events(logGroupName='/aws/lambda/workbench-refresh', limit=20)['events']]"

## Fail-closed proof

Invoke with a too-recent start date so every series falls under the
120-row gate — the invocation must error and NO snapshot may be promoted:

    curl -s -X POST "http://localhost:4566/2015-03-31/functions/workbench-refresh/invocations" \
      -d '{"start":"2026-08-01"}'
    # → errorMessage: "...gates both failed — fail closed, no curve."
    # bucket object count unchanged

## Teardown

    tofu destroy        # removes Lambda + bucket from the emulator

## Honest framing

All of the above is **emulator-based (floci)** — nothing leaves the machine and
nothing is claimed as production AWS. Real-account deployment is an optional
HITL follow-up under the zero-spend guardrail (plan.md G3).
