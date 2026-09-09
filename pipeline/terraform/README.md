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

## Teardown

    tofu destroy        # removes Lambda + bucket from the emulator

## Honest framing

All of the above is **emulator-based (floci)** — nothing leaves the machine and
nothing is claimed as production AWS. Real-account deployment is an optional
HITL follow-up under the zero-spend guardrail (plan.md G3).
