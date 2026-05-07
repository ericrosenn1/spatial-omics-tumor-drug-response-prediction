# model_training RUNBOOK

This runbook describes how to inspect and validate model_training without rerunning the pipelines.

## Active modules

- expression_response_model_v2
- histology_response_model_v2

## Smoke test

From model_training/:

    .\tests\smoke_test_model_training.ps1

The smoke test checks file presence, Python compilation, YAML parsing, and selected key TSV readability.

It does not rerun model training.
