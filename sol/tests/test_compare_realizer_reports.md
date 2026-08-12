# `test_compare_realizer_reports.py`

## Purpose

Protects matched-source aggregation and paired-delta reporting for video-to-jewel ablations.

## Components

### `CompareRealizerReportsTests`

- **Does**: Verifies sorted source pairing, arithmetic macro means, signed per-source deltas, and
  rejection of mismatched source sets or missing panels.
- **Interacts with**: `compare_realizer_reports.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research decision | Candidate and baseline compare the same sources | Pairing validation |
| Paper metrics | Macro results are reproducible from report JSON | Aggregation math |
