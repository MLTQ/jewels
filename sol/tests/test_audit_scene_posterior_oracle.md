# `test_audit_scene_posterior_oracle.py`

## Purpose

Protects the source ownership of the deliberately leaky posterior-oracle diagnostic.

## Components

### `ScenePosteriorOracleTests`

- **Does**: Selects one stable source per exact style/action key and rejects missing report-owned
  sources before any metric is computed.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1i | Oracle sources come only from registered training rows | Source selection |
