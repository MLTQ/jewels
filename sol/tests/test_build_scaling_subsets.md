# `test_build_scaling_subsets.py`

## Purpose

Pins the two invariants the scaling curve depends on: validation examples survive every subset
unchanged, and a subset that would drop or unbalance a class is refused instead of silently
shrinking the prompt space.

## Components

### `test_subset_keeps_validation_and_balances_classes`
- **Does**: A group-2 cutoff keeps two train groups per class, all validation rows, and records
  accurate provenance counts.

### `test_subset_rejects_class_drop`
- **Does**: Removing one class's only group-1 source makes the cutoff unbalanced or
  class-dropping; the builder must raise.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `build_scaling_subsets.py` | Filter and validation rules stay exactly these | Subset semantics |
