# Appearance-contract v1 execution note

The machine exposes a different CUDA ordinal order from `nvidia-smi`: PyTorch ordinal 0 is the RTX
4090, while PyTorch ordinal 1 is the RTX 2070 Super. This was discovered by querying process UUIDs
during the continuation.

The initial read-only calibration, matched 600-step residual-control/raw-response arms, and their
exact audit used `CUDA_VISIBLE_DEVICES=0`; therefore they ran on the shared 4090 under contention,
not the intended 2070. Both arms used the same device and completed normally, so their matched
metrics remain valid. Their wall times are not used as evidence.

An early response launch with one extra validation source was interrupted during teacher loading
before model initialization or optimization and produced no result. A later response-continuation
attempt on the 4090 was interrupted at step 200 after the ordinal mismatch was diagnosed; it
produced no checkpoint and is excluded. A completed 4090 control continuation is also excluded from
selection so the continuation pair is not device-mismatched.

Official continuation outputs live under `continuation_2070/`. Both were restarted from their frozen
step-600 checkpoints with `CUDA_VISIBLE_DEVICES=1`. The control is eligible; response exceeded the
occupancy gate and was not exact-audited. Seed replication also uses ordinal 1.

`scroller-llm.service` automatically restarted after an ordinary stop, so it was given a reversible
runtime mask and stopped again. It must be runtime-unmasked and restarted at session handoff.
