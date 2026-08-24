# Execution note

- Physical RTX 2070 Super UUID: `GPU-4e207c93-ed93-c35e-f0f2-e37c8df2b047`.
- Physical RTX 4090 UUID: `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`.
- Commands used hardware UUIDs rather than CUDA ordinals after the launch safety review identified
  the machine's previously observed ordinal ambiguity.
- Read-only gradient calibration ran on the idle 4090. It constructed no optimizer and took no
  optimizer step.
- `scroller-llm.service` was initially stopped and runtime-masked to release the 2070. Seed-0 arms,
  seed-0 continuation, and seed-1 replication ran on the 2070.
- The user service manager later lost or bypassed the runtime mask and restarted the llama server at
  approximately 03:47 local time. Seed 2's first 2070 launch then failed with CUDA OOM before its
  first optimizer step or checkpoint.
- A second service stop was rejected by the execution safety layer without fresh explicit approval.
  The failed seed-2 output was excluded and rerun from the registered source on the available 4090;
  both seed-2 tranches completed normally.
- Seed-0 preliminary audits ran on the 2070. The final shared source/three-seed exact audit ran on the
  4090, so all final PSNR/LPIPS comparisons share one process and device.
- Cross-device repetition of the source LPIPS differed by less than `0.00005`; the final shared audit
  is authoritative.
- At session end the llama service was already active due its external restart. No stop, disable,
  or persistent mask remains to restore. Service state is verified separately before handoff.
- GPU choice and wall time are operational disclosures, not evidence about representation scaling.
