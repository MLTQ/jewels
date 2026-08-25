# Execution note

- Physical RTX 2070 Super: `GPU-4e207c93-ed93-c35e-f0f2-e37c8df2b047`.
- Physical RTX 4090: `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`.
- All launches used hardware UUIDs rather than CUDA ordinals.
- The 2070 was initially free. With explicit user authorization, the exact enabled
  `scroller-comfyui.service` and one identified LM Studio model child were stopped/unloaded to free
  the 4090; the LM Studio application itself was left running.
- Matched controls sometimes ran concurrently across the two devices for screening. Every load-
  bearing exact comparison was rerendered in one shared 4090 process.
- One audit exposed that an old derivative checkpoint without scale metadata was being interpreted
  with a new scale default. That audit column was discarded. Constructor fallback was restored to
  scale 1, new training explicitly checkpoints scale 32, and `audit_final_seed0_400` contains only
  semantically compatible checkpoints.
- GPU runtime is operational disclosure, not model evidence.
