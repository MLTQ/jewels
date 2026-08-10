# `ltx_scaffold.py`

## Purpose

Runs the official LTX-2.3 distilled pipeline as the first prompt-generated semantic scaffold for
Jewels. It pins generation geometry, GPU visibility, quantization, offload, seed, upstream revision,
and video metadata in a JSON receipt.

## Components

### `ScaffoldConfig`

- **Does**: Defines one reproducible LTX generation and all external asset paths.
- **Rationale**: Prompt, video geometry, model variant, and memory policy are experimental inputs,
  not ambient shell state.

### `validate_config`

- **Does**: Enforces LTX's `64`-pixel geometry and `8*K+1` frame contracts and checks model assets.
- **Interacts with**: The official `ltx_pipelines.distilled` CLI.

### `build_command`

- **Does**: Constructs an argument vector without shell interpolation.
- **Rationale**: Prompts may contain punctuation and quotes; they must remain a single argument.

### `run_scaffold`

- **Does**: Pins the selected CUDA device and expandable allocator, launches LTX, probes the MP4,
  samples aggregate GPU memory/utilization once per second, and records success or failure
  provenance.
- **Interacts with**: `ffprobe` and the external LTX repository revision.

### `_read_gpu_sample`

- **Does**: Queries `nvidia-smi` by stable GPU UUID and returns memory/utilization measurements.
- **Rationale**: The smoke gate needs measured peak VRAM, not an estimate from parameter count.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Aine smoke run | Official repo at `/home/m/LTX-2` with the distilled v1.1 assets | Default paths |
| Jewel guide loader | A playable RGB MP4 with explicit frame count and resolution | Output container |
| Research audit | JSON receipt includes prompt, seed, command, revision, runtime, GPU peak, and probe | Receipt schema |

## Aine baseline

- RTX 4090 UUID: `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`.
- First gate: `512x768`, 49 frames, 24 fps, `fp8-cast`, CPU offload, batch size 1.
- Required upstream revision: `4f8905737aac86a554637cac86c178877a39c744` until deliberately
  upgraded.
- LTX's 46.15 GB distilled checkpoint, 1.0 GB upscaler, and five-shard 24.38 GB Gemma encoder are
  installed. Hugging Face verified each file while completing the gated download.

The official repository is installed at `/home/m/LTX-2`; the launch harness is deployed at
`/home/m/jewels/sol/ltx_scaffold.py`. Its Aine-side lint, unit test, and dry-run gates pass.

## Usage

Run the first bounded scaffold from `/home/m/jewels`:

```bash
python -m sol.ltx_scaffold \
  --prompt "A rider crosses a sunlit field from left to right, stable wide camera, realistic motion." \
  --output /home/m/LTX-2/outputs/jewels_horse_smoke.mp4 \
  --cuda-visible-device GPU-21d45575-7ece-a97c-35a0-294f7bce9c39
```

The command writes both the MP4 and a neighboring JSON receipt. GPU measurements are aggregate
device readings; `peak_above_baseline_mib` subtracts the pre-launch display footprint but should
still be treated as an operational rather than allocator-exact measurement. `--dry-run` validates
geometry and prints the exact argument vector without requiring model assets or starting CUDA work.

The first real run is queued on Aine as `jewels-ltx-smoke.service`. It waits for the pre-existing
`fable2.train` process and then for aggregate 4090 use to fall below 2 GiB before launching; this
avoids overcommitting another experiment's live CUDA allocation.

The scaffold is a teacher signal, not the final rendered state. It will be resized and aligned by
`video_guide.py`; the jewel field remains the persistent editable output.
