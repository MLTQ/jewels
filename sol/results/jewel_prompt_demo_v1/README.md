# Prompt-to-Jewel video demo v1

## What is here

This directory turns the passing prompt-only trajectory-speaker proof into complete playable videos
and a small browser demo. Every MP4 has 49 frames at 216×144, 12 fps, H.264/yuv420p, and a paired
JSON file recording the emitted program, source-backed macro tokens, seed, scene, exact 72,000-Jewel
count, and render settings.

## Exact proof exports

| Prompt | Seed | File |
|---|---:|---|
| a ballerina spinning a pirouette in a studio | 20260914 | `generated/a-ballerina-spinning-a-pirouette-in-seed20260914-exact-940a8520810e.mp4` |
| a golden retriever catching a ball on grass | 20260914 | `generated/a-golden-retriever-catching-a-ball-seed20260914-exact-5e3ffe08b31a.mp4` |
| a welder joining steel with bright sparks | 20260914 | `generated/a-welder-joining-steel-with-bright-seed20260914-exact-c4522ccdb03d.mp4` |

These use the passing Gate 2b0 exact speaker. Their only semantic inputs are the registered prompt
and declared seed.

## Learned demo export

`generated/an-illustrated-dog-leaps-for-a-seed20260914-learned-5a4bb83ac90e.mp4`
was produced through the browser UI from: “an illustrated dog leaps for a red ball in a sunny park.”
The learned speaker emitted the `cartoon` scene and two distinct source-level tokens.

This is useful evidence that nearby free-form wording reaches the program speaker, but it is not an
open-vocabulary result: the learned model can still emit only three scene families and 18
source-backed macro tokens.

## Demo deployment

- GPU service: `jewels-prompt-demo.service` on `m@192.168.0.202`; disabled after the demo at the
  user's request
- Verified deployment GPU: PyTorch `cuda:0`, which maps to the physical RTX 4090 in the shared
  environment
- Host endpoint: `0.0.0.0:7860` (protected by the host firewall)
- Local tunnel endpoint: `http://127.0.0.1:7860`
- Output directory on GPU: `/home/m/jewels-codex-support/sol/results/jewel_prompt_demo_v1/generated`

The interface provides exact proven and learned experimental modes, canonical prompt shortcuts,
seed control, an in-page video player, MP4 download, and expandable program provenance.

The process appeared as `/home/m/LTX-2/.venv/bin/python` in GPU monitors because that existing
environment supplied PyTorch/OpenCLIP dependencies. Inspection found no LTX-2/diffusion imports or
open LTX-2 weights. The service was nevertheless consuming the 4090 renderer workspace, so it is
now stopped and disabled; a dedicated environment plus explicit 2070 validation is follow-up work.

## Verification

- Focused runtime/demo tests: 5 passed.
- All four retained MP4s: 49 decoded frames, 216×144, 12 fps, approximately 4.08 seconds.
- Browser status endpoint: ready; learned speaker available.
- Browser byte-range request: HTTP 206 with the correct `Content-Range`.
- Browser interaction: exact ballerina and free-form dog prompts both generated playable videos;
  the latter emitted `scene 1 / cartoon`, foreground token 11, background token 7, and 72,000
  Jewels.
