# LTX-2.3 scaffold corpus v1

The first prompt-generated scaffold corpus completed on Aine's RTX 4090 on August 11, 2026. It is
a balanced semantic gate for the video-to-jewel realizer: three training phrasings and one held-out
evaluation phrasing for each of Basketball, HorseRiding, PlayingGuitar, and ApplyEyeMakeup.

## Result

| Measurement | Result |
|---|---:|
| Completed / failed | 16 / 0 |
| Geometry | 768x512, 49 frames, 24 fps |
| Duration per clip | 2.041667 s |
| Video / audio | H.264; 48 kHz stereo AAC |
| Aggregate generation time | 2,767.74 s (46.13 min) |
| Mean / min / max per clip | 172.98 / 156.77 / 206.42 s |
| Mean GPU peak | 9,399.31 MiB |
| GPU-peak range | 9,398--9,401 MiB |
| Source-manifest SHA-256 | `5e514966be50240393ff51a65327165e8794edc4898cdecc02269adc982be578` |

All 16 files independently report 768x512 H.264, 49 frames at 24 fps, and 48 kHz stereo AAC.
Receipts and the authoritative generated-corpus manifest remain at
`/home/m/LTX-2/corpora/jewels_ucf_prompt_v1` on Aine.

## Visual audit

Each row below is one prompt/seed sampled at frames 0, 16, 32, and 48. All 16 samples preserve the
requested action and a coherent primary subject over the two-second window. Basketball includes
recognizable ball handling, shooting, and court scenes; HorseRiding preserves horse/rider topology;
PlayingGuitar preserves the instrument and playing pose; ApplyEyeMakeup preserves the close facial
layout, hand, and cosmetic tool. Fine hand/object interactions remain imperfect, as expected from a
general video prior, but the macro-geometry is materially stronger than the washed-out direct jewel
generator.

- [Basketball audit](00_basketball_audit.jpg)
- [HorseRiding audit](01_horseriding_audit.jpg)
- [PlayingGuitar audit](02_playingguitar_audit.jpg)
- [ApplyEyeMakeup audit](03_applyeyemakeup_audit.jpg)

## Throughput interpretation

The observed low average GPU activity is real but not a failed run. Each distilled sample performs
eight half-resolution and three full-resolution denoising steps in about 14 seconds; most of the
roughly 173-second wall time repeatedly streams or rebuilds Gemma, the 22B transformer, upsampler,
and decoders under CPU offload. The official pipeline releases these components inside each call,
so merely retaining a `DistilledPipeline` Python object does not remove the loading cost.

Before scaling by orders of magnitude, benchmark either two concurrent CPU-offloaded workers
(their independent 9.4 GiB peaks should fit in 24 GiB, but shared RAM/PCIe behavior is unmeasured) or
a prompt-embedding/model-lifecycle refactor. The semantic-scaffold gate itself passes.
