# Four-class prompt-streaming smoke corpus

## Preflight outcome

The real UCF inventory supports a leakage-safe four-class experiment with Basketball, HorseRiding,
PlayingGuitar, and ApplyEyeMakeup. The manifest selects the longest eligible clip from source groups
1–4 in every class, yielding 12 training videos and four group-4 validation videos. Every selected
clip contains at least 109 frames for the 96-frame joint-fit contract.

Each class has three training phrasings and one unseen evaluation phrasing. All 16 phrases were
encoded with frozen OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` text weights into unit 512-D vectors.
The cache is cryptographically bound to the manifest.

## Held-out text geometry gate

Each unseen evaluation phrase is classified against class centroids formed only from that class's
three training phrasings.

| Class | Correct cosine | Best wrong | Margin |
|---|---:|---:|---:|
| Basketball | 0.8641 | 0.5881 | 0.2760 |
| HorseRiding | 0.9004 | 0.6610 | 0.2395 |
| PlayingGuitar | 0.9037 | 0.6546 | 0.2490 |
| ApplyEyeMakeup | 0.9102 | 0.6229 | 0.2873 |

Accuracy is 100%, mean margin is 0.2629, and minimum margin is 0.2395. The prompt templates and
frozen text tower therefore provide a usable semantic control signal before any jewel model is
trained.

## Dense fitting status

The 16 examples are deterministically partitioned into four balanced shards. Every shard contains
one independent source group from all four classes; their union is the exact manifest.

All 16 fits are complete on the allocated RTX 2070 SUPER with the validated 96-frame, 120k-jewel
spatial-split settings and atomic recovery every 100 steps. Exact shard-0 checkpoint replays score
32.10 dB for ApplyEyeMakeup, 33.67 dB for Basketball, 30.32 dB for HorseRiding, and 33.15 dB for
PlayingGuitar (mean 32.28 dB). Shards 1, 2, and 3 completed in 8.11, 8.10, and 8.07 hours. A read-only
capacity audit over the union finds maximum `32^3` cell occupancy 253 against 512 slots with zero
overflow.

## Shared-tokenizer smoke

The 500-step four-class tokenizer smoke passes trainability and count recovery, but not visual
quality. A fixed 4,096-point audit over all four group-4 holdouts reaches 15.890 dB macro PSNR and
95.74% count recovery. Broad palette and motion survive, while actor geometry remains diffuse. A
fresh 3,000-step run using the same leakage-safe split and source-derived motion/chroma samples
reaches 16.541 dB and 96.19% count recovery but still fails visually. Seen training windows reach
19.552 dB, so a bounded single-window overfit now separates insufficient exposure from an
intrinsically lossy cell codec. Prompt-prior training remains paused. See
[`tokenizer_smoke/README.md`](tokenizer_smoke/README.md) and
[`tokenizer_3000/README.md`](tokenizer_3000/README.md).

## Shard-0 visual replay

`previews_shard0/` contains one contact sheet, compact MP4, and metric report for each completed
field. Every video is laid out as source on the left and fitted 120k-jewel reconstruction on the
right. Contact-sheet rows are source, reconstruction, and amplified absolute error; columns sample
frames 0, 24, 48, 72, and 95. These files replay finalized checkpoints with the production kNN-64
renderer rather than rerunning optimization.

## Artifacts

- `manifest.json`: source-group ownership, prompts, encoder identity, and fit contract
- `prompts.pt`: validated normalized prompt vectors and per-example ownership
- `prompt_geometry.json`: unseen-template retrieval report
- `previews_shard0/*/compare.mp4`: source/reconstruction animations from the four completed fits
- `previews_shard0/*/contact_sheet.png`: five-timepoint source/reconstruction/error comparisons
- `previews_shard0/*/report.json`: exact replay metrics and checkpoint provenance
- `tokenizer_smoke/*_dense_roundtrip.gif`: four held-out fitted-target/tokenizer comparisons
- `tokenizer_smoke/heldout_eval_4096.json`: high-sample shared-tokenizer audit
- `tokenizer_3000/*_dense_roundtrip.gif`: final held-out comparisons after 3,000 steps
- `tokenizer_3000/train_shard0/*`: seen-source capacity/generalization diagnostic

Fitted checkpoints remain on the compute host under
`/home/m/jewels/corpus/ucf_prompt_smoke/fits_shard*` because each 120k target is substantially larger
than the lightweight provenance artifacts tracked here.
