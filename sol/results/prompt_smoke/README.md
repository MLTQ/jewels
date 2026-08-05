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

Shard 0 (group 1, four training videos) was launched on the allocated RTX 2070 Super with the
validated 96-frame, 120k-jewel spatial-split settings and atomic recovery every 100 steps. Expected
runtime is roughly eight hours. The remaining three shards are held for the incoming multi-GPU
compute.

## Artifacts

- `manifest.json`: source-group ownership, prompts, encoder identity, and fit contract
- `prompts.pt`: validated normalized prompt vectors and per-example ownership
- `prompt_geometry.json`: unseen-template retrieval report

Fitted checkpoints remain on the compute host under
`/home/m/jewels/corpus/ucf_prompt_smoke/fits_shard*` because each 120k target is substantially larger
than the lightweight provenance artifacts tracked here.
