# Scene-posterior oracle Gate 1i protocol

## Decision question

Does the r6 speaker still average when it receives its training-only source posterior, or is the
text-conditioned Gaussian scene prior solely responsible for the missing object structure?

This diagnostic is frozen after Gate 1h and before inspecting any posterior-driven generation.

## Frozen audit

For the lexicographically first r6 training video under each exact style/action prompt, compare:

1. the learned training-source posterior mean (oracle leakage);
2. the correct text scene-prior mean;
3. the prompt-blind scene-prior mean.

Keep the r6 decoder checkpoint, frozen codebook/text model, 72,000-Jewel generation, sampling seed,
continuous centroids, token temperature/top-k, and support renderer unchanged. Evaluate target-owned
token NLL/density, free-run cell/token histogram match, diagnostic voxel PSNR, and qualitative
early/middle/late frames.

## Frozen decision

Call the text scene prior the next bottleneck only if the posterior oracle improves token NLL by at
least 2% and free-run histogram cosine by at least 0.02 over the correct text prior. Otherwise the
conditionally independent local Jewel decoder remains the bottleneck and the next speaker must add
hierarchical/local block tokens. The posterior arm is never valid prompt inference and cannot count
as proof of text-to-video generation.
