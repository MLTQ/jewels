# Deterministic washout audit

This folder isolates the washed-out direct-continuation result without retraining. All panels use
the held-out group-4 targets, exact carried jewels, the same prompt and renderer, and the same future
frames. The audit supplies target birth cells/counts/ranks to the deterministic model and then swaps
one predicted 22-D mark group at a time.

Across Basketball, HorseRiding, PlayingGuitar, and ApplyEyeMakeup, exact topology improves mean PSNR
from 14.060 to only 14.232 dB. Predicted opacity alone retains 24.816 dB, while predicted geometry
alone falls to 14.434 dB and predicted color/gradient alone to 15.255 dB. Only 39.73% of predicted
centers remain in their supplied spatial cell and 25.41% in their full space-time birth cell.

The conclusion is negative and useful: a better sparse occupancy/count loss is required eventually,
but it cannot fix the current washout. Deterministic per-mark regression averages incompatible
geometries and colors even when topology is exact.

- `washout_report.json` contains all per-class metrics and topology checks.
- `*_washout_decomposition.gif` animates carried-only, oracle-mark, mark-group, and free controls.
- `*_contact.png` shows three representative times from each animation.
