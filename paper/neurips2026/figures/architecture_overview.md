# Architecture overview figure

## Intent

This vector figure distinguishes the system that has been demonstrated from the next research gate.
The solid path is a prompt-and-seed-only program that renders a video through hierarchical Jewel
tokens. The dashed amber box prevents the source-backed macro vocabulary from being mistaken for
an open-vocabulary generative prior.

## Contract

- The figure is included from `main.tex` inside a `\resizebox{\linewidth}{!}{...}` wrapper.
- Solid teal boxes mean implemented and experimentally tested.
- The macro dictionary must remain labeled as source-backed until reusable source-independent
  trajectory prototypes are actually demonstrated.
- Dashed amber means proposed next work, never a completed contribution.

## Maintenance

If the token hierarchy or ownership scheme changes, update this figure and the method section
together. Never remove the solid-versus-dashed legend.
