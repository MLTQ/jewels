# `package.json`

## Purpose

Pins the small standalone viewer's Three.js and Vite runtime. The viewer is isolated from the
research Python environment and builds to ordinary static files. Vite is pinned past the
path-traversal and arbitrary-file-read advisories affecting early 7.x releases.

## Components

### `dev`

- **Does**: Starts the viewer on the local loopback interface with live reload.

### `build`

- **Does**: Produces the deployable static bundle and checks module resolution.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Viewer source modules | Three.js 0.179 APIs and addon import paths | Major dependency changes |
| Local operator | `npm run dev` starts a loopback server | Script renaming |
