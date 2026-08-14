"""Rebind an existing prompt cache to a manifest with identical prompt text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sol.prompt_embeddings import (
    build_prompt_cache,
    load_prompt_cache,
    manifest_digest,
    save_prompt_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    parent_manifest = json.loads(Path(args.parent_manifest).read_text())
    parent = load_prompt_cache(args.parent_cache)
    if parent.manifest_sha256 != manifest_digest(parent_manifest):
        raise ValueError("parent cache does not own the parent manifest")
    manifest = json.loads(Path(args.manifest).read_text())
    cache = build_prompt_cache(manifest, parent.prompts, parent.embeddings)
    save_prompt_cache(cache, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "prompts": len(cache.prompts),
                "parent_manifest_sha256": parent.manifest_sha256,
                "manifest_sha256": cache.manifest_sha256,
            }
        )
    )


if __name__ == "__main__":
    main()
