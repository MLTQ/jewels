import json
import tempfile
import unittest
from pathlib import Path

from sol.generate_ltx_corpus import (
    CorpusRuntime,
    build_plan,
    scaffold_config,
    select_plan,
    write_corpus_manifest,
)


def source_manifest() -> dict[str, object]:
    classes = []
    for class_name in ("Basketball", "HorseRiding", "Guitar", "Makeup"):
        classes.append(
            {
                "class_name": class_name,
                "train_prompts": [
                    f"{class_name} train zero",
                    f"{class_name} train one",
                    f"{class_name} train two",
                ],
                "evaluation_prompts": [f"{class_name} evaluation"],
            }
        )
    return {"schema": "ucf-prompt-streaming-v1", "classes": classes}


class LTXCorpusTests(unittest.TestCase):
    def test_plan_balances_roles_and_deterministic_seeds(self) -> None:
        plan = build_plan(source_manifest(), seed_base=1_000, prompt_suffix="Stable.")
        self.assertEqual(len(plan), 16)
        self.assertEqual(
            [item.prompt_role for item in plan[:4]],
            ["train", "train", "train", "evaluation"],
        )
        self.assertEqual([item.seed for item in plan[:4]], [1000, 1001, 1002, 1003])
        self.assertEqual(plan[4].seed, 1100)
        self.assertEqual(len({item.stem for item in plan}), 16)
        self.assertEqual(plan[0].source_prompt, "Basketball train zero")
        self.assertEqual(
            plan[0].generation_prompt, "Basketball train zero. Stable."
        )

    def test_manifest_recognizes_only_matching_completed_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(source_manifest()))
            plan = build_plan(source_manifest())[:1]
            runtime = CorpusRuntime(
                ltx_root=root / "ltx",
                output_dir=root / "corpus",
                cuda_visible_device="GPU-test",
            )
            config = scaffold_config(plan[0], runtime)
            config.output.parent.mkdir(parents=True)
            config.output.write_bytes(b"video")
            receipt = {
                "status": "complete",
                "config": {
                    "prompt": config.prompt,
                    "output": str(config.output),
                    "seed": config.seed,
                    "height": config.height,
                    "width": config.width,
                    "num_frames": config.num_frames,
                    "frame_rate": config.frame_rate,
                    "quantization": config.quantization,
                    "offload": config.offload,
                    "max_batch_size": config.max_batch_size,
                },
                "elapsed_seconds": 12.5,
                "ltx_revision": "abc",
                "gpu_monitor": {"peak_memory_mib": 9000},
                "video_probe": {"format": {"duration": "2.0"}},
            }
            config.output.with_suffix(".json").write_text(json.dumps(receipt))
            payload = write_corpus_manifest(source_path, plan, runtime)
            self.assertEqual(payload["summary"]["complete"], 1)
            self.assertEqual(payload["summary"]["pending"], 0)
            self.assertEqual(payload["examples"][0]["status"], "complete")
            self.assertEqual(
                payload["examples"][0]["result"]["peak_memory_mib"], 9000
            )

            receipt["config"]["seed"] = 999
            config.output.with_suffix(".json").write_text(json.dumps(receipt))
            payload = write_corpus_manifest(source_path, plan, runtime)
            self.assertEqual(payload["summary"]["complete"], 0)
            self.assertEqual(payload["summary"]["failed"], 1)

    def test_plan_can_select_one_balanced_prompt_role(self) -> None:
        plan = build_plan(source_manifest(), seed_base=1_000)
        evaluation = select_plan(plan, prompt_role="evaluation")
        self.assertEqual(len(evaluation), 4)
        self.assertEqual(
            [item.class_name for item in evaluation],
            ["Basketball", "HorseRiding", "Guitar", "Makeup"],
        )
        self.assertEqual([item.seed for item in evaluation], [1003, 1103, 1203, 1303])
        self.assertIs(select_plan(plan), plan)
        with self.assertRaisesRegex(ValueError, "unsupported prompt role"):
            select_plan(plan, prompt_role="validation")

    def test_invalid_source_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            build_plan({"schema": "wrong", "classes": []})


if __name__ == "__main__":
    unittest.main()
