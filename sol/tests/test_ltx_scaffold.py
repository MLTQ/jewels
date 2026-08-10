import tempfile
import unittest
from pathlib import Path

from sol.ltx_scaffold import (
    ScaffoldConfig,
    _parse_gpu_sample,
    build_command,
    validate_config,
)


class LTXScaffoldTests(unittest.TestCase):
    def config(self, root: Path, **updates: object) -> ScaffoldConfig:
        values: dict[str, object] = {
            "ltx_root": root,
            "distilled_checkpoint": root / "distilled.safetensors",
            "spatial_upsampler": root / "upsampler.safetensors",
            "gemma_root": root / "gemma",
            "prompt": "A rider crosses a bright field.",
            "output": root / "output.mp4",
            "cuda_visible_device": "GPU-test",
        }
        values.update(updates)
        return ScaffoldConfig(**values)  # type: ignore[arg-type]

    def test_command_keeps_prompt_as_one_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory), prompt='A rider says "go"; then turns.')
            command = build_command(config)
        prompt_index = command.index("--prompt") + 1
        self.assertEqual(command[prompt_index], 'A rider says "go"; then turns.')
        self.assertIn("--quantization", command)
        self.assertIn("fp8-cast", command)
        self.assertIn("--offload", command)
        self.assertIn("cpu", command)

    def test_geometry_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validate_config(self.config(root), require_assets=False)
            with self.assertRaisesRegex(ValueError, "multiple of 64"):
                validate_config(self.config(root, width=750), require_assets=False)
            with self.assertRaisesRegex(ValueError, "8\\*K \\+ 1"):
                validate_config(self.config(root, num_frames=48), require_assets=False)

    def test_required_assets_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "missing LTX runtime assets"):
                validate_config(self.config(root))

    def test_gpu_sample_parser_rejects_malformed_rows(self) -> None:
        self.assertEqual(_parse_gpu_sample("1234, 87\n"), (1234, 87))
        with self.assertRaisesRegex(ValueError, "unexpected nvidia-smi output"):
            _parse_gpu_sample("1234 MiB")


if __name__ == "__main__":
    unittest.main()
