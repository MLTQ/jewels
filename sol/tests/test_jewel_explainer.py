"""Contract and smoke tests for the narrated Jewel explainer series."""

from __future__ import annotations

from pathlib import Path
import unittest

from sol.jewel_explainer_episodes import EPISODES
from sol.jewel_explainer_scenes import SCENE_RENDERERS, draw_shot
from sol.jewel_explainer_style import HEIGHT, WIDTH, reveal, smooth
from sol.render_jewel_explainer import format_srt_time, subtitle_rows, validate_specs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class JewelExplainerTests(unittest.TestCase):
    def test_easing_is_bounded_monotonic_and_has_exact_endpoints(self) -> None:
        values = [smooth(index / 100) for index in range(101)]
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertTrue(all(first <= second for first, second in zip(values, values[1:])))
        self.assertEqual(reveal(-1.0, 0.2, 0.8), 0.0)
        self.assertEqual(reveal(2.0, 0.2, 0.8), 1.0)
        with self.assertRaisesRegex(ValueError, "positive width"):
            reveal(0.5, 1.0, 1.0)

    def test_series_spec_and_evidence_sources_are_complete(self) -> None:
        validate_specs(PROJECT_ROOT)
        self.assertEqual([episode.number for episode in EPISODES], list(range(1, 7)))
        self.assertEqual(sum(len(episode.shots) for episode in EPISODES), 42)
        self.assertEqual(
            {shot.visual for episode in EPISODES for shot in episode.shots},
            set(SCENE_RENDERERS),
        )

    def test_on_screen_copy_avoids_tofu_prone_modifier_glyphs(self) -> None:
        rendered_copy = repr(EPISODES)
        unsupported = set("ᵢₖₜᵀ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎⁴")
        self.assertFalse(unsupported.intersection(rendered_copy))

    def test_every_shot_draws_a_complete_rgb_frame(self) -> None:
        for episode in EPISODES:
            for shot in episode.shots:
                with self.subTest(episode=episode.number, shot=shot.title):
                    image = draw_shot(episode, shot, 0.57, 0.5, {})
                    self.assertEqual(image.size, (WIDTH, HEIGHT))
                    self.assertEqual(image.mode, "RGB")

    def test_subtitle_timing_is_ordered_and_respects_shot_durations(self) -> None:
        episode = EPISODES[0]
        durations = [10.0] * len(episode.shots)
        rows = subtitle_rows(episode.shots, durations, tail_seconds=0.7)
        self.assertGreater(len(rows), len(episode.shots))
        self.assertEqual(rows[0][0], 0.0)
        self.assertLessEqual(rows[-1][1], sum(durations))
        self.assertTrue(all(start < end for start, end, _ in rows))
        self.assertTrue(
            all(first[0] <= second[0] for first, second in zip(rows, rows[1:]))
        )

    def test_srt_time_rounding(self) -> None:
        self.assertEqual(format_srt_time(0), "00:00:00,000")
        self.assertEqual(format_srt_time(3661.2345), "01:01:01,234")


if __name__ == "__main__":
    unittest.main()
