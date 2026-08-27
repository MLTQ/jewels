"""Contract and smoke tests for the narrated Jewel explainer series."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest

from PIL import Image, ImageColor

from sol.jewel_explainer_episodes import EPISODES
from sol.jewel_explainer_scenes import (
    CONTENT_TOP,
    SCENE_RENDERERS,
    draw_shot,
    time_slice_dot_alpha,
    time_slice_polygon,
)
from sol.jewel_explainer_style import (
    FOREGROUND,
    HEIGHT,
    LIGHT_THEME,
    WIDTH,
    JewelCanvas,
    reveal,
    smooth,
)
from sol.render_jewel_explainer import (
    format_srt_time,
    focus_evidence_asset,
    merge_episode_records,
    narration_duration_bounds,
    qwen_token_ceiling,
    subtitle_rows,
    validate_specs,
)


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

    def test_only_jewel_geometry_episode_uses_eggshell_theme(self) -> None:
        self.assertEqual(EPISODES[1].theme, "light")
        self.assertTrue(
            all(episode.theme == "dark" for episode in EPISODES if episode.number != 2)
        )
        canvas = JewelCanvas(theme=EPISODES[1].theme)
        self.assertEqual(canvas.theme.background, LIGHT_THEME.background)
        self.assertEqual(canvas.color(FOREGROUND), LIGHT_THEME.foreground)

    def test_video_slice_advances_only_along_drawn_time_axis(self) -> None:
        front = time_slice_polygon(0.0)
        middle = time_slice_polygon(0.5)
        back = time_slice_polygon(1.0)
        for start, halfway, finish in zip(front, middle, back):
            self.assertEqual(
                halfway,
                ((start[0] + finish[0]) / 2, (start[1] + finish[1]) / 2),
            )
            self.assertEqual(
                (finish[0] - start[0], finish[1] - start[1]),
                (135, -85),
            )

    def test_video_slice_reveals_dots_only_after_crossing_their_time(self) -> None:
        point_times = (0.2, 0.5, 0.8)
        self.assertEqual(
            [time_slice_dot_alpha(point_time, 0.0) for point_time in point_times],
            [0.0, 0.0, 0.0],
        )
        halfway = [time_slice_dot_alpha(point_time, 0.5) for point_time in point_times]
        self.assertEqual(halfway[0], 1.0)
        self.assertEqual(halfway[1:], [0.0, 0.0])
        self.assertEqual(
            [time_slice_dot_alpha(point_time, 1.0) for point_time in point_times],
            [1.0, 1.0, 1.0],
        )
        sweep = [time_slice_dot_alpha(0.5, step / 20) for step in range(21)]
        self.assertTrue(all(first <= second for first, second in zip(sweep, sweep[1:])))
        with self.assertRaisesRegex(ValueError, "feather"):
            time_slice_dot_alpha(0.2, 0.5, feather=0.0)

    def test_video_slice_stays_behind_the_black_front_frame(self) -> None:
        image = draw_shot(EPISODES[1], EPISODES[1].shots[0], 0.3, 0.1, {})
        self.assertEqual(
            image.getpixel((400, 260)),
            ImageColor.getrgb(LIGHT_THEME.foreground),
        )

    def test_on_screen_copy_avoids_tofu_prone_modifier_glyphs(self) -> None:
        rendered_copy = repr(EPISODES)
        unsupported = set("ᵢₖₜᵀ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎⁴")
        self.assertFalse(unsupported.intersection(rendered_copy))

    def test_public_scripts_are_concise_and_avoid_internal_shorthand(self) -> None:
        public_copy = " ".join(
            f"{shot.title} {shot.narration} {shot.caption}"
            for episode in EPISODES
            for shot in episode.shots
        )
        for shorthand in ("JRGB", "NLL", "KNN", "arg top"):
            self.assertNotIn(shorthand, public_copy)
        for episode in EPISODES:
            for shot in episode.shots:
                with self.subTest(episode=episode.number, shot=shot.title):
                    self.assertLessEqual(len(shot.narration.split()), 80)
                    self.assertLessEqual(len(shot.caption.split()), 18)

    def test_every_shot_draws_a_complete_rgb_frame(self) -> None:
        for episode in EPISODES:
            for shot in episode.shots:
                with self.subTest(episode=episode.number, shot=shot.title):
                    image = draw_shot(episode, shot, 0.57, 0.5, {})
                    self.assertEqual(image.size, (WIDTH, HEIGHT))
                    self.assertEqual(image.mode, "RGB")

    def test_diagrams_cannot_overpaint_the_header_safe_zone(self) -> None:
        episode = EPISODES[1]
        shot = episode.shots[0]
        expected = JewelCanvas(theme=episode.theme)
        expected.header(episode.number, episode.title, shot.title)
        rendered = draw_shot(episode, shot, 1.0, 0.5, {})
        safe_zone = (0, 0, WIDTH, CONTENT_TOP)
        self.assertEqual(
            rendered.crop(safe_zone).tobytes(),
            expected.image.crop(safe_zone).tobytes(),
        )

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

    def test_partial_render_replaces_only_selected_inventory_records(self) -> None:
        previous = [
            {"episode": 1, "version": "keep"},
            {"episode": 2, "version": "old"},
        ]
        rendered = [{"episode": 2, "version": "new"}]
        self.assertEqual(
            merge_episode_records(previous, rendered),
            [
                {"episode": 1, "version": "keep"},
                {"episode": 2, "version": "new"},
            ],
        )

    def test_tall_evidence_sheets_are_reflowed_for_video(self) -> None:
        coherent = focus_evidence_asset(
            "coherent", Image.new("RGB", (330, 1482), "white")
        )
        proof = focus_evidence_asset(
            "proof-sheet", Image.new("RGB", (614, 1102), "white")
        )
        unchanged = focus_evidence_asset(
            "evidence", Image.new("RGB", (2340, 1440), "white")
        )
        self.assertEqual(coherent.size, (990, 198))
        self.assertEqual(proof.size, (1332, 170))
        self.assertEqual(unchanged.size, (2340, 1440))

    def test_qwen_duration_guard_caps_runaway_audio_tokens(self) -> None:
        text = " ".join(["Jewel"] * 77)
        expected, minimum, maximum = narration_duration_bounds(text)
        self.assertAlmostEqual(expected, 77 * 60 / 145)
        self.assertLess(minimum, expected)
        self.assertLess(expected, maximum)
        self.assertLess(maximum, 55)
        self.assertEqual(qwen_token_ceiling(text, 900), math.ceil(maximum * 12.5))
        self.assertLess(qwen_token_ceiling(text, 900), 700)

    def test_actual_jewel_asset_is_traceable_to_four_checkpoint_rows(self) -> None:
        assets = PROJECT_ROOT / "sol" / "results" / "jewel_explainer_series_v1" / "assets"
        metadata = json.loads((assets / "actual_jewel_isolation.json").read_text())
        checkpoint = assets / metadata["checkpoint"]
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.assertEqual(metadata["schema"], "jewel-isolation-explainer-v1")
        self.assertEqual(metadata["field_jewels"], 6471)
        self.assertEqual(metadata["field_shape"], [64, 160, 306])
        self.assertEqual(metadata["checkpoint_sha256"], digest)
        self.assertEqual(metadata["output_frames"], 108)
        selected = metadata["selected_jewels"]
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({row["field_index"] for row in selected}), 4)
        self.assertTrue(
            all(
                len(row["covariance"]) == 3
                and all(len(values) == 3 for values in row["covariance"])
                for row in selected
            )
        )


if __name__ == "__main__":
    unittest.main()
