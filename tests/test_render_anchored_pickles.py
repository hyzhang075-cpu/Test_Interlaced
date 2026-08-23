import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.render_anchored_pickles import (
    load_anchored_pickle,
    output_paths_for,
    render_tip_trajectory_png,
)


def _straight_shape():
    shape = np.zeros((60, 3), dtype=float)
    shape[:, 0] = np.linspace(0.0, 0.1, 60)
    return shape


def _write_frames(path, inserts):
    with Path(path).open("wb") as stream:
        for index, insert in enumerate(inserts):
            pickle.dump(
                {
                    "timestamp": 100.0 + 0.1 * index,
                    "r_results": _straight_shape(),
                    "insert_value": insert,
                },
                stream,
            )


class RenderAnchoredPicklesTests(unittest.TestCase):
    def test_loader_anchors_shapes_and_ignores_leading_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sample.pkl"
            _write_frames(source, [0.0, 156.8, 166.8])

            data = load_anchored_pickle(source)

        self.assertEqual(data.shapes_m.shape, (3, 60, 3))
        np.testing.assert_allclose(data.relative_inserts_mm, [0.0, 0.0, 10.0])
        np.testing.assert_allclose(data.shapes_m[2, :, 0] - data.shapes_m[1, :, 0], 0.01)
        np.testing.assert_allclose(data.anchor_m, [0.03, 0.0, 0.0])
        self.assertAlmostEqual(data.insertion_baseline_mm, 156.8)
        self.assertFalse(np.any(data.clamped))

    def test_loader_rejects_non_60_point_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "bad.pkl"
            with source.open("wb") as stream:
                pickle.dump(
                    {
                        "timestamp": 1.0,
                        "r_results": np.zeros((59, 3)),
                        "insert_value": 1.0,
                    },
                    stream,
                )

            with self.assertRaisesRegex(ValueError, r"\(60, 3\)"):
                load_anchored_pickle(source)

    def test_output_paths_are_distinct_from_legacy_outputs(self):
        source = Path("VTK_ShapeData_20260820_192216.pkl")

        png, video = output_paths_for(Path("tips"), Path("videos"), source)

        self.assertEqual(png, Path("tips/VTK_ShapeData_20260820_192216_anchored_tip_trajectory.png"))
        self.assertEqual(
            video,
            Path("videos/VTK_ShapeData_20260820_192216/VTK_ShapeData_20260820_192216_anchored_full.mp4"),
        )

    def test_png_renderer_draws_real_anchored_trajectory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sample.pkl"
            output = Path(temp_dir) / "trajectory.png"
            _write_frames(source, [156.8, 166.8, 176.8])
            data = load_anchored_pickle(source)

            render_tip_trajectory_png(data, output, title="Anchored test")

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
