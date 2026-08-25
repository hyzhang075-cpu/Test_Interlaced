import argparse
import io
import threading
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace

import numpy as np
import vtk

import demonstrator as demonstrator_module
from entrance_anchor import EntranceAnchorConstraint, apply_entrance_anchor
from demonstrator import Demonstrator, _build_anchor_marker_segments
from main import add_anchor_arguments


class DemonstratorAnchorPipelineTests(unittest.TestCase):
    def test_software_shape_removes_positive_x_insertion_only_when_anchor_enabled(self):
        """Anchored JSON output must undo the insertion that receiving software adds."""
        shape_mm = np.array([[200.0, 10.0, -5.0], [250.0, 20.0, -4.0]])

        adjusted = demonstrator_module._shape_for_software_send(
            shape_mm,
            anchor_enabled=True,
            insert_value_mm=156.8,
        )

        np.testing.assert_allclose(adjusted, [[43.2, 10.0, -5.0], [93.2, 20.0, -4.0]])
        np.testing.assert_array_equal(shape_mm, [[200.0, 10.0, -5.0], [250.0, 20.0, -4.0]])

    def test_software_shape_is_unchanged_when_anchor_disabled(self):
        """Existing non-anchored JSON coordinates must remain backward compatible."""
        shape_mm = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

        adjusted = demonstrator_module._shape_for_software_send(
            shape_mm,
            anchor_enabled=False,
            insert_value_mm=12.5,
        )

        np.testing.assert_array_equal(adjusted, shape_mm)

    def test_software_shape_ignores_missing_or_invalid_insertion(self):
        """UDP startup and malformed samples must not corrupt outgoing coordinates."""
        shape_mm = np.array([[10.0, 20.0, 30.0]])

        for invalid in (None, np.nan, [1.0, 2.0]):
            with self.subTest(invalid=invalid):
                adjusted = demonstrator_module._shape_for_software_send(
                    shape_mm,
                    anchor_enabled=True,
                    insert_value_mm=invalid,
                )
                np.testing.assert_array_equal(adjusted, shape_mm)

    def test_anchor_marker_segments_form_four_scaled_3d_diagonals(self):
        segments = _build_anchor_marker_segments(
            anchor_m=np.array([1.0, 2.0, 3.0]),
            scale_factor=10.0,
            half_size=2.0,
        )

        np.testing.assert_array_equal(
            segments,
            np.array(
                [
                    [[8.0, 18.0, 28.0], [12.0, 22.0, 32.0]],
                    [[8.0, 18.0, 32.0], [12.0, 22.0, 28.0]],
                    [[8.0, 22.0, 28.0], [12.0, 18.0, 32.0]],
                    [[8.0, 22.0, 32.0], [12.0, 18.0, 28.0]],
                ]
            ),
        )

    def test_vtk_anchor_marker_tracks_constraint_visibility_and_position(self):
        constraint = EntranceAnchorConstraint(enabled=False)
        harness = SimpleNamespace(
            renderer=vtk.vtkRenderer(),
            entrance_anchor_constraint=constraint,
            entrance_anchor_lock=threading.RLock(),
        )
        Demonstrator.add_anchor_marker(harness)

        self.assertEqual(harness.anchor_marker_actor.GetVisibility(), 0)
        np.testing.assert_allclose(harness.anchor_marker_actor.GetProperty().GetColor(), [1.0, 1.0, 1.0])
        self.assertAlmostEqual(harness.anchor_marker_actor.GetProperty().GetLineWidth(), 3.0)
        self.assertEqual(harness.anchor_marker_points.GetNumberOfPoints(), 8)
        self.assertEqual(harness.anchor_marker_lines.GetNumberOfCells(), 4)

        curve = np.array([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0]])
        Demonstrator._update_anchor_marker(harness, scale_factor=100.0)
        self.assertEqual(harness.anchor_marker_actor.GetVisibility(), 0)

        constraint.configure(enabled=True, entrance_offset_m=0.07)
        constraint.apply(curve, 156.8)
        Demonstrator._update_anchor_marker(harness, scale_factor=100.0)

        self.assertEqual(harness.anchor_marker_actor.GetVisibility(), 1)
        rendered_points = np.array(
            [harness.anchor_marker_points.GetPoint(index) for index in range(8)]
        )
        np.testing.assert_allclose(np.mean(rendered_points, axis=0), [13.0, 0.0, 0.0])

        constraint.configure(enabled=False)
        Demonstrator._update_anchor_marker(harness, scale_factor=100.0)
        self.assertEqual(harness.anchor_marker_actor.GetVisibility(), 0)

    def test_stage_is_noop_when_disabled_and_translates_when_enabled(self):
        curve = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

        disabled = apply_entrance_anchor(
            EntranceAnchorConstraint(enabled=False),
            curve,
            166.8,
        )
        enabled_state = EntranceAnchorConstraint(enabled=True)
        apply_entrance_anchor(enabled_state, curve, 156.8)
        enabled = apply_entrance_anchor(enabled_state, curve, 166.8)

        np.testing.assert_allclose(disabled, curve)
        np.testing.assert_allclose(enabled[:, 0], [0.01, 0.11])
        np.testing.assert_allclose(enabled[:, 1:], 0.0)

    def test_main_anchor_arguments_default_to_7cm_and_accept_override(self):
        parser = argparse.ArgumentParser()
        add_anchor_arguments(parser)

        defaults = parser.parse_args([])
        overridden = parser.parse_args(["--anchor_offset_m", "0.085"])

        self.assertFalse(defaults.anchor_constraint_enabled)
        self.assertAlmostEqual(defaults.anchor_offset_m, 0.07)
        self.assertAlmostEqual(overridden.anchor_offset_m, 0.085)

    def test_main_anchor_arguments_reject_invalid_offset_before_runtime_setup(self):
        parser = argparse.ArgumentParser()
        add_anchor_arguments(parser)

        for invalid in ("-0.01", "nan", "inf"):
            with self.subTest(invalid=invalid), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--anchor_offset_m", invalid])

    def test_process_interface_reconfigures_and_applies_under_shared_lock(self):
        harness = SimpleNamespace(
            entrance_anchor_constraint=EntranceAnchorConstraint(enabled=False),
            entrance_anchor_lock=threading.RLock(),
        )
        curve = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0]])

        Demonstrator.configure_anchor_constraint(
            harness,
            enabled=True,
            entrance_offset_m=0.08,
        )
        first = Demonstrator._apply_anchor_constraint(harness, curve, 0.0)
        Demonstrator._apply_anchor_constraint(harness, curve, 200.0)
        advanced = Demonstrator._apply_anchor_constraint(harness, curve, 210.0)

        np.testing.assert_allclose(first, curve)
        np.testing.assert_allclose(advanced[:, 0], [0.01, 0.31])
        np.testing.assert_allclose(harness.entrance_anchor_constraint.anchor_m, [0.22, 0.0, 0.0])
        self.assertAlmostEqual(harness.entrance_anchor_constraint.insertion_baseline_mm, 200.0)

    def test_configuration_waits_for_inflight_frame_application(self):
        apply_entered = threading.Event()
        release_apply = threading.Event()
        configure_finished = threading.Event()

        class BlockingConstraint(EntranceAnchorConstraint):
            def apply(self, curve_m, insert_value_mm=None):
                apply_entered.set()
                self.assert_released = release_apply.wait(timeout=2.0)
                return super().apply(curve_m, insert_value_mm)

        constraint = BlockingConstraint(enabled=True)
        harness = SimpleNamespace(
            entrance_anchor_constraint=constraint,
            entrance_anchor_lock=threading.RLock(),
        )
        curve = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0]])

        apply_thread = threading.Thread(
            target=Demonstrator._apply_anchor_constraint,
            args=(harness, curve, 156.8),
        )
        configure_thread = threading.Thread(
            target=lambda: (
                Demonstrator.configure_anchor_constraint(
                    harness,
                    enabled=True,
                    entrance_offset_m=0.08,
                ),
                configure_finished.set(),
            )
        )
        apply_thread.start()
        self.assertTrue(apply_entered.wait(timeout=1.0))
        configure_thread.start()
        self.assertFalse(configure_finished.wait(timeout=0.05))

        release_apply.set()
        apply_thread.join(timeout=2.0)
        configure_thread.join(timeout=2.0)

        self.assertFalse(apply_thread.is_alive())
        self.assertFalse(configure_thread.is_alive())
        self.assertTrue(constraint.assert_released)
        self.assertTrue(configure_finished.is_set())
        self.assertAlmostEqual(constraint.entrance_offset_m, 0.08)
        self.assertIsNone(constraint.anchor_m)


if __name__ == "__main__":
    unittest.main()
