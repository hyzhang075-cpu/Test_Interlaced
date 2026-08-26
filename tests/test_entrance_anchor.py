import unittest

import numpy as np

from entrance_anchor import EntranceAnchorConstraint


class EntranceAnchorConstraintTests(unittest.TestCase):
    def test_default_entrance_offset_is_6cm(self):
        constraint = EntranceAnchorConstraint(enabled=True)

        self.assertAlmostEqual(constraint.entrance_offset_m, 0.06)

    def test_first_nonzero_insert_is_baseline_and_next_10mm_advances_target(self):
        constraint = EntranceAnchorConstraint(enabled=True, entrance_offset_m=0.07)
        curve = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

        placeholder = constraint.apply(curve, 0.0)
        baseline = constraint.apply(curve, 156.8)
        advanced = constraint.apply(curve, 166.8)

        np.testing.assert_allclose(placeholder.curve_m, curve)
        np.testing.assert_allclose(baseline.curve_m, curve)
        np.testing.assert_allclose(advanced.curve_m[:, 0], [0.01, 0.11])
        np.testing.assert_allclose(advanced.anchored_point_m, [0.03, 0.0, 0.0])
        self.assertAlmostEqual(advanced.relative_insert_mm, 10.0)
        self.assertAlmostEqual(constraint.insertion_baseline_mm, 156.8)

    def test_uses_true_reverse_polyline_arclength_and_interpolates_segment(self):
        constraint = EntranceAnchorConstraint(enabled=True, entrance_offset_m=0.07)
        curve = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.05, 0.05, 0.0],
            ]
        )

        result = constraint.apply(curve, None)

        np.testing.assert_allclose(result.source_point_m, [0.03, 0.0, 0.0])
        np.testing.assert_allclose(result.anchor_m, [-0.02, 0.05, 0.0])
        np.testing.assert_allclose(result.anchored_point_m, result.anchor_m)
        np.testing.assert_allclose(result.translation_m, [-0.05, 0.05, 0.0])

    def test_constraint_preserves_shape_point_count_and_pairwise_vectors(self):
        curve = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.03, 0.04, 0.0],
                [0.08, 0.04, 0.02],
            ]
        )

        result = EntranceAnchorConstraint(enabled=True).apply(curve, None)

        self.assertEqual(result.curve_m.shape, curve.shape)
        np.testing.assert_allclose(np.diff(result.curve_m, axis=0), np.diff(curve, axis=0))

    def test_zero_after_baseline_holds_last_valid_relative_insertion(self):
        constraint = EntranceAnchorConstraint(enabled=True)
        curve = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        constraint.apply(curve, 156.8)
        advanced = constraint.apply(curve, 176.8)

        dropout = constraint.apply(curve, 0.0)

        self.assertAlmostEqual(dropout.relative_insert_mm, 20.0)
        np.testing.assert_allclose(dropout.curve_m, advanced.curve_m)

    def test_disabled_constraint_returns_an_independent_unchanged_curve(self):
        curve = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

        result = EntranceAnchorConstraint(enabled=False).apply(curve, 166.8)

        np.testing.assert_array_equal(result.curve_m, curve)
        self.assertIsNot(result.curve_m, curve)
        self.assertIsNone(result.anchor_m)
        self.assertFalse(result.applied)

    def test_distance_beyond_curve_clamps_to_start_without_changing_point_count(self):
        constraint = EntranceAnchorConstraint(enabled=True)
        curve = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])

        result = constraint.apply(curve, None)

        self.assertTrue(result.clamped)
        self.assertAlmostEqual(result.used_distance_m, 0.05)
        self.assertEqual(result.curve_m.shape, (2, 3))
        np.testing.assert_allclose(result.source_point_m, curve[0])

    def test_rejects_nonfinite_or_wrong_shape_curves(self):
        constraint = EntranceAnchorConstraint(enabled=True)

        with self.assertRaisesRegex(ValueError, "shape"):
            constraint.apply(np.zeros((3, 2)), None)
        with self.assertRaisesRegex(ValueError, "finite"):
            constraint.apply(np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]), None)

    def test_configure_new_distance_resets_anchor_and_insertion_zero(self):
        constraint = EntranceAnchorConstraint(enabled=True, entrance_offset_m=0.07)
        old_curve = np.array([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0]])
        constraint.apply(old_curve, 156.8)
        constraint.apply(old_curve, 166.8)

        constraint.configure(enabled=True, entrance_offset_m=0.08)

        self.assertTrue(constraint.enabled)
        self.assertAlmostEqual(constraint.entrance_offset_m, 0.08)
        self.assertIsNone(constraint.anchor_m)
        self.assertIsNone(constraint.insertion_baseline_mm)
        self.assertEqual(constraint.last_relative_insert_mm, 0.0)

        new_curve = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0]])
        placeholder = constraint.apply(new_curve, 0.0)
        baseline = constraint.apply(new_curve, 200.0)

        np.testing.assert_allclose(placeholder.anchor_m, [0.22, 0.0, 0.0])
        self.assertAlmostEqual(baseline.relative_insert_mm, 0.0)
        self.assertAlmostEqual(constraint.insertion_baseline_mm, 200.0)

    def test_invalid_configure_is_atomic(self):
        constraint = EntranceAnchorConstraint(enabled=True, entrance_offset_m=0.07)
        curve = np.array([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0]])
        constraint.apply(curve, 156.8)
        original_anchor = constraint.anchor_m.copy()

        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            constraint.configure(enabled=False, entrance_offset_m=np.nan)

        self.assertTrue(constraint.enabled)
        self.assertAlmostEqual(constraint.entrance_offset_m, 0.07)
        np.testing.assert_allclose(constraint.anchor_m, original_anchor)
        self.assertAlmostEqual(constraint.insertion_baseline_mm, 156.8)


if __name__ == "__main__":
    unittest.main()
