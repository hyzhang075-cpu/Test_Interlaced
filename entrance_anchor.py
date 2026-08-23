"""Translation-only bronchial entrance anchoring for reconstructed fibre curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AnchorConstraintResult:
    """One frame of anchor output plus diagnostics used by live and replay paths."""

    curve_m: np.ndarray
    applied: bool
    anchor_m: Optional[np.ndarray]
    source_point_m: Optional[np.ndarray]
    anchored_point_m: Optional[np.ndarray]
    translation_m: np.ndarray
    relative_insert_mm: float
    requested_distance_m: float
    used_distance_m: float
    clamped: bool


class EntranceAnchorConstraint:
    """Keep a material point on a reconstructed curve at one fixed entrance point."""

    def __init__(self, enabled: bool, entrance_offset_m: float = 0.07):
        entrance_offset_m = float(entrance_offset_m)
        if not np.isfinite(entrance_offset_m) or entrance_offset_m < 0.0:
            raise ValueError("entrance_offset_m must be a finite nonnegative value")

        self.enabled = bool(enabled)
        self.entrance_offset_m = entrance_offset_m
        self.anchor_m = None
        self.insertion_baseline_mm = None
        self.last_relative_insert_mm = 0.0

    @staticmethod
    def _validate_curve(curve_m):
        curve = np.asarray(curve_m, dtype=float)
        if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2:
            raise ValueError(f"curve_m must have shape (N, 3) with N >= 2, got {curve.shape}")
        if not np.all(np.isfinite(curve)):
            raise ValueError("curve_m must contain only finite values")
        return curve

    def _relative_insert_mm(self, insert_value_mm):
        if insert_value_mm is None:
            return self.last_relative_insert_mm

        values = np.asarray(insert_value_mm, dtype=float).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            return self.last_relative_insert_mm

        current_mm = float(values[0])
        if abs(current_mm) <= np.finfo(float).eps:
            return self.last_relative_insert_mm

        if self.insertion_baseline_mm is None:
            self.insertion_baseline_mm = current_mm
            self.last_relative_insert_mm = 0.0
        else:
            self.last_relative_insert_mm = current_mm - self.insertion_baseline_mm
        return self.last_relative_insert_mm

    @staticmethod
    def _point_from_tip(curve, requested_distance_m):
        segment_vectors = np.diff(curve, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        total_length = float(np.sum(segment_lengths))
        used_distance = float(np.clip(requested_distance_m, 0.0, total_length))
        remaining = used_distance

        if remaining <= np.finfo(float).eps:
            return curve[-1].copy(), used_distance, requested_distance_m > total_length

        for segment_index in range(len(segment_lengths) - 1, -1, -1):
            segment_length = float(segment_lengths[segment_index])
            if segment_length <= np.finfo(float).eps:
                continue
            if remaining <= segment_length:
                end = curve[segment_index + 1]
                start = curve[segment_index]
                fraction_from_end = remaining / segment_length
                point = end + fraction_from_end * (start - end)
                return point, used_distance, requested_distance_m > total_length
            remaining -= segment_length

        return curve[0].copy(), used_distance, requested_distance_m > total_length

    def apply(self, curve_m, insert_value_mm=None):
        """Translate one curve so its insertion-dependent material point hits the entrance."""
        curve = self._validate_curve(curve_m)
        if not self.enabled:
            return AnchorConstraintResult(
                curve_m=curve.copy(),
                applied=False,
                anchor_m=None,
                source_point_m=None,
                anchored_point_m=None,
                translation_m=np.zeros(3, dtype=float),
                relative_insert_mm=0.0,
                requested_distance_m=0.0,
                used_distance_m=0.0,
                clamped=False,
            )

        if self.anchor_m is None:
            self.anchor_m = curve[-1].copy() - np.array([self.entrance_offset_m, 0.0, 0.0])

        relative_insert_mm = self._relative_insert_mm(insert_value_mm)
        requested_distance_m = max(
            0.0,
            self.entrance_offset_m + relative_insert_mm / 1000.0,
        )
        source_point_m, used_distance_m, clamped = self._point_from_tip(
            curve,
            requested_distance_m,
        )
        translation_m = self.anchor_m - source_point_m
        anchored_curve = curve + translation_m

        return AnchorConstraintResult(
            curve_m=anchored_curve,
            applied=True,
            anchor_m=self.anchor_m.copy(),
            source_point_m=source_point_m,
            anchored_point_m=source_point_m + translation_m,
            translation_m=translation_m,
            relative_insert_mm=relative_insert_mm,
            requested_distance_m=requested_distance_m,
            used_distance_m=used_distance_m,
            clamped=clamped,
        )


def apply_entrance_anchor(constraint, curve_m, insert_value_mm=None):
    """Apply the shared live-pipeline stage and return only the output curve."""
    if not isinstance(constraint, EntranceAnchorConstraint):
        raise TypeError("constraint must be an EntranceAnchorConstraint")
    return constraint.apply(curve_m, insert_value_mm).curve_m
