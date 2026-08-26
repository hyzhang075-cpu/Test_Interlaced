"""Position-and-frame bronchial entrance anchoring for reconstructed fibre curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class AnchorConstraintResult:
    """One frame of anchor output plus diagnostics used by live and replay paths."""

    curve_m: np.ndarray
    applied: bool
    anchor_m: Optional[np.ndarray]
    source_point_m: Optional[np.ndarray]
    anchored_point_m: Optional[np.ndarray]
    translation_m: np.ndarray
    rotation_matrix: np.ndarray
    anchor_frame: Optional[np.ndarray]
    source_frame: Optional[np.ndarray]
    relative_insert_mm: float
    requested_distance_m: float
    used_distance_m: float
    clamped: bool


class EntranceAnchorConstraint:
    """Keep a material point on a reconstructed curve at one fixed entrance point."""

    def __init__(self, enabled: bool, entrance_offset_m: float = 0.06):
        entrance_offset_m = float(entrance_offset_m)
        if not np.isfinite(entrance_offset_m) or entrance_offset_m < 0.0:
            raise ValueError("entrance_offset_m must be a finite nonnegative value")

        self.enabled = bool(enabled)
        self.entrance_offset_m = entrance_offset_m
        self.anchor_m = None
        self.anchor_frame = None
        self.insertion_baseline_mm = None
        self.last_relative_insert_mm = 0.0

    def configure(self, enabled=None, entrance_offset_m=None, reset=True):
        """Atomically update runtime settings and optionally start a fresh anchor session."""
        new_enabled = self.enabled if enabled is None else bool(enabled)
        new_offset_m = self.entrance_offset_m
        if entrance_offset_m is not None:
            new_offset_m = float(entrance_offset_m)
            if not np.isfinite(new_offset_m) or new_offset_m < 0.0:
                raise ValueError("entrance_offset_m must be a finite nonnegative value")

        self.enabled = new_enabled
        self.entrance_offset_m = new_offset_m
        if reset:
            self.anchor_m = None
            self.anchor_frame = None
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

    @staticmethod
    def _next_insertion_state(insert_value_mm, baseline_mm, last_relative_mm):
        if insert_value_mm is None:
            return last_relative_mm, baseline_mm, last_relative_mm

        values = np.asarray(insert_value_mm, dtype=float).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            return last_relative_mm, baseline_mm, last_relative_mm

        current_mm = float(values[0])
        if abs(current_mm) <= np.finfo(float).eps:
            return last_relative_mm, baseline_mm, last_relative_mm

        if baseline_mm is None:
            return 0.0, current_mm, 0.0
        relative_mm = current_mm - baseline_mm
        return relative_mm, baseline_mm, relative_mm

    @staticmethod
    def _point_from_tip(curve, requested_distance_m):
        segment_vectors = np.diff(curve, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        total_length = float(np.sum(segment_lengths))
        used_distance = float(np.clip(requested_distance_m, 0.0, total_length))
        remaining = used_distance

        if remaining <= np.finfo(float).eps:
            return (
                curve[-1].copy(),
                len(curve) - 2,
                0.0,
                used_distance,
                requested_distance_m > total_length,
            )

        for segment_index in range(len(segment_lengths) - 1, -1, -1):
            segment_length = float(segment_lengths[segment_index])
            if segment_length <= np.finfo(float).eps:
                continue
            if remaining <= segment_length:
                end = curve[segment_index + 1]
                start = curve[segment_index]
                fraction_from_end = remaining / segment_length
                point = end + fraction_from_end * (start - end)
                return (
                    point,
                    segment_index,
                    fraction_from_end,
                    used_distance,
                    requested_distance_m > total_length,
                )
            remaining -= segment_length

        return curve[0].copy(), 0, 1.0, used_distance, requested_distance_m > total_length

    @staticmethod
    def _validate_frames(curve, T, Y, Z):
        if T is None and Y is None and Z is None:
            return None
        if T is None or Y is None or Z is None:
            raise ValueError("T, Y and Z must be provided together")

        frames = tuple(np.asarray(axis, dtype=float) for axis in (T, Y, Z))
        for name, axis in zip(("T", "Y", "Z"), frames):
            if axis.shape != curve.shape:
                raise ValueError(f"{name} must have shape {curve.shape}, got {axis.shape}")
            if not np.all(np.isfinite(axis)):
                raise ValueError(f"{name} must contain only finite values")
        return frames

    @staticmethod
    def _project_to_rotation(matrix):
        u, singular_values, vt = np.linalg.svd(matrix)
        scale = max(float(singular_values[0]), 1.0)
        if singular_values[-1] <= 1e-6 * scale or np.linalg.det(matrix) <= 0.0:
            raise ValueError("local T/Y/Z frame is degenerate or left-handed")
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        return u @ correction @ vt

    @classmethod
    def _frame_at_location(cls, frames, segment_index, fraction_from_end):
        T, Y, Z = frames
        start_frame = np.column_stack(
            (T[segment_index], Y[segment_index], Z[segment_index])
        )
        end_frame = np.column_stack(
            (T[segment_index + 1], Y[segment_index + 1], Z[segment_index + 1])
        )
        start_frame = cls._project_to_rotation(start_frame)
        end_frame = cls._project_to_rotation(end_frame)
        rotations = Rotation.from_matrix(np.stack((start_frame, end_frame)))
        fraction_from_start = 1.0 - fraction_from_end
        return Slerp([0.0, 1.0], rotations)([fraction_from_start]).as_matrix()[0]

    def apply(self, curve_m, insert_value_mm=None, T=None, Y=None, Z=None):
        """Rigidly align one curve's entrance point and optional local frame."""
        curve = self._validate_curve(curve_m)
        if not self.enabled:
            return AnchorConstraintResult(
                curve_m=curve.copy(),
                applied=False,
                anchor_m=None,
                source_point_m=None,
                anchored_point_m=None,
                translation_m=np.zeros(3, dtype=float),
                rotation_matrix=np.eye(3, dtype=float),
                anchor_frame=None,
                source_frame=None,
                relative_insert_mm=0.0,
                requested_distance_m=0.0,
                used_distance_m=0.0,
                clamped=False,
            )

        frames = self._validate_frames(curve, T, Y, Z)
        recapture_joint_session = (
            frames is not None
            and self.anchor_m is not None
            and self.anchor_frame is None
        )
        anchor_m = None if recapture_joint_session else self.anchor_m
        anchor_frame = None if recapture_joint_session else self.anchor_frame
        baseline_mm = None if recapture_joint_session else self.insertion_baseline_mm
        last_relative_mm = 0.0 if recapture_joint_session else self.last_relative_insert_mm

        if anchor_m is None:
            anchor_m = curve[-1].copy() - np.array([self.entrance_offset_m, 0.0, 0.0])

        (
            relative_insert_mm,
            baseline_mm,
            last_relative_mm,
        ) = self._next_insertion_state(
            insert_value_mm,
            baseline_mm,
            last_relative_mm,
        )
        requested_distance_m = max(
            0.0,
            self.entrance_offset_m + relative_insert_mm / 1000.0,
        )
        (
            source_point_m,
            segment_index,
            fraction_from_end,
            used_distance_m,
            clamped,
        ) = self._point_from_tip(
            curve,
            requested_distance_m,
        )
        source_frame = None
        rotation_matrix = np.eye(3, dtype=float)
        if frames is not None:
            source_frame = self._frame_at_location(
                frames,
                segment_index,
                fraction_from_end,
            )
            if anchor_frame is None:
                anchor_frame = source_frame.copy()
            rotation_matrix = anchor_frame @ source_frame.T

        translation_m = anchor_m - source_point_m
        translated_curve = curve + translation_m
        anchored_curve = (
            (translated_curve - anchor_m) @ rotation_matrix.T + anchor_m
        )
        anchored_point_m = (
            (source_point_m + translation_m - anchor_m) @ rotation_matrix.T
            + anchor_m
        )

        self.anchor_m = anchor_m
        self.anchor_frame = anchor_frame
        self.insertion_baseline_mm = baseline_mm
        self.last_relative_insert_mm = last_relative_mm

        return AnchorConstraintResult(
            curve_m=anchored_curve,
            applied=True,
            anchor_m=anchor_m.copy(),
            source_point_m=source_point_m,
            anchored_point_m=anchored_point_m,
            translation_m=translation_m,
            rotation_matrix=rotation_matrix,
            anchor_frame=None if anchor_frame is None else anchor_frame.copy(),
            source_frame=None if source_frame is None else source_frame.copy(),
            relative_insert_mm=relative_insert_mm,
            requested_distance_m=requested_distance_m,
            used_distance_m=used_distance_m,
            clamped=clamped,
        )


def apply_entrance_anchor(constraint, curve_m, insert_value_mm=None, T=None, Y=None, Z=None):
    """Apply the shared live-pipeline stage and return only the output curve."""
    if not isinstance(constraint, EntranceAnchorConstraint):
        raise TypeError("constraint must be an EntranceAnchorConstraint")
    if T is None and Y is None and Z is None:
        return constraint.apply(curve_m, insert_value_mm).curve_m
    return constraint.apply(curve_m, insert_value_mm, T, Y, Z).curve_m
