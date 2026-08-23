"""Render translation-only bronchial entrance anchored Pickle recordings."""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entrance_anchor import EntranceAnchorConstraint


matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]

from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter


@dataclass(frozen=True)
class AnchoredPickleData:
    source_path: Path
    times_s: np.ndarray
    shapes_m: np.ndarray
    raw_inserts_mm: np.ndarray
    relative_inserts_mm: np.ndarray
    anchor_m: np.ndarray
    insertion_baseline_mm: float
    requested_distances_m: np.ndarray
    used_distances_m: np.ndarray
    clamped: np.ndarray
    anchor_residuals_m: np.ndarray


def _scalar_insert_or_none(value, path, frame_index):
    if value is None:
        return None
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f"{path} frame {frame_index} insert_value must be a finite scalar or None")
    return float(values[0])


def load_anchored_pickle(path):
    """Load one Demonstrator stream and apply the shared entrance constraint frame by frame."""
    path = Path(path)
    constraint = EntranceAnchorConstraint(enabled=True)
    timestamps = []
    shapes = []
    raw_inserts = []
    relative_inserts = []
    requested_distances = []
    used_distances = []
    clamped = []
    residuals = []

    with path.open("rb") as stream:
        frame_index = 0
        while True:
            try:
                frame = pickle.load(stream)
            except EOFError:
                break
            except Exception as exc:
                raise ValueError(f"{path} frame {frame_index + 1} could not be unpickled: {exc}") from exc

            frame_index += 1
            if not isinstance(frame, dict):
                raise ValueError(f"{path} frame {frame_index} must be a dictionary")
            missing = [key for key in ("timestamp", "r_results", "insert_value") if key not in frame]
            if missing:
                raise ValueError(f"{path} frame {frame_index} missing fields: {', '.join(missing)}")

            timestamp = float(frame["timestamp"])
            if not np.isfinite(timestamp):
                raise ValueError(f"{path} frame {frame_index} timestamp must be finite")
            shape = np.asarray(frame["r_results"], dtype=float)
            if shape.shape != (60, 3):
                raise ValueError(
                    f"{path} frame {frame_index} r_results must have shape (60, 3), got {shape.shape}"
                )
            if not np.all(np.isfinite(shape)):
                raise ValueError(f"{path} frame {frame_index} r_results contains NaN or inf")

            raw_insert = _scalar_insert_or_none(frame["insert_value"], path, frame_index)
            result = constraint.apply(shape, raw_insert)

            timestamps.append(timestamp)
            shapes.append(result.curve_m)
            raw_inserts.append(np.nan if raw_insert is None else raw_insert)
            relative_inserts.append(result.relative_insert_mm)
            requested_distances.append(result.requested_distance_m)
            used_distances.append(result.used_distance_m)
            clamped.append(result.clamped)
            residuals.append(float(np.linalg.norm(result.anchored_point_m - result.anchor_m)))

    if not shapes:
        raise ValueError(f"no frames found in {path}")
    if constraint.insertion_baseline_mm is None:
        raise ValueError(f"{path} has no finite non-zero insert_value baseline")

    timestamp_array = np.asarray(timestamps, dtype=float)
    times_s = timestamp_array - timestamp_array[0]
    if np.any(np.diff(times_s) < 0.0):
        raise ValueError(f"{path} timestamps are not monotonic")

    return AnchoredPickleData(
        source_path=path,
        times_s=times_s,
        shapes_m=np.stack(shapes),
        raw_inserts_mm=np.asarray(raw_inserts, dtype=float),
        relative_inserts_mm=np.asarray(relative_inserts, dtype=float),
        anchor_m=np.asarray(constraint.anchor_m, dtype=float),
        insertion_baseline_mm=float(constraint.insertion_baseline_mm),
        requested_distances_m=np.asarray(requested_distances, dtype=float),
        used_distances_m=np.asarray(used_distances, dtype=float),
        clamped=np.asarray(clamped, dtype=bool),
        anchor_residuals_m=np.asarray(residuals, dtype=float),
    )


def output_paths_for(trajectory_output, video_output, source_path):
    stem = Path(source_path).stem
    png_path = Path(trajectory_output) / f"{stem}_anchored_tip_trajectory.png"
    video_path = Path(video_output) / stem / f"{stem}_anchored_full.mp4"
    return png_path, video_path


def _configure_equal_axes(ax, points_mm):
    points = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    padding = np.maximum(2.0, 0.05 * span)
    lower = minimum - padding
    upper = maximum + padding
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(np.maximum(upper - lower, 1.0))
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_zlabel("Z / mm")
    ax.grid(True, color="lightgrey", linestyle=":", linewidth=0.6)


def render_tip_trajectory_png(data, output_path, title=None):
    """Render all constrained tip positions and the fixed entrance anchor."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tips_mm = data.shapes_m[:, -1, :] * 1000.0
    anchor_mm = data.anchor_m * 1000.0
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.plot(*tips_mm.T, color="red", linewidth=1.5, label="Anchored tip trajectory")
        ax.scatter(*tips_mm[0], color="green", s=65, label="Start tip")
        ax.scatter(*tips_mm[-1], color="blue", s=65, label="End tip")
        ax.scatter(*anchor_mm, color="black", marker="x", s=100, linewidths=2.5, label="Fixed entrance anchor")
        _configure_equal_axes(ax, np.vstack((tips_mm, anchor_mm)))
        ax.set_title(title or f"{data.source_path.stem} - Anchored Fiber Tip Trajectory")
        ax.legend(loc="upper right")
        ax.view_init(elev=20, azim=45)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


def _frame_indices(frame_count, max_frames):
    if frame_count <= 0 or max_frames <= 0:
        raise ValueError("frame_count and max_frames must be positive")
    return np.unique(np.linspace(0, frame_count - 1, min(frame_count, max_frames), dtype=int))


def render_shape_video(data, output_path, max_frames=1200, fps=20, dpi=100):
    """Render sampled full-shape frames, fixed entrance marker, and accumulated tip path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shapes_mm = data.shapes_m * 1000.0
    anchor_mm = data.anchor_m * 1000.0
    indices = _frame_indices(len(shapes_mm), int(max_frames))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _configure_equal_axes(ax, np.vstack((shapes_mm.reshape(-1, 3), anchor_mm)))
    ax.view_init(elev=20, azim=45)
    shape_line, = ax.plot([], [], [], color="blue", linewidth=2.0, label="Anchored fibre")
    tip_line, = ax.plot([], [], [], color="red", linewidth=1.0, alpha=0.8, label="Tip trajectory")
    tip_dot = ax.scatter([], [], [], color="red", s=35)
    ax.scatter(*anchor_mm, color="black", marker="x", s=100, linewidths=2.5, label="Fixed entrance anchor")
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right")
    writer = FFMpegWriter(fps=int(fps), bitrate=8000, metadata={"title": output_path.stem})
    tip_history = []

    try:
        with writer.saving(fig, str(output_path), dpi=int(dpi)):
            for frame_index in indices:
                shape = shapes_mm[frame_index]
                tip_history.append(shape[-1])
                history = np.asarray(tip_history)
                shape_line.set_data(shape[:, 0], shape[:, 1])
                shape_line.set_3d_properties(shape[:, 2])
                tip_line.set_data(history[:, 0], history[:, 1])
                tip_line.set_3d_properties(history[:, 2])
                tip_dot._offsets3d = ([shape[-1, 0]], [shape[-1, 1]], [shape[-1, 2]])
                time_text.set_text(
                    f"t={data.times_s[frame_index]:.2f} s  "
                    f"relative insertion={data.relative_inserts_mm[frame_index]:.2f} mm"
                )
                writer.grab_frame()
    finally:
        plt.close(fig)
    return output_path


def _capture_hhmm(path):
    try:
        return int(Path(path).stem.rsplit("_", 1)[-1][:4])
    except (ValueError, IndexError):
        return None


def discover_pickle_files(input_path, from_time=None, to_time=None):
    input_path = Path(input_path)
    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(input_path.glob("*.pkl"))
    else:
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    selected = []
    for path in candidates:
        hhmm = _capture_hhmm(path)
        if from_time is not None and (hhmm is None or hhmm < int(from_time)):
            continue
        if to_time is not None and (hhmm is None or hhmm > int(to_time)):
            continue
        selected.append(path)
    if not selected:
        raise ValueError("no Pickle files matched the requested time range")
    return selected


def _atomic_render(render, final_path):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.stem}.part{final_path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        render(temporary)
        temporary.replace(final_path)
    finally:
        temporary.unlink(missing_ok=True)


def process_file(source, trajectory_output, video_output, max_frames, fps, overwrite=False):
    png_path, video_path = output_paths_for(trajectory_output, video_output, source)
    if not overwrite and png_path.exists() and video_path.exists():
        print(f"[skip] outputs already exist for {Path(source).name}")
        return None

    print(f"[load] {source}")
    data = load_anchored_pickle(source)
    if overwrite or not png_path.exists():
        _atomic_render(lambda path: render_tip_trajectory_png(data, path), png_path)
    if overwrite or not video_path.exists():
        _atomic_render(
            lambda path: render_shape_video(data, path, max_frames=max_frames, fps=fps),
            video_path,
        )

    unclamped = ~data.clamped
    max_residual = float(np.max(data.anchor_residuals_m[unclamped])) if np.any(unclamped) else float("nan")
    print(
        f"[done] {Path(source).name}: frames={len(data.times_s)}, "
        f"baseline={data.insertion_baseline_mm:.6f} mm, clamps={int(np.sum(data.clamped))}, "
        f"max_anchor_residual={max_residual:.3e} m"
    )
    print(f"       PNG={png_path}")
    print(f"       MP4={video_path}")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one Demonstrator .pkl or a directory")
    parser.add_argument("--from-time", type=int, help="inclusive HHMM filename filter, e.g. 1922")
    parser.add_argument("--to-time", type=int, help="inclusive HHMM filename filter, e.g. 1956")
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--video-output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    files = discover_pickle_files(args.input, args.from_time, args.to_time)
    failures = []
    for index, source in enumerate(files, start=1):
        print(f"[{index}/{len(files)}]")
        try:
            process_file(
                source,
                args.trajectory_output,
                args.video_output,
                args.max_frames,
                args.fps,
                args.overwrite,
            )
        except Exception as exc:
            failures.append((source, exc))
            print(f"[failed] {source}: {exc}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
