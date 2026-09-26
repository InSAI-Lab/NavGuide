#!/usr/bin/env python3
"""Replay shared frontend JSONL detections without camera, models or cloud."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navguide.core.context import SceneType, TaskMode
from navguide.core.pipeline import EvaluationCondition, NavGuidePipeline
from navguide.core.proximity import CameraIntrinsics, ProximityEstimator
from navguide.core.selection import ActionCue, DetectionCandidate


def reject_constant(value: str):
    raise ValueError(f"nonfinite JSON number: {value}")


def load_json(text: str) -> Any:
    return json.loads(text, parse_constant=reject_constant)


def read_frames(path: Path) -> Iterator[Dict[str, Any]]:
    previous = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                frame = load_json(line)
                if not isinstance(frame, dict):
                    raise ValueError("each frame must be a JSON object")
                timestamp = frame["timestamp"]
                yaw = frame["yaw_rate_dps"]
                if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                    raise ValueError("timestamp must be a finite number in seconds")
                if previous is not None and timestamp < previous:
                    raise ValueError("timestamps must be nondecreasing")
                if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
                    raise ValueError("yaw_rate_dps must be finite degrees per second")
                if not isinstance(frame["detections"], list):
                    raise ValueError("detections must be a list")
                frame["candidates"] = [DetectionCandidate(**entry) for entry in frame["detections"]]
                frame["scene"] = SceneType(frame.get("scene", "sidewalk"))
                if "task_mode" in frame:
                    frame["task_mode"] = TaskMode(frame["task_mode"])
                for dimension in ("frame_width", "frame_height"):
                    value = frame.get(dimension, 640 if dimension == "frame_width" else 480)
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        raise ValueError(f"{dimension} must be a positive integer")
                    frame[dimension] = value
                previous = timestamp
                yield frame
            except (KeyError, ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def cue_record(cue: ActionCue) -> Dict[str, Any]:
    return {"category": cue.category, "bbox": cue.bbox,
            "track_id": cue.source_candidate.track_id if cue.source_candidate else None,
            "score": cue.relevance_score, "factors": cue.relevance_factors,
            "coarse_distance_m": cue.distance_m, "signature": cue.semantic_signature,
            "urgent": cue.urgency_flag, "requested_target": cue.requested_target_flag}


def replay(input_path: Path, conditions, output_path=None, frames_output_path=None,
           calibration_path=None, lang="zh") -> Dict[str, Any]:
    calibration = load_json(Path(calibration_path).read_text(encoding="utf-8")) if calibration_path else {}
    intrinsics_config = calibration.get("intrinsics", {})
    priors = calibration.get("class_height_priors", {})
    pipelines = {
        condition: NavGuidePipeline(
            EvaluationCondition(condition), lang=lang,
            proximity_estimator=ProximityEstimator(CameraIntrinsics(**intrinsics_config), priors),
        ) for condition in conditions
    }
    if not pipelines:
        raise ValueError("at least one local condition is required")
    paths = [Path(value).resolve() for value in (input_path, output_path, frames_output_path, calibration_path) if value]
    if len(paths) != len(set(paths)):
        raise ValueError("input, output, frame output and calibration paths must be distinct")
    for output in (output_path, frames_output_path):
        if output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    digest = hashlib.sha256()
    with input_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    first = last = None
    with ExitStack() as stack:
        frame_stream = stack.enter_context(Path(frames_output_path).open("w", encoding="utf-8")) if frames_output_path else None
        for frame_index, frame in enumerate(read_frames(input_path)):
            timestamp = frame["timestamp"]
            first = timestamp if first is None else first
            last = timestamp
            for condition, pipeline in pipelines.items():
                if "task_mode" in frame:
                    pipeline.set_task_mode(frame["task_mode"], frame.get("target_query"))
                elif "target_query" in frame:
                    pipeline.set_task_mode(pipeline.context.task_mode, frame["target_query"])
                if "user_weights" in frame:
                    pipeline.set_user_weights(frame["user_weights"])
                result = pipeline.process(
                    frame["candidates"], yaw_rate_dps=frame["yaw_rate_dps"],
                    raw_scene=frame["scene"], frame_width=frame["frame_width"],
                    frame_height=frame["frame_height"], now=timestamp,
                )
                if frame_stream:
                    record = {
                        "frame": frame_index, "timestamp": timestamp, "condition": condition,
                        "raw_items": result.raw_item_count, "retained_items": result.retained_item_count,
                        "critical_detected": result.critical_detected_count,
                        "critical_retained": result.critical_retained_count,
                        "temporally_suppressed": result.repetition_suppressed_count,
                        "selected": [cue_record(cue) for cue in result.selected_cues],
                        "eligible": [cue_record(cue) for cue in result.eligible_cues],
                        "deferred": [cue_record(cue) for cue in result.deferred_cues],
                        "phrases": result.guidance_phrases,
                    }
                    frame_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    span = last - first if first is not None else 0.0
    report = {
        "schema_version": 1, "input": input_path.name, "input_sha256": digest.hexdigest(),
        "timestamp_span_sec": span,
        "implementation": {
            "weight_profile": "source distribution defaults",
            "top_k": 3, "same_class_iou_threshold": 0.6,
            "repetition_window_sec": 3.0, "yaw_rate_threshold_dps": 25.0,
            "history_commit_stage": "motion-gate eligible content",
            "repetition_scope": "all cue flags; no urgency or target bypass",
            "calibration": calibration or {"intrinsics": {"vertical_fov_deg": 60.0}, "source": "uncalibrated defaults"},
            "baseline_T": "new, reappearing or semantic-change track events; same-class IoU fallback 0.3; expiry 3 s; no Top-K",
        },
        "conditions": {},
        "limitations": [
            "Metrics cover supplied frontend detections; detection recall requires annotated scene data.",
            "Critical items use class and hazard flag proxies; safety evaluation requires independent annotations.",
            "Eligible item rates count content released by the policy before audio scheduling.",
            "Processing time covers local policy execution and text generation.",
            "Paper result reproduction requires the original study records, parameters and audio-trigger measurements.",
            "The bundled example is a synthetic fixture for functional validation.",
        ],
    }
    for condition, pipeline in pipelines.items():
        summary = pipeline.get_metrics_summary()
        summary["eligible_items_per_minute_over_timestamp_span"] = (
            pipeline.total_eligible_items * 60.0 / span if span > 0 else None)
        report["conditions"][condition] = summary
    if output_path:
        Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples" / "synthetic_replay.jsonl")
    parser.add_argument("--conditions", nargs="+", choices=["P", "T", "B"], default=["P", "T", "B"])
    parser.add_argument("--output", type=Path, help="write the aggregate JSON report")
    parser.add_argument("--frames-output", type=Path, help="write decision traces as JSONL")
    parser.add_argument("--calibration", type=Path, help="camera intrinsics and class height prior JSON")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh")
    args = parser.parse_args(argv)
    try:
        report = replay(args.input, args.conditions, args.output, args.frames_output, args.calibration, args.lang)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if not args.output:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
