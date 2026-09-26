"""Contract tests for the public algorithm and reproducible replay path."""

from tests.language_data import text as localized_text
import json
import math
import tempfile
import unittest
from pathlib import Path

from navguide.core.context import ContextSmoother, GuidanceContext, SceneType, TaskMode
from navguide.core.gating import InertialMotionGate
from navguide.core.pipeline import EvaluationCondition, NavGuidePipeline
from navguide.core.proximity import CameraIntrinsics, ProximityBin, ProximityEstimator
from navguide.core.selection import (
    DetectionCandidate,
    SemanticMaximizationPolicy,
    compute_iou,
    same_class_dedup,
)
from scripts.evaluate import read_frames, replay

ROOT = Path(__file__).resolve().parents[1]


def ordinary(category="chair", **kwargs):
    values = dict(category=category, confidence=0.9, bbox=(100, 100, 180, 210))
    values.update(kwargs)
    return DetectionCandidate(**values)


class AlgorithmContractTests(unittest.TestCase):
    def test_iou_strict_boundary_and_category_isolation(self):
        high = ordinary(bbox=(0, 0, 4, 1))
        boundary = ordinary(confidence=0.8, bbox=(1, 0, 5, 1))
        overlap = ordinary(confidence=0.7, bbox=(0.99, 0, 4.99, 1))
        other = ordinary("table", confidence=0.6, bbox=(0, 0, 4, 1))
        self.assertEqual(compute_iou(high.bbox, boundary.bbox), 0.6)
        self.assertEqual(
            same_class_dedup([high, boundary, overlap, other]), [high, boundary, other]
        )

    def test_five_factor_product_is_auditable(self):
        pipeline = NavGuidePipeline()
        result = pipeline.process([ordinary()], now=0)
        cue = result.selected_cues[0]
        self.assertEqual(
            set(cue.relevance_factors), {"task", "scene", "user", "proximity", "hazard"}
        )
        self.assertAlmostEqual(
            cue.relevance_score, cue.confidence * math.prod(cue.relevance_factors.values())
        )

    def test_gate_boundary_both_signs(self):
        cue = NavGuidePipeline().process([ordinary()], now=0).selected_cues[0]
        gate = InertialMotionGate()
        for yaw in (24.999, -24.999, 0):
            self.assertTrue(gate.evaluate_gate(cue, yaw))
        for yaw in (25, -25, 25.001, -25.001):
            self.assertFalse(gate.evaluate_gate(cue, yaw))

    def test_repetition_exact_boundary(self):
        pipeline = NavGuidePipeline()
        self.assertTrue(pipeline.process([ordinary()], now=0).should_speak)
        self.assertFalse(pipeline.process([ordinary()], now=2.999).should_speak)
        self.assertTrue(pipeline.process([ordinary()], now=3.0).should_speak)

    def test_deferred_observation_does_not_commit_history(self):
        pipeline = NavGuidePipeline()
        initial = pipeline.process([ordinary()], yaw_rate_dps=25, now=0)
        self.assertEqual(len(initial.deferred_cues), 1)
        self.assertEqual(pipeline.smp.cue_history, {})
        self.assertTrue(pipeline.process([ordinary()], yaw_rate_dps=0, now=0.1).should_speak)

    def test_no_queued_replay_after_object_disappears(self):
        pipeline = NavGuidePipeline()
        pipeline.process([ordinary()], yaw_rate_dps=35, now=0)
        result = pipeline.process([], yaw_rate_dps=0, now=0.1)
        self.assertFalse(result.should_speak)
        self.assertEqual(result.eligible_cues, [])

    def test_urgent_repetition_is_not_a_gate_bypass(self):
        pipeline = NavGuidePipeline()
        item = ordinary("car", urgency_override=True)
        first = pipeline.process([item], yaw_rate_dps=35, now=0)
        repeated = pipeline.process([item], yaw_rate_dps=35, now=0.1)
        self.assertTrue(first.should_speak)
        self.assertFalse(repeated.should_speak)
        self.assertEqual(repeated.repetition_suppressed_count, 1)
        self.assertEqual(pipeline.motion_gate.stats.total_evaluated, 1)

    def test_requested_target_repetition_is_not_a_gate_bypass(self):
        pipeline = NavGuidePipeline()
        pipeline.set_task_mode(TaskMode.TARGET_SEARCH, "cup")
        item = ordinary("cup")
        self.assertTrue(pipeline.process([item], yaw_rate_dps=-35, now=0).should_speak)
        self.assertFalse(pipeline.process([item], yaw_rate_dps=-35, now=0.1).should_speak)

    def test_urgent_and_target_cannot_recover_excluded_items(self):
        for urgent, target in ((True, None), (False, "cup")):
            with self.subTest(urgent=urgent, target=target):
                pipeline = NavGuidePipeline()
                if target:
                    pipeline.set_task_mode(TaskMode.TARGET_SEARCH, target)
                pipeline.set_user_weights({"cup": 0})
                candidates = [ordinary("cup", urgency_override=urgent)] + [
                    ordinary(cat) for cat in ("chair", "table", "door")
                ]
                result = pipeline.process(candidates, yaw_rate_dps=35, now=0)
                self.assertNotIn("cup", [cue.category for cue in result.selected_cues])
                self.assertLessEqual(len(result.selected_cues), 3)

    def test_selection_metrics_are_before_temporal_and_motion_stages(self):
        pipeline = NavGuidePipeline()
        item = ordinary("bollard", urgency_override=False)
        first = pipeline.process([item], now=0)
        repeated = pipeline.process([item], now=1)
        gated = pipeline.process([item], yaw_rate_dps=25, now=3)
        for result in (first, repeated, gated):
            self.assertEqual(result.retained_item_count, 1)
            self.assertEqual(result.critical_retained_count, 1)
            self.assertEqual(result.item_reduction, 0)
            self.assertEqual(result.conditional_retention, 1)
        self.assertFalse(repeated.should_speak)
        self.assertFalse(gated.should_speak)

    def test_critical_count_preserves_exact_frontend_identity(self):
        pipeline = NavGuidePipeline(top_k=1)
        pipeline.set_user_weights({"chair": 1000, "cone": 0})
        result = pipeline.process([ordinary("cone", is_hazard=True), ordinary("chair")], now=0)
        self.assertEqual(result.critical_detected_count, 1)
        self.assertEqual(result.critical_retained_count, 0)
        self.assertEqual(result.conditional_retention, 0)

    def test_empty_denominators_are_unavailable(self):
        pipeline = NavGuidePipeline()
        result = pipeline.process([], now=0)
        self.assertIsNone(result.item_reduction)
        self.assertIsNone(result.conditional_retention)
        self.assertIsNone(pipeline.get_metrics_summary()["conditional_retention_rate"])

    def test_processing_does_not_fabricate_audio_trigger_latency(self):
        pipeline = NavGuidePipeline()
        result = pipeline.process([ordinary()], now=0)
        self.assertIsNone(result.capture_to_trigger_latency_ms)
        self.assertGreaterEqual(result.processing_latency_ms, 0)
        self.assertEqual(pipeline.get_metrics_summary()["latency_ms"]["count"], 0)
        pipeline.record_audio_trigger(100, 100.125, result)
        self.assertEqual(result.capture_to_trigger_latency_ms, 125)
        self.assertEqual(pipeline.get_metrics_summary()["latency_ms"]["count"], 1)
        with self.assertRaises(ValueError):
            pipeline.record_audio_trigger(100, 99)

    def test_all_top_three_items_appear_in_speech(self):
        result = NavGuidePipeline().process(
            [ordinary(cat) for cat in ("chair", "table", "door")], now=0
        )
        self.assertEqual(len(result.eligible_cues), 3)
        for phrase in result.guidance_phrases:
            self.assertIn(phrase, result.speech_text)

    def test_broadcast_has_correct_directions_proximity_and_no_cap(self):
        pipeline = NavGuidePipeline(EvaluationCondition.B)
        items = [
            ordinary(cat, bbox=(500, 0, 600, 300)) for cat in ("chair", "table", "door", "cup")
        ]
        first = pipeline.process(items, yaw_rate_dps=40, now=0)
        repeated = pipeline.process(items, yaw_rate_dps=40, now=0.1)
        self.assertEqual(len(first.eligible_cues), 4)
        self.assertEqual(len(repeated.eligible_cues), 4)
        self.assertEqual(first.eligible_cues[0].clock_hour, 2)
        for cue in first.eligible_cues:
            self.assertEqual(
                cue.proximity_bin, pipeline.prox_est.categorize_proximity(cue.distance_m)
            )

    def test_tracking_baseline_emits_events_and_reappearance(self):
        pipeline = NavGuidePipeline(EvaluationCondition.T)
        item = ordinary(track_id=7)
        self.assertTrue(pipeline.process([item], now=0).should_speak)
        self.assertFalse(pipeline.process([item], now=1).should_speak)
        moved = ordinary(track_id=7, bbox=(400, 100, 480, 210))
        self.assertTrue(pipeline.process([moved], now=1.5).should_speak)
        pipeline.process([], now=2)
        self.assertTrue(pipeline.process([moved], now=4.5).should_speak)

    def test_tracking_without_ids_associates_overlapping_observations(self):
        pipeline = NavGuidePipeline(EvaluationCondition.T)
        self.assertTrue(pipeline.process([ordinary()], now=0).should_speak)
        self.assertFalse(
            pipeline.process([ordinary(bbox=(101, 100, 181, 210))], now=1).should_speak
        )

    def test_remote_condition_never_silently_uses_local_pipeline(self):
        with self.assertRaises(ValueError):
            NavGuidePipeline(EvaluationCondition.REMOTE)
        pipeline = NavGuidePipeline()
        pipeline.condition = EvaluationCondition.REMOTE
        with self.assertRaises(ValueError):
            pipeline.process([], now=0)

    def test_context_snapshot_is_not_mutated_by_later_updates(self):
        pipeline = NavGuidePipeline()
        first = pipeline.process([ordinary()], now=0)
        pipeline.set_task_mode(TaskMode.TARGET_SEARCH, "cup")
        pipeline.set_user_weights({"chair": 2})
        self.assertEqual(first.context.task_mode, TaskMode.PATH_NAVIGATION)
        self.assertEqual(first.context.user_weights, {})

    def test_target_matches_words_not_substrings(self):
        context = GuidanceContext(task_mode=TaskMode.TARGET_SEARCH, target_query="car")
        self.assertTrue(context.matches_target("car"))
        self.assertFalse(context.matches_target("carpet"))
        context.target_query = localized_text("object.cup")
        self.assertTrue(context.matches_target("cup"))
        context.target_query = " "
        self.assertFalse(context.matches_target("chair"))

    def test_red_substring_is_not_a_signal(self):
        pipeline = NavGuidePipeline()
        result = pipeline.process([ordinary("red backpack")], now=0)
        self.assertEqual(result.critical_detected_count, 0)
        self.assertNotIn(localized_text("signal.red"), result.speech_text)
        actual = pipeline.process([ordinary("traffic light", signal_color="red")], now=1)
        self.assertEqual(actual.selected_cues[0].action_type, "stop")
        self.assertTrue(actual.selected_cues[0].urgency_flag)

    def test_green_and_obstacle_wording_do_not_invent_safe_directions(self):
        pipeline = NavGuidePipeline()
        green = pipeline.process([ordinary("traffic light", signal_color="green")], now=0)
        self.assertNotIn(localized_text("action.proceed_forward"), green.speech_text)
        obstacle = pipeline.process([ordinary("blind_path_obstacle", is_hazard=True)], now=1)
        self.assertNotIn(localized_text("action.walk_straight"), obstacle.speech_text)

    def test_signal_color_change_is_new_semantic_content(self):
        pipeline = NavGuidePipeline()
        green = pipeline.process([ordinary("traffic light", signal_color="green")], now=0)
        red = pipeline.process([ordinary("traffic light", signal_color="red")], now=0.1)
        self.assertTrue(red.should_speak)
        self.assertNotEqual(
            green.selected_cues[0].semantic_signature, red.selected_cues[0].semantic_signature
        )

    def test_invalid_detection_inputs_fail_early(self):
        for kwargs in (
            {"confidence": float("nan")},
            {"confidence": 1.1},
            {"bbox": (0, 0, 0, 1)},
            {"bbox": (0, 0, 1, float("inf"))},
            {"category": " "},
            {"urgency_override": "false"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ordinary(**kwargs)

    def test_invalid_configuration_and_clock_inputs(self):
        for operation in (
            lambda: SemanticMaximizationPolicy(top_k=0),
            lambda: SemanticMaximizationPolicy(repetition_window_sec=-1),
            lambda: InertialMotionGate(0),
            lambda: ContextSmoother(2, 3),
            lambda: GuidanceContext(user_weights={"chair": float("nan")}),
            lambda: CameraIntrinsics(vertical_fov_deg=180),
            lambda: ProximityEstimator(step_length_m=0),
            lambda: ProximityEstimator(class_height_priors={"chair": -1}),
        ):
            with self.assertRaises(ValueError):
                operation()
        pipeline = NavGuidePipeline()
        for kwargs in ({"yaw_rate_dps": float("nan")}, {"frame_width": 0}, {"now": float("inf")}):
            with self.assertRaises(ValueError):
                pipeline.process([], **kwargs)
        pipeline.process([], now=10)
        with self.assertRaises(ValueError):
            pipeline.process([], now=9)

    def test_zero_timestamp_and_calibrated_focal_length(self):
        smp = SemanticMaximizationPolicy()
        self.assertTrue(smp.select_and_build_cues([ordinary()], GuidanceContext(), 640, 480, now=0))
        self.assertFalse(
            smp.select_and_build_cues([ordinary()], GuidanceContext(), 640, 480, now=0)
        )
        intrinsics = CameraIntrinsics(reference_height=1000, calibrated_focal_length_y=500)
        self.assertEqual(intrinsics.get_focal_length_y(500), 250)
        estimator = ProximityEstimator(intrinsics, {"chair": 1})
        self.assertEqual(estimator.estimate_distance_m((0, 0, 100, 100), "chair", 640, 500), 2.5)
        self.assertEqual(estimator.categorize_proximity(1.5), ProximityBin.NEAR)
        self.assertEqual(estimator.categorize_proximity(2.5), ProximityBin.MEDIUM)
        self.assertEqual(estimator.categorize_proximity(4.5), ProximityBin.FAR)

    def test_smoother_ties_keep_current_scene(self):
        smoother = ContextSmoother(4, 2)
        for scene in (SceneType.SIDEWALK, SceneType.SIDEWALK, SceneType.INDOOR, SceneType.INDOOR):
            smoother.update(scene)
        self.assertEqual(smoother.current_scene, SceneType.SIDEWALK)


class ReplayTests(unittest.TestCase):
    def test_synthetic_fixture_replay_produces_explainable_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path, trace_path = (
                Path(directory) / "report.json",
                Path(directory) / "trace.jsonl",
            )
            report = replay(
                ROOT / "examples/synthetic_replay.jsonl", ["P", "T", "B"], report_path, trace_path
            )
            self.assertEqual(report, json.loads(report_path.read_text()))
            self.assertEqual(len(trace_path.read_text().splitlines()), 30)
            expected = {"P": (13, 8), "T": (14, 8), "B": (15, 15)}
            for condition, (retained, eligible) in expected.items():
                summary = report["conditions"][condition]
                self.assertEqual(summary["frames"], 10)
                self.assertEqual(summary["total_raw_items"], 15)
                self.assertEqual(summary["total_retained_items"], retained)
                self.assertEqual(summary["total_eligible_items"], eligible)
                self.assertEqual(summary["latency_ms"]["count"], 0)
            first = json.loads(trace_path.read_text().splitlines()[0])
            self.assertEqual(len(first["selected"][0]["factors"]), 5)

    def test_replay_rejects_invalid_frames_with_line_number(self):
        bad_lines = [
            '{"timestamp":0,"yaw_rate_dps":NaN,"detections":[]}',
            '{"timestamp":0,"detections":[]}',
            '{"timestamp":0,"yaw_rate_dps":0,"detections":[{"category":"cup","confidence":0.5,"bbox":[0,0,0,1]}]}',
            '{"timestamp":0,"yaw_rate_dps":0,"frame_width":false,"detections":[]}',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            for line in bad_lines:
                path.write_text(line)
                with self.subTest(line=line), self.assertRaisesRegex(ValueError, "bad.jsonl:1"):
                    list(read_frames(path))

    def test_replay_never_overwrites_its_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            content = '{"timestamp":0,"yaw_rate_dps":0,"detections":[]}\n'
            path.write_text(content)
            with self.assertRaises(ValueError):
                replay(path, ["P"], path)
            self.assertEqual(path.read_text(), content)

    def test_replay_creates_output_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports/nested/summary.json"
            frames = Path(directory) / "traces/nested/frames.jsonl"
            replay(ROOT / "examples/synthetic_replay.jsonl", ["P"], output, frames)
            self.assertTrue(output.is_file())
            self.assertEqual(len(frames.read_text().splitlines()), 10)


if __name__ == "__main__":
    unittest.main()
