# -*- coding: utf-8 -*-
"""Tests for cue selection, motion gating, phrasing and pipeline statistics."""

from tests.language_data import text as localized_text

import time
import unittest

from navguide.core.context import ContextSmoother, GuidanceContext, SceneType, TaskMode
from navguide.core.proximity import CameraIntrinsics, ProximityBin, ProximityEstimator
from navguide.core.selection import (
    ActionCue,
    DetectionCandidate,
    SemanticMaximizationPolicy,
    compute_iou,
    same_class_dedup,
)
from navguide.core.gating import InertialMotionGate
from navguide.core.phrasing import ActionFirstPhraser
from navguide.core.pipeline import EvaluationCondition, NavGuidePipeline


class TestNavGuide(unittest.TestCase):

    def test_same_class_dedup(self):
        """Test SameClassDedup removes overlaps > 0.6 for same class only."""
        # Two overlapping cars (IoU > 0.6)
        cand1 = DetectionCandidate(category="car", confidence=0.9, bbox=(100, 100, 200, 200))
        cand2 = DetectionCandidate(category="car", confidence=0.7, bbox=(105, 105, 205, 205))
        # Cross-category overlaps remain separate detections.
        cand3 = DetectionCandidate(category="person", confidence=0.85, bbox=(110, 110, 190, 190))
        # A separate car far away (IoU = 0)
        cand4 = DetectionCandidate(category="car", confidence=0.8, bbox=(400, 400, 500, 500))

        raw = [cand1, cand2, cand3, cand4]
        deduped = same_class_dedup(raw, iou_threshold=0.6)

        # cand2 should be suppressed by cand1; cand3 (person) and cand4 (distant car) kept
        self.assertEqual(len(deduped), 3)
        self.assertIn(cand1, deduped)
        self.assertNotIn(cand2, deduped)
        self.assertIn(cand3, deduped)
        self.assertIn(cand4, deduped)

    def test_proximity_estimator(self):
        """Test pinhole height-based distance and coarse proximity binning."""
        estimator = ProximityEstimator()
        # Person height prior = 1.7m. Frame H = 480.
        # If person occupies large height (e.g. 300px), should be close (< 2m)
        d_close = estimator.estimate_distance_m(
            bbox=(100, 50, 200, 350), class_name="person", frame_width=640, frame_height=480
        )
        # If person occupies small height (e.g. 40px), should be far (> 4m)
        d_far = estimator.estimate_distance_m(
            bbox=(100, 50, 150, 90), class_name="person", frame_width=640, frame_height=480
        )
        self.assertLess(d_close, d_far)
        self.assertLess(d_close, 3.0)
        self.assertGreater(d_far, 4.0)

        # Check binning
        self.assertEqual(estimator.categorize_proximity(1.2), ProximityBin.CRITICAL_NEAR)
        self.assertEqual(estimator.categorize_proximity(2.0), ProximityBin.NEAR)
        self.assertEqual(estimator.categorize_proximity(3.5), ProximityBin.MEDIUM)
        self.assertEqual(estimator.categorize_proximity(6.0), ProximityBin.FAR)

    def test_context_smoother(self):
        """Test that transient scene switches are filtered out."""
        smoother = ContextSmoother(window_size=8, switch_threshold=4)
        smoother.reset(SceneType.SIDEWALK)

        # Normal sidewalk frames
        smoother.update(SceneType.SIDEWALK)
        smoother.update(SceneType.SIDEWALK)
        self.assertEqual(smoother.current_scene, SceneType.SIDEWALK)

        # Single transient glitch frame (e.g. indoor misclassification)
        smoother.update(SceneType.INDOOR)
        # Should stay SIDEWALK due to smoothing
        self.assertEqual(smoother.current_scene, SceneType.SIDEWALK)

        # Persistent crosswalk observations (>= 4 times)
        for _ in range(4):
            smoother.update(SceneType.CROSSWALK)
        # Now confirmed switch to CROSSWALK
        self.assertEqual(smoother.current_scene, SceneType.CROSSWALK)

    def test_smp_5factor_ranking_and_topk(self):
        """Test SMP factorized scoring and Top-3 budget."""
        smp = SemanticMaximizationPolicy(top_k=3)
        ctx = GuidanceContext(task_mode=TaskMode.PATH_NAVIGATION, scene_type=SceneType.SIDEWALK)

        # Create 5 detections of varying relevance
        d1 = DetectionCandidate(
            category="blindpath", confidence=0.8, bbox=(200, 200, 440, 450)
        )  # Very high task relevance
        d2 = DetectionCandidate(
            category="car", confidence=0.85, bbox=(250, 100, 390, 300), is_moving=True
        )  # Moving hazard
        d3 = DetectionCandidate(
            category="bollard", confidence=0.75, bbox=(280, 250, 360, 400), is_hazard=True
        )  # Obstacle on sidewalk
        d4 = DetectionCandidate(
            category="cup", confidence=0.99, bbox=(10, 10, 30, 40)
        )  # Confident but irrelevant in path nav
        d5 = DetectionCandidate(
            category="chair", confidence=0.90, bbox=(10, 100, 80, 200)
        )  # Low relevance

        cues = smp.select_and_build_cues(
            [d1, d2, d3, d4, d5], ctx, frame_width=640, frame_height=480, now=100.0
        )

        # Top-3 bound: Must retain at most 3 candidates
        self.assertLessEqual(len(cues), 3)

        # Blindpath, car, bollard should rank above irrelevant cup despite cup's 0.99 confidence
        retained_categories = [c.category for c in cues]
        self.assertNotIn("cup", retained_categories)
        self.assertIn("blindpath", retained_categories)

    def test_target_search_mode_weighting(self):
        """Test that in TARGET_SEARCH mode, target query receives highest priority."""
        smp = SemanticMaximizationPolicy(top_k=3)
        ctx = GuidanceContext(task_mode=TaskMode.TARGET_SEARCH, target_query="cup")

        d_target = DetectionCandidate(category="cup", confidence=0.7, bbox=(250, 200, 350, 350))
        d_other1 = DetectionCandidate(category="chair", confidence=0.9, bbox=(100, 100, 200, 250))
        d_other2 = DetectionCandidate(category="table", confidence=0.9, bbox=(400, 100, 550, 300))

        cues = smp.select_and_build_cues(
            [d_target, d_other1, d_other2], ctx, frame_width=640, frame_height=480, now=100.0
        )
        self.assertGreater(len(cues), 0)
        # Target must be selected as top item and flagged as target
        top_cue = cues[0]
        self.assertEqual(top_cue.category, "cup")
        self.assertTrue(top_cue.requested_target_flag)

    def test_repetition_control_3s_window(self):
        """Test that identical semantic signatures within 3s are suppressed."""
        smp = SemanticMaximizationPolicy(repetition_window_sec=3.0)
        ctx = GuidanceContext(task_mode=TaskMode.PATH_NAVIGATION)

        d1 = DetectionCandidate(
            category="pole", confidence=0.8, bbox=(280, 200, 360, 420), is_hazard=False
        )

        # First call at t = 0.0: should be allowed
        cues_t0 = smp.select_and_build_cues([d1], ctx, frame_width=640, frame_height=480, now=0.0)
        self.assertEqual(len(cues_t0), 1)

        # Second call at t = 1.5s (within 3s): should be suppressed
        cues_t1 = smp.select_and_build_cues([d1], ctx, frame_width=640, frame_height=480, now=1.5)
        self.assertEqual(len(cues_t1), 0)

        # Third call at t = 3.5s (after 3s): should be allowed again
        cues_t3 = smp.select_and_build_cues([d1], ctx, frame_width=640, frame_height=480, now=3.5)
        self.assertEqual(len(cues_t3), 1)

    def test_inertial_motion_gating(self):
        """
        Test Equation 2: g_c(t) = 1[ |omega_t| < 25°/s or u_c or q_c ]
        and verify no queued replay for deferred cues.
        """
        gate = InertialMotionGate(yaw_rate_threshold_dps=25.0)

        # 1. Ordinary cue: not urgent, not target
        ordinary_cue = ActionCue(
            category="pole",
            category_zh=localized_text("object.pole"),
            confidence=0.8,
            bbox=(100, 100, 150, 300),
            distance_m=3.0,
            proximity_bin=ProximityBin.MEDIUM,
            clock_hour=12,
            clock_direction_zh=localized_text("clock.twelve"),
            clock_direction_en="12 o'clock",
            action_type="avoid",
            action_zh=localized_text("action.caution"),
            action_en="Notice",
            urgency_flag=False,
            requested_target_flag=False,
        )

        # Motion below the threshold allows ordinary cues.
        eligible, deferred = gate.filter_eligible_cues([ordinary_cue], yaw_rate_dps=10.0)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(len(deferred), 0)

        # Motion above the threshold defers ordinary cues.
        eligible, deferred = gate.filter_eligible_cues([ordinary_cue], yaw_rate_dps=35.0)
        self.assertEqual(len(eligible), 0)
        self.assertEqual(len(deferred), 1)

        # 2. Urgent cue: u_c = True
        urgent_cue = ActionCue(
            category="car",
            category_zh=localized_text("object.car"),
            confidence=0.9,
            bbox=(200, 100, 400, 350),
            distance_m=1.8,
            proximity_bin=ProximityBin.CRITICAL_NEAR,
            clock_hour=12,
            clock_direction_zh=localized_text("clock.twelve"),
            clock_direction_en="12 o'clock",
            action_type="avoid",
            action_zh=localized_text("action.avoid"),
            action_en="Caution yield",
            urgency_flag=True,
            requested_target_flag=False,
        )
        # Urgent cues remain eligible during rapid turns.
        eligible, deferred = gate.filter_eligible_cues([urgent_cue], yaw_rate_dps=40.0)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(len(deferred), 0)

        # 3. Requested target cue: q_c = True
        target_cue = ActionCue(
            category="cup",
            category_zh=localized_text("object.cup"),
            confidence=0.85,
            bbox=(200, 200, 250, 300),
            distance_m=1.0,
            proximity_bin=ProximityBin.NEAR,
            clock_hour=1,
            clock_direction_zh=localized_text("clock.one"),
            clock_direction_en="1 o'clock",
            action_type="found",
            action_zh=localized_text("action.target_found"),
            action_en="Target found",
            urgency_flag=False,
            requested_target_flag=True,
        )
        # Requested targets remain eligible during rapid turns.
        eligible, deferred = gate.filter_eligible_cues([target_cue], yaw_rate_dps=30.0)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(len(deferred), 0)

    def test_action_first_phrasing(self):
        """Test Action-First phrase synthesis format."""
        phraser = ActionFirstPhraser()
        cue = ActionCue(
            category="car",
            category_zh=localized_text("object.car"),
            confidence=0.9,
            bbox=(300, 100, 450, 350),
            distance_m=2.0,
            proximity_bin=ProximityBin.NEAR,
            clock_hour=1,
            clock_direction_zh=localized_text("clock.one"),
            clock_direction_en="1 o'clock",
            action_type="avoid",
            action_zh=localized_text("action.avoid"),
            action_en="Caution yield",
            urgency_flag=True,
        )

        phrase_zh = phraser.generate_phrase(cue, lang="zh")
        # Format: action, clock direction, proximity, and object.
        self.assertTrue(phrase_zh.startswith(localized_text("action.avoid")))
        self.assertIn(localized_text("clock.one"), phrase_zh)
        self.assertIn(localized_text("distance.two_meters"), phrase_zh)
        self.assertIn(localized_text("object.car"), phrase_zh)

        phrase_en = phraser.generate_phrase(cue, lang="en")
        self.assertTrue(phrase_en.startswith("Caution yield"))
        self.assertIn("1 o'clock", phrase_en)

    def test_navguide_pipeline_metrics(self):
        """Test end-to-end Algorithm 1 execution and Equation 3 metrics."""
        pipeline = NavGuidePipeline(condition=EvaluationCondition.P)

        # Create a batch of raw detections
        raw_detections = [
            DetectionCandidate(
                category="car", confidence=0.88, bbox=(200, 100, 350, 300), is_moving=True
            ),  # Hazard
            DetectionCandidate(
                category="car", confidence=0.60, bbox=(205, 105, 345, 295)
            ),  # Duplicate car
            DetectionCandidate(
                category="blindpath", confidence=0.85, bbox=(150, 200, 490, 450)
            ),  # Nav path
            DetectionCandidate(category="person", confidence=0.70, bbox=(500, 150, 560, 350)),
            DetectionCandidate(category="trash can", confidence=0.55, bbox=(50, 250, 100, 350)),
            DetectionCandidate(category="chair", confidence=0.50, bbox=(10, 100, 50, 200)),
        ]

        res = pipeline.process(
            detections=raw_detections,
            yaw_rate_dps=12.0,  # Below 25 deg/s
            frame_width=640,
            frame_height=480,
        )

        # Raw count = 6
        self.assertEqual(res.raw_item_count, 6)
        # Retained after Top-3 and deduplication <= 3
        self.assertLessEqual(res.retained_item_count, 3)
        # Item reduction R_item = 1 - N_keep / N_raw > 0
        self.assertGreater(res.item_reduction, 0.4)
        # Safety critical item (car) retained -> C_ret = 1.0
        self.assertGreater(res.conditional_retention, 0.0)
        # Spoken phrase generated
        self.assertTrue(res.should_speak)
        self.assertGreater(len(res.speech_text), 0)

        # Summary verification
        summary = pipeline.get_metrics_summary()
        self.assertEqual(summary["condition"], "P")
        self.assertIn("item_reduction_rate", summary)
        self.assertIn("conditional_retention_rate", summary)

    def test_conditions_comparison_p_vs_t_vs_b(self):
        """
        Test comparisons between NAVGUIDE (P), Tracking-only (T), and Broadcast (B)
        as described in Section 3.1 & Table 2 of the paper.
        """
        raw_detections = [
            DetectionCandidate(
                category="car", confidence=0.90, bbox=(100, 100, 250, 300), is_moving=True
            ),
            DetectionCandidate(category="person", confidence=0.85, bbox=(260, 150, 320, 350)),
            DetectionCandidate(category="bicycle", confidence=0.80, bbox=(350, 150, 420, 300)),
            DetectionCandidate(category="chair", confidence=0.60, bbox=(10, 50, 60, 150)),
            DetectionCandidate(category="trash can", confidence=0.50, bbox=(500, 200, 550, 300)),
            DetectionCandidate(category="pole", confidence=0.65, bbox=(580, 100, 620, 350)),
        ]

        pipe_p = NavGuidePipeline(condition=EvaluationCondition.P, top_k=3)
        pipe_t = NavGuidePipeline(condition=EvaluationCondition.T)
        pipe_b = NavGuidePipeline(condition=EvaluationCondition.B)

        res_p = pipe_p.process(raw_detections, yaw_rate_dps=5.0)
        res_t = pipe_t.process(raw_detections, yaw_rate_dps=5.0)
        res_b = pipe_b.process(raw_detections, yaw_rate_dps=5.0)

        # B forwards all valid detections: N_keep == N_raw
        self.assertEqual(res_b.retained_item_count, len(raw_detections))
        self.assertEqual(res_b.item_reduction, 0.0)

        # P enforces Top-3 content budget: N_keep <= 3
        self.assertLessEqual(res_p.retained_item_count, 3)
        self.assertGreater(res_p.item_reduction, 0.4)

        # Both P and T retain safety-critical hazards (car)
        self.assertGreater(res_p.conditional_retention, 0.0)
        self.assertGreater(res_t.conditional_retention, 0.0)

    def test_scanning_motion_gating_reduction(self):
        """
        Test Section 3.2: Controlled static-scene scanning.
        Scheduling with the gate suppresses nonurgent cues when |omega_t| >= 25 deg/s,
        while urgent warnings bypass the gate.
        """
        pipeline = NavGuidePipeline(condition=EvaluationCondition.P, yaw_rate_threshold_dps=25.0)

        # Ordinary nonurgent detections (e.g. static poles/chairs)
        nonurgent_detections = [
            DetectionCandidate(category="pole", confidence=0.75, bbox=(100, 100, 150, 300)),
            DetectionCandidate(category="chair", confidence=0.70, bbox=(200, 100, 260, 250)),
        ]

        # Case 1: Scanning / rapid turn at 35 deg/s
        res_scanning = pipeline.process(nonurgent_detections, yaw_rate_dps=35.0)
        # Rapid turns defer ordinary cues.
        self.assertEqual(len(res_scanning.eligible_cues), 0)
        self.assertGreater(len(res_scanning.deferred_cues), 0)
        self.assertFalse(res_scanning.should_speak)

        # Case 2: Scanning with an urgent safety hazard (approaching car)
        urgent_detections = [
            DetectionCandidate(
                category="car", confidence=0.92, bbox=(200, 100, 380, 320), is_moving=True
            ),
        ]
        res_urgent_scan = pipeline.process(urgent_detections, yaw_rate_dps=35.0)
        # Urgent hazards remain eligible above the motion threshold.
        self.assertEqual(len(res_urgent_scan.eligible_cues), 1)
        self.assertTrue(res_urgent_scan.should_speak)
        self.assertIn(localized_text("object.car"), res_urgent_scan.speech_text)

    def test_navigation_master_integration(self):
        """Test NavigationMaster initializes NAVGUIDE and syncs modes/IMU motion."""
        from navguide.navigation.coordinator import NavigationMaster

        class DummyNavigator:
            def __init__(self):
                self.last_detected_obstacles = []

            def reset(self):
                pass

            def process_frame(self, bgr):
                from dataclasses import dataclass

                @dataclass
                class DummyRes:
                    guidance_text = ""
                    annotated_image = bgr
                    state_info = {"crosswalk_stage": "not_detected", "state": "NAVIGATING"}

                return DummyRes()

        master = NavigationMaster(DummyNavigator(), DummyNavigator())
        self.assertIsNotNone(master.navguide)
        self.assertEqual(master.navguide.condition, EvaluationCondition.P)

        # Test IMU motion update
        master.update_imu_motion(18.5)
        self.assertEqual(master.navguide.latest_yaw_rate_dps, 18.5)

        # Test task mode switches
        master.start_blind_path_navigation()
        self.assertEqual(master.navguide.context.task_mode, TaskMode.PATH_NAVIGATION)
        self.assertEqual(master.navguide.context.scene_type, SceneType.SIDEWALK)

        master.start_item_search(target_query="cup")
        self.assertEqual(master.navguide.context.task_mode, TaskMode.TARGET_SEARCH)
        self.assertEqual(master.navguide.context.target_query, "cup")

        master.stop_navigation()
        self.assertEqual(master.navguide.context.task_mode, TaskMode.SCENE_EXPLORATION)


if __name__ == "__main__":
    unittest.main()
