"""Exercise path inference failures without loading camera, model or audio backends."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def path_module(monkeypatch):
    cv2 = SimpleNamespace(
        TERM_CRITERIA_EPS=2,
        TERM_CRITERIA_COUNT=1,
        COLOR_BGR2GRAY=6,
        cvtColor=lambda image, code: image[:, :, 0].copy(),
        bitwise_or=np.bitwise_or,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(
        sys.modules,
        "navguide.audio.player",
        SimpleNamespace(play_voice_text=Mock()),
    )
    module_name = "path_detection_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "navguide/navigation/path_following.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def model_returning(results):
    return SimpleNamespace(predict=Mock(return_value=results))


def segmentation_result(masks, classes, confidence=None):
    if confidence is None:
        confidence = [0.9] * len(classes)
    return SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array(classes), conf=np.array(confidence)),
    )


def suppress_rendering(navigator):
    navigator._draw_command_button = Mock(side_effect=lambda image, text: image)
    navigator._draw_visualizations = lambda image, visualizations: image
    navigator._add_mask_visualization = Mock()
    navigator._execute_state_machine = Mock(return_value="Continue")


@pytest.mark.parametrize("failure", ["model_unavailable", "inference_failed"])
def test_black_image_never_receives_a_fabricated_path(path_module, image, failure):
    model = None
    if failure == "inference_failed":
        model = SimpleNamespace(predict=Mock(side_effect=RuntimeError("Inference failed")))
    navigator = path_module.BlindPathNavigator(yolo_model=model)

    result = navigator._detect_path_and_crosswalk(image)

    assert result.available is False
    assert result.reason == failure
    assert result.blind_path_mask is None
    assert result.crosswalk_mask is None


@pytest.mark.parametrize(
    "results",
    [
        [],
        [SimpleNamespace(masks=None, boxes=None)],
        [SimpleNamespace(masks=None, boxes=SimpleNamespace(cls=np.array([])))],
        [segmentation_result(np.empty((0, 480, 640)), [])],
        [segmentation_result(np.zeros((1, 480, 640)), [1])],
    ],
)
def test_empty_success_is_distinct_from_unavailable(path_module, image, results):
    navigator = path_module.BlindPathNavigator(yolo_model=model_returning(results))
    suppress_rendering(navigator)

    detection = navigator._detect_path_and_crosswalk(image)
    processed = navigator.process_frame(image)

    assert detection.available is True
    assert detection.reason is None
    assert detection.blind_path_mask is None
    assert detection.crosswalk_mask is None
    assert processed.state_info["detection_available"] is True
    assert processed.state_info["detection_reason"] is None
    assert processed.guidance_text == ""
    navigator._execute_state_machine.assert_not_called()
    path_module.play_voice_text.assert_not_called()


@pytest.mark.parametrize(
    "results",
    [
        None,
        [SimpleNamespace(masks=None, boxes=SimpleNamespace(cls=np.array([1])))],
        [segmentation_result(np.ones((1, 480, 640)), [1, 0])],
        [segmentation_result(np.full((1, 480, 640), np.nan), [1])],
        [segmentation_result(np.ones((1, 480, 640)), [1], [np.nan])],
    ],
)
def test_invalid_outputs_do_not_become_empty_success(path_module, image, results):
    navigator = path_module.BlindPathNavigator(yolo_model=model_returning(results))

    detection = navigator._detect_path_and_crosswalk(image)

    assert detection.available is False
    assert detection.reason == "inference_failed"
    assert detection.blind_path_mask is None
    assert detection.crosswalk_mask is None


def test_real_masks_are_filtered_and_combined(path_module, image):
    masks = np.zeros((4, 480, 640), dtype=np.float32)
    masks[0, 10:30, 10:30] = 1
    masks[1, 30:50, 10:30] = 1
    masks[2, 350:400, :] = 1
    masks[3, :, :] = 1
    model = model_returning([segmentation_result(masks, [1, 1, 0, 1], [0.8, 0.9, 0.7, 0.1])])
    navigator = path_module.BlindPathNavigator(yolo_model=model)

    detection = navigator._detect_path_and_crosswalk(image)

    assert detection.available is True
    assert np.count_nonzero(detection.blind_path_mask) == 800
    assert np.count_nonzero(detection.crosswalk_mask) == 32000
    assert detection.blind_path_mask.dtype == np.uint8
    model.predict.assert_called_once_with(image, verbose=False, conf=0.2, classes=[0, 1])


def test_partial_output_is_discarded_after_inference_error(path_module, image):
    valid = segmentation_result(np.ones((1, 480, 640)), [1])
    invalid = SimpleNamespace(masks=None, boxes=SimpleNamespace(cls=np.array([0])))
    navigator = path_module.BlindPathNavigator(yolo_model=model_returning([valid, invalid]))

    result = navigator._detect_path_and_crosswalk(image)

    assert result.available is False
    assert result.blind_path_mask is None
    assert result.crosswalk_mask is None


@pytest.mark.parametrize("failure", ["model_unavailable", "inference_failed"])
def test_failure_clears_navigation_state_and_recovery_starts_fresh(path_module, image, failure):
    masks = np.zeros((1, 480, 640), dtype=np.float32)
    masks[0, 200:, 280:360] = 1
    model = model_returning([segmentation_result(masks, [1])])
    navigator = path_module.BlindPathNavigator(yolo_model=model)
    suppress_rendering(navigator)
    initial = navigator.process_frame(image)
    assert initial.state_info["detection_available"] is True
    assert navigator.prev_blind_path_mask is not None
    navigator._execute_state_machine.reset_mock()
    path_module.play_voice_text.reset_mock()

    navigator.current_state = path_module.STATE_MANEUVERING_TURN
    navigator.maneuver_target_info = {"direction": "left"}
    navigator.prev_crosswalk_mask = np.ones((480, 640), dtype=np.uint8)
    navigator.last_blindpath_mask = navigator.prev_blind_path_mask.copy()
    navigator.last_crosswalk_mask = navigator.prev_crosswalk_mask.copy()
    navigator.pending_crosswalk_voice = {"voice_text": "Cross now", "priority": 100}
    navigator.pending_obstacle_voice = "Obstacle ahead"
    navigator.last_detected_obstacles = [{"name": "car"}]
    navigator.flow_points = {"blind_path": [1]}
    navigator.centerline_history = [[1, 2]]
    navigator.crosswalk_monitor.in_arrival_state = True
    navigator.crosswalk_monitor.area_history.append({"area": 0.4})
    if failure == "inference_failed":
        model.predict.side_effect = RuntimeError("Inference failed")
    else:
        navigator.yolo_model = None

    failed = navigator.process_frame(image)

    assert failed.guidance_text == ""
    assert failed.visualizations == []
    assert failed.state_info["detection_available"] is False
    assert failed.state_info["detection_reason"] == failure
    assert failed.state_info["frame_count"] == 2
    assert navigator.current_state == path_module.STATE_ONBOARDING
    assert navigator.maneuver_target_info is None
    assert navigator.prev_blind_path_mask is None
    assert navigator.prev_crosswalk_mask is None
    assert navigator.last_blindpath_mask is None
    assert navigator.last_crosswalk_mask is None
    assert navigator.pending_crosswalk_voice is None
    assert navigator.pending_obstacle_voice is None
    assert not navigator.last_detected_obstacles
    assert not navigator.flow_points
    assert not navigator.centerline_history
    assert not navigator.crosswalk_monitor.in_arrival_state
    assert not navigator.crosswalk_monitor.area_history
    navigator._execute_state_machine.assert_not_called()
    path_module.play_voice_text.assert_not_called()
    assert navigator._draw_command_button.call_args.args[1] == path_module.localized_text(
        "path.detection_unavailable"
    )

    model.predict.side_effect = None
    navigator.yolo_model = model
    recovered = navigator.process_frame(image)

    assert recovered.state_info["detection_available"] is True
    assert recovered.state_info["detection_reason"] is None
    assert recovered.state_info["frame_count"] == 3
    navigator._execute_state_machine.assert_called_once()
    assert navigator.prev_blind_path_mask is not None


def test_current_empty_frame_drops_previous_masks(path_module, image):
    navigator = path_module.BlindPathNavigator(yolo_model=model_returning([]))
    suppress_rendering(navigator)
    navigator.prev_blind_path_mask = np.ones((480, 640), dtype=np.uint8)
    navigator.prev_crosswalk_mask = np.ones((480, 640), dtype=np.uint8)

    result = navigator.process_frame(image)

    assert result.state_info["detection_available"] is True
    assert navigator.prev_blind_path_mask is None
    assert navigator.prev_crosswalk_mask is None
    assert result.guidance_text == ""
    path_module.play_voice_text.assert_not_called()
