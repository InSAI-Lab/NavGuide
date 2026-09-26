"""Exercise real signal parsing and crossing transitions without model runtimes."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(monkeypatch, name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def traffic_module(monkeypatch):
    cv2 = SimpleNamespace(
        TERM_CRITERIA_EPS=2,
        TERM_CRITERIA_COUNT=1,
        COLOR_BGR2GRAY=6,
        cvtColor=lambda image, code: image[:, :, 0].copy(),
        rectangle=Mock(),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    import navguide.io

    frame_buffer = SimpleNamespace(wait_raw_bgr=Mock(), send_vis_bgr=Mock())
    monkeypatch.setattr(navguide.io, "frame_buffer", frame_buffer, raising=False)
    monkeypatch.setitem(sys.modules, "navguide.io.frame_buffer", frame_buffer)
    return load_module(
        monkeypatch, "traffic_lights_under_test", "navguide/perception/traffic_lights.py"
    )


@pytest.fixture
def crossing_module(monkeypatch, traffic_module):
    import navguide.perception

    monkeypatch.setattr(navguide.perception, "traffic_lights", traffic_module, raising=False)
    monkeypatch.setitem(sys.modules, "navguide.perception.traffic_lights", traffic_module)
    monkeypatch.setitem(
        sys.modules, "navguide.perception.obstacles", SimpleNamespace(ObstacleDetectorClient=None)
    )
    monkeypatch.setenv("NAVGUIDE_OBS_AUTO", "0")
    return load_module(
        monkeypatch, "street_crossing_under_test", "navguide/navigation/street_crossing.py"
    )


@pytest.fixture
def image():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def model_result(labels, confidences=None, names_as_list=False):
    confidences = confidences if confidences is not None else [0.9] * len(labels)
    boxes = [
        SimpleNamespace(
            cls=np.array([index]),
            conf=np.array([confidence]),
            xyxy=np.array([[10, 8, 20, 28]], dtype=float),
        )
        for index, confidence in enumerate(confidences)
    ]
    names = labels if names_as_list else dict(enumerate(labels))
    return SimpleNamespace(names=names, boxes=boxes)


def use_prediction(traffic_module, prediction):
    model = Mock(return_value=[prediction])
    model.names = prediction.names
    traffic_module._model = model
    return model


def waiting_navigator(crossing_module):
    navigator = crossing_module.CrossStreetNavigator()
    navigator.state = crossing_module.STATE_WAIT_LIGHT
    navigator._draw_command_button = lambda image, text: image
    navigator._draw_visualizations = lambda image, visualizations: image
    return navigator


@pytest.mark.parametrize(
    "label,color",
    [("stop", "red"), ("countdown_stop", "red"), ("go", "green"), ("countdown_go", "yellow")],
)
@pytest.mark.parametrize("names_as_list", [False, True])
def test_recognized_labels_and_colors_are_current_frame_data(
    traffic_module, image, label, color, names_as_list
):
    model = use_prediction(traffic_module, model_result([label], names_as_list=names_as_list))

    first = traffic_module.process_single_frame(image)
    second = traffic_module.process_single_frame(image)

    assert first["available"] is True
    assert first["reason"] is None
    assert first["detected_light"] == label
    assert first["detected_colors"] == [color]
    assert first["stable_light"] is None
    assert first["detections"][0]["bbox"] == [10, 8, 20, 28]
    assert second["stable_light"] == label
    assert model.call_args.args[0] is image
    assert not np.any(image)


@pytest.mark.parametrize(
    "labels", [[], ["crossing", "blank", "countdown_blank"], ["unknown_signal"]]
)
def test_no_signal_clears_previous_green_confirmation(traffic_module, image, labels):
    model = use_prediction(traffic_module, model_result(["go"]))
    traffic_module.process_single_frame(image)
    assert traffic_module.process_single_frame(image)["stable_light"] == "go"
    model.return_value = [model_result(labels)]

    missing = traffic_module.process_single_frame(image)

    assert missing["available"] is True
    assert missing["reason"] is None
    assert missing["detected_light"] is None
    assert missing["detected_colors"] == []
    assert missing["stable_light"] is None
    assert traffic_module._detection_history == []
    model.return_value = [model_result(["go"])]
    assert traffic_module.process_single_frame(image)["stable_light"] is None


@pytest.mark.parametrize("blocking_label", ["stop", "countdown_stop", "countdown_go"])
def test_restrictive_signal_blocks_higher_confidence_green(traffic_module, image, blocking_label):
    model = use_prediction(traffic_module, model_result(["go"]))
    traffic_module.process_single_frame(image)
    traffic_module.process_single_frame(image)
    model.return_value = [model_result(["go", blocking_label], [0.99, 0.5])]

    result = traffic_module.process_single_frame(image)

    assert result["detected_light"] == blocking_label
    assert result["stable_light"] is None
    assert "green" in result["detected_colors"]
    model.return_value = [model_result(["go"])]
    assert traffic_module.process_single_frame(image)["stable_light"] is None


def test_missing_weights_report_unavailable_without_importing_a_model(
    traffic_module, image, monkeypatch, tmp_path
):
    factory = Mock()
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    traffic_module.YOLO_MODEL_PATH = str(tmp_path / "missing.pt")
    traffic_module._detection_history = ["go", "go"]

    assert traffic_module.init_model() is False
    result = traffic_module.process_single_frame(image)

    factory.assert_not_called()
    assert traffic_module.get_model_status()["available"] is False
    assert result["available"] is False
    assert result["reason"] == "model_unavailable"
    assert "weights not found" in result["error"]
    assert result["stable_light"] is None
    assert traffic_module._detection_history == []


def test_model_load_failure_preserves_explicit_error(traffic_module, monkeypatch, tmp_path):
    weights = tmp_path / "signal.pt"
    weights.write_bytes(b"test fixture")
    traffic_module.YOLO_MODEL_PATH = str(weights)
    factory = Mock(side_effect=RuntimeError("Unsupported weights"))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))

    assert traffic_module.init_model() is False
    assert traffic_module.get_model_status() == {
        "available": False,
        "reason": "model_unavailable",
        "error": "Unsupported weights",
    }


def test_inference_failure_clears_votes_and_recovers_fresh(traffic_module, image):
    model = use_prediction(traffic_module, model_result(["go"]))
    traffic_module.process_single_frame(image)
    traffic_module.process_single_frame(image)
    model.side_effect = RuntimeError("Device disconnected")

    failed = traffic_module.process_single_frame(image)

    assert failed["available"] is False
    assert failed["reason"] == "inference_failed"
    assert failed["error"] == "Device disconnected"
    assert failed["stable_light"] is None
    assert failed["detected_colors"] == []
    assert traffic_module._detection_history == []
    model.side_effect = None
    assert traffic_module.process_single_frame(image)["stable_light"] is None


@pytest.mark.parametrize("prediction", [None, [SimpleNamespace(boxes=[object()])]])
def test_malformed_prediction_is_an_inference_failure(traffic_module, image, prediction):
    model = use_prediction(traffic_module, model_result(["go"]))
    model.return_value = prediction

    result = traffic_module.process_single_frame(image)

    assert result["available"] is False
    assert result["reason"] == "inference_failed"
    assert result["detected_light"] is None


def test_low_confidence_and_invalid_coordinates_cannot_confirm_green(traffic_module, image):
    result = model_result(["go", "go", "go"], [0.1, float("nan"), 0.9])
    result.boxes[2].xyxy[0, 0] = float("nan")
    use_prediction(traffic_module, result)

    detected = traffic_module.process_single_frame(image)

    assert detected["available"] is True
    assert detected["detections"] == []
    assert detected["stable_light"] is None


def test_missing_signal_module_keeps_crossing_wait_state(crossing_module, image, monkeypatch):
    navigator = waiting_navigator(crossing_module)
    navigator.green_light_counter = crossing_module.GREEN_LIGHT_STABLE_FRAMES - 1
    monkeypatch.setattr(crossing_module, "TRAFFIC_LIGHT_AVAILABLE", False)
    monkeypatch.setattr(crossing_module, "traffic_lights", None)

    result = navigator.process_frame(image)

    assert navigator.state == crossing_module.STATE_WAIT_LIGHT
    assert navigator.green_light_counter == 0
    assert result.guidance_text == crossing_module.localized_text("crossing.detector_unavailable")
    assert result.visualizations


def test_crossing_uses_raw_frames_and_requires_fresh_consecutive_confirmation(
    crossing_module, traffic_module, image
):
    navigator = waiting_navigator(crossing_module)
    model = use_prediction(traffic_module, model_result(["go"]))
    seen = []

    def predict(raw, **kwargs):
        seen.append(raw)
        return [model_result(["go"])]

    model.side_effect = predict
    for _ in range(crossing_module.GREEN_LIGHT_STABLE_FRAMES):
        navigator.process_frame(image)
        assert navigator.state == crossing_module.STATE_WAIT_LIGHT
    assert navigator.green_light_counter == crossing_module.GREEN_LIGHT_STABLE_FRAMES - 1
    navigator.process_frame(image)
    assert navigator.state == crossing_module.STATE_CROSSING
    assert all(frame is image for frame in seen)


@pytest.mark.parametrize("interruption", ["missing", "unknown", "failure", "red", "countdown"])
def test_crossing_discards_partial_green_confirmation(
    crossing_module, traffic_module, image, interruption
):
    navigator = waiting_navigator(crossing_module)
    model = use_prediction(traffic_module, model_result(["go"]))
    for _ in range(crossing_module.GREEN_LIGHT_STABLE_FRAMES):
        navigator.process_frame(image)
    assert navigator.green_light_counter == crossing_module.GREEN_LIGHT_STABLE_FRAMES - 1
    if interruption == "failure":
        model.side_effect = RuntimeError("Inference failed")
    else:
        labels = {
            "missing": [],
            "unknown": ["unknown"],
            "red": ["stop"],
            "countdown": ["countdown_go"],
        }[interruption]
        model.return_value = [model_result(labels)]

    navigator.process_frame(image)

    assert navigator.state == crossing_module.STATE_WAIT_LIGHT
    assert navigator.green_light_counter == 0
    model.side_effect = None
    model.return_value = [model_result(["go"])]
    navigator.process_frame(image)
    assert navigator.state == crossing_module.STATE_WAIT_LIGHT
    assert navigator.green_light_counter == 0


def test_stale_green_result_without_current_signal_does_not_advance(
    crossing_module, traffic_module, image, monkeypatch
):
    navigator = waiting_navigator(crossing_module)
    navigator.green_light_counter = crossing_module.GREEN_LIGHT_STABLE_FRAMES - 1
    monkeypatch.setattr(
        traffic_module,
        "process_single_frame",
        lambda raw: {
            "available": True,
            "detected_light": None,
            "stable_light": "go",
            "detections": [],
        },
    )

    navigator.process_frame(image)

    assert navigator.state == crossing_module.STATE_WAIT_LIGHT
    assert navigator.green_light_counter == 0


def test_configured_model_is_loaded_once_and_reports_ready(traffic_module, monkeypatch, tmp_path):
    weights = tmp_path / "signal.pt"
    weights.write_bytes(b"test fixture")
    traffic_module.YOLO_MODEL_PATH = str(weights)
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))

    assert traffic_module.init_model() is True
    assert traffic_module.init_model() is True

    factory.assert_called_once_with(str(weights))
    assert traffic_module.get_model_status() == {"available": True, "reason": None, "error": None}


@pytest.mark.parametrize("response", [None, {}, {"available": False, "stable_light": "go"}])
def test_missing_or_unavailable_response_resets_crossing_confirmation(
    crossing_module, traffic_module, image, monkeypatch, response
):
    navigator = waiting_navigator(crossing_module)
    navigator.green_light_counter = crossing_module.GREEN_LIGHT_STABLE_FRAMES - 1
    monkeypatch.setattr(traffic_module, "process_single_frame", lambda raw: response)

    navigator.process_frame(image)

    assert navigator.state == crossing_module.STATE_WAIT_LIGHT
    assert navigator.green_light_counter == 0


def test_unexpected_detector_exception_resets_both_confirmation_layers(
    crossing_module, traffic_module, image, monkeypatch
):
    navigator = waiting_navigator(crossing_module)
    navigator.green_light_counter = crossing_module.GREEN_LIGHT_STABLE_FRAMES - 1
    traffic_module._detection_history = ["go", "go"]
    monkeypatch.setattr(
        traffic_module, "process_single_frame", Mock(side_effect=RuntimeError("Backend failure"))
    )

    navigator.process_frame(image)

    assert navigator.state == crossing_module.STATE_WAIT_LIGHT
    assert navigator.green_light_counter == 0
    assert traffic_module._detection_history == []


def test_render_failure_preserves_current_model_results(traffic_module, image, monkeypatch):
    use_prediction(traffic_module, model_result(["stop"]))
    monkeypatch.setattr(
        traffic_module.cv2, "rectangle", Mock(side_effect=RuntimeError("Renderer failure"))
    )

    result = traffic_module.process_single_frame(image)

    assert result["available"] is True
    assert result["reason"] is None
    assert result["detected_light"] == "stop"
    assert np.array_equal(result["vis_image"], image)
