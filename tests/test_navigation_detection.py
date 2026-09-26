"""Detection failures must not release stale navigation or crossing cues."""

from types import SimpleNamespace

import numpy as np
import pytest

from navguide.navigation.coordinator import (
    BLINDPATH_NAV,
    CROSSING,
    RECOVERY,
    WAIT_TRAFFIC_LIGHT,
    NavigationMaster,
    TrafficLightDetector,
)


class SignalBackend:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.resets = 0

    def process_single_frame(self, image):
        if self.error:
            raise self.error
        return self.result

    def reset_detection_state(self):
        self.resets += 1


def signal_result(current=None, stable=None, available=True):
    return {
        "available": available,
        "reason": None if available else "model_unavailable",
        "detected_light": current,
        "stable_light": stable,
    }


@pytest.mark.parametrize("pixel", [(0, 0, 0), (0, 255, 0)])
@pytest.mark.parametrize(
    "result,error",
    [
        (signal_result(available=False), None),
        (signal_result(), None),
        (None, RuntimeError("Inference failed")),
        (None, None),
    ],
)
def test_signal_pixels_do_not_replace_model_evidence(pixel, result, error):
    image = np.full((48, 64, 3), pixel, dtype=np.uint8)
    detector = TrafficLightDetector(SignalBackend(result, error))
    color, meta = detector.detect(image)
    assert color == "unknown"
    assert meta["stable_color"] == "unknown"


@pytest.mark.parametrize(
    "current,stable,color,stable_color",
    [
        ("stop", "stop", "red", "red"),
        ("countdown_stop", "countdown_stop", "red", "red"),
        ("countdown_go", "countdown_go", "yellow", "yellow"),
        ("go", "go", "green", "green"),
        ("stop", "go", "red", "unknown"),
        (None, "go", "unknown", "unknown"),
        ("go", None, "green", "unknown"),
    ],
)
def test_current_signal_must_agree_with_stable_result(current, stable, color, stable_color):
    detector = TrafficLightDetector(SignalBackend(signal_result(current, stable)))
    actual, meta = detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))
    assert actual == color
    assert meta["stable_color"] == stable_color


class PathNavigator:
    def __init__(self, available):
        self.available = available
        self.last_detected_obstacles = []

    def reset(self):
        pass

    def process_frame(self, image):
        return SimpleNamespace(
            annotated_image=image,
            guidance_text="stale guidance" if not self.available else "",
            state_info={
                "state": "NAVIGATING" if self.available else "ONBOARDING",
                "crosswalk_stage": "not_detected" if self.available else "ready",
                "detection_available": self.available,
                "detection_reason": None if self.available else "inference_failed",
            },
        )


def test_unavailable_path_resets_transitions_and_suppresses_cached_guidance(monkeypatch):
    navigator = PathNavigator(False)
    master = NavigationMaster(navigator, navigator)
    master.state = BLINDPATH_NAV
    master.cnt_crosswalk_seen = master.FRAMES_CROSS_SEEN
    master.cnt_align_ready = master.FRAMES_ALIGN_READY
    master.tl_major.push("green")

    def reject_cached_cues(*args, **kwargs):
        pytest.fail("Unavailable segmentation must not replay cached cues")

    monkeypatch.setattr(master, "_collect_and_process_navguide_cues", reject_cached_cues)
    result = master.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    assert result.state == RECOVERY
    assert not result.guidance_text
    assert result.extras["detection_available"] is False
    assert result.extras["detection_reason"] == "inference_failed"
    assert master.cnt_crosswalk_seen == master.cnt_align_ready == 0
    assert master.tl_major.history() == []

    navigator.available = True
    monkeypatch.setattr(master, "_collect_and_process_navguide_cues", lambda *args, **kw: None)
    recovered = master.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    assert recovered.state == BLINDPATH_NAV
    assert recovered.extras["detection_available"] is True


@pytest.mark.parametrize(
    "result",
    [
        signal_result("go", "go", available=False),
        signal_result(None, "go"),
        signal_result("stop", "go"),
        signal_result("go", None),
    ],
)
def test_waiting_state_rejects_missing_or_stale_green(result):
    navigator = PathNavigator(True)
    master = NavigationMaster(navigator, navigator)
    master.state = WAIT_TRAFFIC_LIGHT
    master.tld = TrafficLightDetector(SignalBackend(result))
    for _ in range(8):
        master.tl_major.push("green")
    output = master.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    assert output.state == WAIT_TRAFFIC_LIGHT
    assert output.extras["traffic_light"] != "green"


def test_confirmed_current_green_allows_crossing():
    navigator = PathNavigator(True)
    master = NavigationMaster(navigator, navigator)
    master.state = WAIT_TRAFFIC_LIGHT
    master.tld = TrafficLightDetector(SignalBackend(signal_result("go", "go")))
    output = master.process_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    assert output.state == CROSSING


def test_new_navigation_session_clears_signal_history():
    navigator = PathNavigator(True)
    master = NavigationMaster(navigator, navigator)
    backend = SignalBackend(signal_result("go", "go"))
    master.tld = TrafficLightDetector(backend)
    master.tl_major.push("green")
    master.start_blind_path_navigation()
    assert backend.resets == 1
    assert master.tl_major.history() == []
