"""Test detector adaptation with synthetic model results, without GPU dependencies."""
import sys
import threading
from types import SimpleNamespace

import pytest

from navguide.runtime.vision import DEFAULT_CLASSES, YOLOEFrontend


class Array:
    def __init__(self, value):
        self.value = value
    def cpu(self):
        return self
    def tolist(self):
        return self.value


def fake_model(name, identity):
    boxes = SimpleNamespace(xyxy=Array([[10, 10, 50, 80]]), cls=Array([0]),
                            conf=Array([0.9]), id=Array([identity]))
    result = SimpleNamespace(boxes=boxes, names={0: name})
    return SimpleNamespace(track=lambda *args, **kwargs: [result])


def image_stubs(monkeypatch, dimensions=(640, 480)):
    class Header:
        format = "JPEG"
        size = dimensions
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=SimpleNamespace(open=lambda stream: Header())))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(uint8="uint8", frombuffer=lambda *args, **kwargs: b"pixels"))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        IMREAD_COLOR=1, imdecode=lambda *args: SimpleNamespace(shape=(480, 640, 3))))


def test_specialist_labels_and_track_namespaces_are_mapped_without_fake_motion(monkeypatch):
    image_stubs(monkeypatch)
    frontend = YOLOEFrontend.__new__(YOLOEFrontend)
    frontend._model_lock = threading.RLock()
    frontend.device = "cpu"
    frontend.model = fake_model("car", 1)
    frontend.specialists = [fake_model("red", 1), fake_model("red", 1)]
    frontend.specialist_kinds = ["BLIND_PATH_MODEL", "TRAFFIC_LIGHT_MODEL"]
    detections, width, height = frontend.detect(b"jpeg stub", 640 * 480)
    assert (width, height) == (640, 480)
    assert detections[0].category == "car"
    assert detections[0].is_moving is False and detections[0].urgency_override is False
    assert detections[1].category == "red" and detections[1].signal_color is None
    assert detections[2].category == "traffic light" and detections[2].signal_color == "red"
    assert detections[2].urgency_override is True
    assert len({item.track_id for item in detections}) == 3


def test_dimension_limit_rejects_before_native_decode(monkeypatch):
    image_stubs(monkeypatch, dimensions=(10000, 10000))
    frontend = YOLOEFrontend.__new__(YOLOEFrontend)
    frontend._model_lock = threading.RLock()
    with pytest.raises(ValueError, match="pixel limit"):
        frontend.detect(b"jpeg stub", 640 * 480)


def test_target_change_resets_tracking_only_when_vocabulary_changes():
    frontend = YOLOEFrontend.__new__(YOLOEFrontend)
    frontend._model_lock = threading.RLock()
    reset_calls = []
    classes_calls = []
    frontend.model = SimpleNamespace(
        get_text_pe=lambda classes: "embeddings",
        set_classes=lambda classes, embeddings: classes_calls.append(classes),
        predictor=SimpleNamespace(trackers=[SimpleNamespace(reset=lambda: reset_calls.append(True))]),
    )
    frontend.classes = list(DEFAULT_CLASSES)
    frontend.specialists = []
    frontend.set_target(None)
    assert not reset_calls
    frontend.set_target("cup")
    assert reset_calls == [True]
    assert classes_calls[-1][-1] == "cup"
    frontend.set_target("cup")
    assert len(reset_calls) == 1
