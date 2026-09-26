# -*- coding: utf-8 -*-
"""Crosswalk navigation with segmentation, optical flow and traffic signals."""

from navguide.i18n import text as localized_text
import torch
import os
from navguide.runtime.config import asset_path
import time
import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image, ImageDraw, ImageFont = None, None, None


try:
    from navguide.perception.obstacles import ObstacleDetectorClient
except Exception:
    ObstacleDetectorClient = None


try:
    from navguide.perception import traffic_lights

    TRAFFIC_LIGHT_AVAILABLE = True
except Exception:
    TRAFFIC_LIGHT_AVAILABLE = False
    traffic_lights = None

logger = logging.getLogger(__name__)


STATE_SEEKING = "SEEKING_CROSSWALK"
STATE_WAIT_LIGHT = "WAIT_TRAFFIC_LIGHT"
STATE_CROSSING = "CROSSING"


CROSSWALK_MIN_CONF = float(os.getenv("CROSSWALK_MIN_CONF", "0.3"))
CROSSWALK_MIN_AREA = int(os.getenv("CROSSWALK_MIN_AREA", "5000"))
BLIND_MIN_CONF = float(
    os.getenv("BLIND_MIN_CONF", "0.34")
)  # Minimum confidence for tactile paving detections.
ANGLE_THRESH_DEG = float(os.getenv("CROSSWALK_ANGLE_THRESH_DEG", "5.0"))
OFFSET_THRESH = float(os.getenv("CROSSWALK_OFFSET_THRESH", "0.08"))

# Alignment thresholds for distant crosswalks.
SEEKING_ANGLE_THRESH_DEG = 15.0
SEEKING_OFFSET_THRESH = 0.20

# Require all proximity thresholds before entering the signal-wait state.
CROSSWALK_NEAR_AREA_RATIO = 0.30
CROSSWALK_NEAR_BOTTOM_RATIO = 0.80
CROSSWALK_NEAR_MIN_HEIGHT_RATIO = 0.35


GREEN_LIGHT_STABLE_FRAMES = 5  # Consecutive green frames required for a stable signal.

# Class identifiers must match the model training labels.
CW_ID = int(os.getenv("NAVGUIDE_SEG_CW_ID", "0"))
BP_ID = int(os.getenv("NAVGUIDE_SEG_BP_ID", "1"))

# Aliases for crosswalk and tactile paving classes.
_CW = {"zebra_crossing", "zebra crossing", "zebra", "crosswalk", "road_crossing", "road crossing"}
_BP = {"blind_path", "tactile_paving", "tactile paving", "blind path"}


BP_VALID_IOU_THR = 0.40  # Reject tactile paving masks above this crosswalk overlap.


INNER_OFFSET_PX_LOCK = 5
EDGE_DILATE_PX = 2
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
)
FEATURE_PARAMS = dict(maxCorners=600, qualityLevel=0.001, minDistance=5, blockSize=7)


MASK_EMA_ALPHA = 0.6  # Exponential moving average weight.
TRACK_MIN_POINTS = 30  # Minimum tracked feature count.
TRACK_RESEED_EVERY = 12  # Reseed features after this many frames with successful segmentation.

# Visualization colors use OpenCV BGR order.
VIS_COLORS = {
    "crosswalk": (0, 165, 255),
    "centerline": (255, 255, 0),
    "target_point": (255, 0, 255),
    "hint": (0, 255, 255),
    "stripes": (0, 128, 255),
    "heading": (0, 0, 255),
}


@dataclass
class CrossStreetResult:
    """Crossing guidance, annotations and state information."""

    annotated_image: Optional[np.ndarray] = None
    guidance_text: str = ""
    visualizations: List[Dict[str, Any]] = None
    should_switch_to_blindpath: bool = False

    def __post_init__(self):
        if self.visualizations is None:
            self.visualizations = []


def _score_of(d) -> float:
    """Read detection confidence, defaulting to zero when unavailable."""
    for k in ("conf", "confidence", "score", "prob"):
        v = getattr(d, k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                break
    return 0.0


def _norm_name(s: str) -> str:
    """Normalize a detector category name."""
    return str(s).lower().replace("_", " ").strip()


def _in_set(name: str, pool: set) -> bool:
    """Match a normalized category against its aliases."""
    return _norm_name(name) in {_norm_name(x) for x in pool}


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the intersection over union of two masks."""
    if a is None or b is None:
        return 0.0
    ai = a > 0
    bi = b > 0
    inter = np.logical_and(ai, bi).sum()
    union = np.logical_or(ai, bi).sum()
    return float(inter) / float(union + 1e-6)


def _looks_like_blind_path(bp_mask: np.ndarray, cw_mask: np.ndarray, H: int, W: int) -> bool:
    """Reject stripe and curb masks using shape and crosswalk overlap."""
    if bp_mask is None:
        return False
    ys, xs = np.where(bp_mask > 0)
    if xs.size < 80:
        return False

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    angle_deg = np.degrees(np.arctan2(v[1], v[0]))
    if angle_deg > 90:
        angle_deg -= 180
    if angle_deg < -90:
        angle_deg += 180

    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    aspect = h / float(w + 1e-6)  # Tactile paving is expected to have a near-vertical major axis.
    iou_cw = _mask_iou(bp_mask, cw_mask)

    # Reject horizontal stripes.
    if abs(angle_deg) <= 20.0:
        return False
    # Reject overly wide shapes.
    if aspect < 0.52:
        return False
    # Reject masks that overlap the crosswalk excessively.
    if iou_cw >= BP_VALID_IOU_THR:
        return False
    # Reject narrow strips near the lower image boundary.
    bottom = bp_mask[int(0.88 * H) :, :]
    if bottom.sum() > 0:
        bottom_share = bottom.sum() / float((bp_mask > 0).sum() + 1e-6)
        if bottom_share > 0.50 and (w / float(W)) < 0.35:
            return False
    return True


def _cls_of(d):
    """Read the detector class identifier."""
    for k in ("cls", "class_id", "category_id"):
        v = getattr(d, k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return None


class CrossStreetNavigator:
    """Coordinate crosswalk alignment, signal waiting and crossing guidance."""

    def __init__(self, seg_model=None, coco_model=None, obs_model=None, device_id: str = "esp32"):
        self.seg_model = seg_model
        self.device_id = device_id
        self.frame_counter = 0
        self.last_guidance = ""
        self.crosswalk_detected = False
        self.last_guide_time = 0
        self.guide_interval = 3.0  # Speech interval in seconds.

        self.state = STATE_SEEKING
        self.green_light_counter = 0
        self.last_traffic_light = None
        self.last_seeking_guidance = ""  # Throttle repeated alignment instructions.
        self.last_waiting_light_time = 0
        self.crossing_end_announced = False
        self.last_crosswalk_seen_time = 0
        self.last_blindpath_announce_time = 0  # Throttle repeated tactile paving announcements.

        self.prev_mask = None
        self.prev_mask_float = None
        self.prev_mask_ts = 0.0
        self.old_gray = None
        self.p0 = None
        self.last_seed_frame = 0

        self.obstacle_detector = obs_model
        self.prev_gray = None
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0
        self.OBSTACLE_DETECTION_INTERVAL = int(os.getenv("NAVGUIDE_OBS_INTERVAL", "15"))
        self.OBSTACLE_CACHE_DURATION_FRAMES = int(os.getenv("NAVGUIDE_OBS_CACHE_FRAMES", "0"))

        self.CROSSWALK_DETECTION_INTERVAL = int(os.getenv("NAVGUIDE_CROSSWALK_INTERVAL", "4"))
        self.last_crosswalk_detection_frame = 0
        self.last_detected_crosswalk_mask = None
        self.last_detected_blindpath_mask = None

        # Load an obstacle detector when none is supplied.
        if self.obstacle_detector is None and os.getenv("NAVGUIDE_OBS_AUTO", "1") != "0":
            try:
                if ObstacleDetectorClient is not None:
                    model_path = asset_path("NAVGUIDE_OBS_MODEL", "model/yoloe-11l-seg.pt")
                    self.obstacle_detector = ObstacleDetectorClient(model_path)
                    logger.info("[CROSS_STREET] Obstacle detector loaded automatically")
                else:
                    logger.warning(
                        "[CROSS_STREET] ObstacleDetectorClient unavailable; skipping automatic loading"
                    )
            except Exception as e:
                logger.warning(f"[CROSS_STREET] Automatic obstacle detector loading failed: {e}")

        # Adapt models that expose predict but not detect.
        if (
            self.seg_model
            and hasattr(self.seg_model, "predict")
            and not hasattr(self.seg_model, "detect")
        ):
            logger.info("[CROSS_STREET] Adapting YOLO model")
            self.seg_model = YOLOModelWrapper(self.seg_model)

        logger.info(
            f"[CROSS_STREET] Crosswalk detection interval={self.CROSSWALK_DETECTION_INTERVAL} frames"
        )

        if self.seg_model and torch.cuda.is_available():
            try:
                if hasattr(self.seg_model, "model") and hasattr(self.seg_model.model, "to"):
                    self.seg_model.model.to("cuda")
                elif hasattr(self.seg_model, "to"):
                    self.seg_model.to("cuda")
                logger.info("[CROSS_STREET] Model moved to GPU")
            except Exception as e:
                logger.warning(f"[CROSS_STREET] Could not move model to GPU: {e}")

    def reset(self):
        """Reset navigation, tracking and speech state."""
        self.frame_counter = 0
        self.last_guidance = ""
        self.crosswalk_detected = False
        self.last_guide_time = 0

        self.state = STATE_SEEKING
        self.green_light_counter = 0
        self.last_traffic_light = None
        self.last_seeking_guidance = ""
        self.last_waiting_light_time = 0
        self.crossing_end_announced = False
        self.last_crosswalk_seen_time = 0
        self.last_blindpath_announce_time = 0

        self.prev_mask = None
        self.prev_mask_float = None
        self.prev_mask_ts = 0.0
        self.old_gray = None
        self.p0 = None
        self.last_seed_frame = 0

        self.prev_gray = None
        self.last_detected_obstacles = []
        self.last_obstacle_detection_frame = 0

        if TRAFFIC_LIGHT_AVAILABLE and traffic_lights:
            traffic_lights.reset_detection_state()
        logger.info("[CROSS_STREET] Navigator reset")

    @staticmethod
    def _inner_offset_edge(mask_bin: np.ndarray, offset_px=5, edge_dilate_px=2) -> np.ndarray:
        """Extract an inset mask boundary for optical flow features."""
        if offset_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * offset_px + 1, 2 * offset_px + 1))
            eroded = cv2.erode(mask_bin.astype(np.uint8), k, iterations=1)
        else:
            eroded = mask_bin.astype(np.uint8)
        edges = cv2.Canny(eroded * 255, 50, 150)
        if edge_dilate_px > 0:
            k2 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * edge_dilate_px + 1, 2 * edge_dilate_px + 1)
            )
            edges = cv2.dilate(edges, k2, iterations=1)
        return edges  # uint8 0/255

    @staticmethod
    def _hull_mask_from_points(points: np.ndarray, shape_hw: tuple) -> Optional[np.ndarray]:
        """Rasterize the convex hull of a set of points."""
        if points is None or len(points) < 3:
            return None
        H, W = shape_hw
        pts = points.reshape(-1, 2).astype(np.float32)
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        poly = hull.reshape(-1, 2).astype(np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        return mask

    def _seed_points_from_mask(
        self, gray: np.ndarray, mask_bin: np.ndarray
    ) -> Optional[np.ndarray]:
        """Seed Lucas-Kanade features on the inset mask boundary."""
        edge_mask = self._inner_offset_edge(
            mask_bin, offset_px=INNER_OFFSET_PX_LOCK, edge_dilate_px=EDGE_DILATE_PX
        )
        try:
            pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)
            return pts
        except Exception as e:
            logger.warning(f"[CROSS_STREET] goodFeaturesToTrack failed: {e}")
            return None

    @staticmethod
    def _ensure_binary_mask(mask: np.ndarray, shape_hw: tuple) -> np.ndarray:
        """Resize and threshold a mask to a uint8 array containing zero or one."""
        H, W = shape_hw
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)

    def _postprocess_mask(self, mask_bin: np.ndarray) -> np.ndarray:
        """Clean mask boundaries and remove small connected components."""
        try:
            m = (mask_bin > 0).astype(np.uint8)
            H, W = m.shape[:2]
            # Apply opening and closing before removing small components.
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                keep_area = max(
                    int(0.003 * H * W), 1500
                )  # Minimum area is 0.3 percent of the frame or 1500 pixels.
                keep_labels = np.where(areas >= keep_area)[0] + 1
                m2 = np.zeros_like(m)
                for lbl in keep_labels:
                    m2[labels == lbl] = 1
                if m2.sum() > 0:
                    m = m2
            return (m > 0).astype(np.uint8)
        except Exception:
            return (mask_bin > 0).astype(np.uint8)

    @staticmethod
    def _largest_contour(mask_bin: np.ndarray):
        cts, _ = cv2.findContours(
            (mask_bin > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cts:
            return None
        return max(cts, key=cv2.contourArea)

    def _mask_center(self, mask: np.ndarray):
        """Return the mask centroid from image moments, or None if empty."""
        M = cv2.moments((mask > 0).astype(np.uint8))
        if abs(M["m00"]) < 1e-6:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _is_crosswalk_near(self, mask: np.ndarray, h: int, w: int) -> bool:
        """Require sufficient mask area, height and bottom position for proximity."""
        if mask is None:
            return False
        area = int(mask.sum())
        area_ratio = float(area) / float(h * w)

        ys = np.where(mask > 0)[0]
        if ys.size == 0:
            return False
        top_y = int(ys.min())
        bottom_y = int(ys.max())
        mask_height = bottom_y - top_y + 1
        height_ratio = float(mask_height) / float(h)
        bottom_ratio = float(bottom_y) / float(h)

        # Require area, bottom position and relative height together.

        is_near = (
            area_ratio >= CROSSWALK_NEAR_AREA_RATIO
            and bottom_ratio >= CROSSWALK_NEAR_BOTTOM_RATIO
            and height_ratio >= CROSSWALK_NEAR_MIN_HEIGHT_RATIO
        )
        return is_near

    def _is_crosswalk_almost_done(self, mask: np.ndarray, h: int, w: int) -> bool:
        """Check whether a small crosswalk mask lies at the bottom of the frame."""
        if mask is None:
            return False
        area = int(mask.sum())
        area_ratio = float(area) / float(h * w)

        ys = np.where(mask > 0)[0]
        if ys.size == 0:
            return False

        top_y = int(ys.min())
        bottom_y = int(ys.max())

        top_ratio = float(top_y) / float(h)
        bottom_ratio = float(bottom_y) / float(h)

        # A nearly completed crossing has a small mask concentrated at the bottom.

        is_almost_done = top_ratio > 0.7 and bottom_ratio > 0.85 and area_ratio < 0.08
        return is_almost_done

    def _compute_far_distance_alignment(self, mask: np.ndarray, h: int, w: int) -> tuple:
        """Estimate distant crosswalk angle and offset using mask geometry."""
        ys, xs = np.where(mask > 0)
        if xs.size < 50:
            return 0.0, 0.0

        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        mean = pts.mean(axis=0)
        cov = np.cov((pts - mean).T)
        eigvals, eigvecs = np.linalg.eig(cov)
        v = eigvecs[:, np.argmax(eigvals)]

        angle = np.degrees(np.arctan2(v[1], v[0]))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180

        cx = float(mean[0])
        offset = (cx - (w / 2.0)) / max(1.0, w / 2.0)

        return float(angle), float(offset)

    def _draw_line_vertical_angle(
        self, image, center, angle_deg, length_ratio=0.7, color=(255, 255, 0), thickness=3
    ):
        """Draw a line through center, with zero vertical and positive angles left."""
        H, W = image.shape[:2]
        half_len = int(0.5 * length_ratio * min(H, W))
        rad = np.radians(angle_deg)
        # The upward vertical reference is (0, -1).
        # Its rotated direction is (sin(angle), -cos(angle)).
        vx = np.sin(rad)
        vy = -np.cos(rad)
        x0, y0 = center
        p1 = (int(x0 - vx * half_len), int(y0 - vy * half_len))
        p2 = (int(x0 + vx * half_len), int(y0 + vy * half_len))
        cv2.line(image, p1, p2, color, thickness)

    def _draw_dashed_line_vertical_angle(
        self,
        image,
        center,
        angle_deg,
        length_ratio=0.7,
        dash=12,
        gap=8,
        color=(255, 255, 255),
        thickness=2,
    ):
        """Draw a dashed line through center using vertical angle coordinates."""
        H, W = image.shape[:2]
        half_len = int(0.5 * length_ratio * min(H, W))
        rad = np.radians(angle_deg)
        vx = np.sin(rad)
        vy = -np.cos(rad)
        x0, y0 = center
        x1, y1 = int(x0 - vx * half_len), int(y0 - vy * half_len)
        x2, y2 = int(x0 + vx * half_len), int(y0 + vy * half_len)

        total_len = int(np.hypot(x2 - x1, y2 - y1))
        if total_len <= 0:
            return
        dx = (x2 - x1) / total_len
        dy = (y2 - y1) / total_len
        s = 0
        while s < total_len:
            e = min(s + dash, total_len)
            xa, ya = int(x1 + dx * s), int(y1 + dy * s)
            xb, yb = int(x1 + dx * e), int(y1 + dy * e)
            cv2.line(image, (xa, ya), (xb, yb), color, thickness)
            s += dash + gap

    def _offset_from_centerline(
        self, center_pt, angle_vertical_deg, width, height, y_ratio=0.75
    ) -> float:
        """Return normalized horizontal offset at the lookahead row.

        The angle uses zero for vertical, matching _draw_line_vertical_angle.
        Positive offsets lie to the right of the image center."""
        if center_pt is None:
            return 0.0
        x0, y0 = center_pt
        rad = np.radians(angle_vertical_deg)
        # Use the same angle convention as _draw_line_vertical_angle.
        vx = np.sin(rad)
        vy = -np.cos(rad)

        # Evaluate the centerline at the lookahead row.
        y_target = float(int(height * y_ratio))

        # Avoid division by zero for nearly horizontal lines.
        if abs(vy) < 1e-6:
            x_at = float(x0)
        else:
            t = (y_target - float(y0)) / vy
            x_at = float(x0) + t * vx

        x_at = float(np.clip(x_at, 0, width - 1))
        # Normalize horizontal offset; right is positive.
        return float((x_at - (width / 2.0)) / max(1.0, width / 2.0))

    def _compute_angle_and_offset(self, mask: np.ndarray) -> tuple:
        """Estimate crosswalk angle and offset using PCA as a fallback."""
        H, W = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        if xs.size < 50:
            return 0.0, 0.0

        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        mean = pts.mean(axis=0)
        cov = np.cov((pts - mean).T)
        eigvals, eigvecs = np.linalg.eig(cov)
        v = eigvecs[:, np.argmax(eigvals)]

        angle = np.degrees(np.arctan2(v[1], v[0]))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180

        cx = float(mean[0])
        offset = (cx - (W / 2.0)) / max(1.0, W / 2.0)

        return float(angle), float(offset)

    def _estimate_angle_by_stripes(
        self, mask: np.ndarray, gray: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Estimate a stripe normal using weighted Hough lines and robust clustering.

        Return angle_deg, selected lines, confidence in [0, 1] and line count.
        Angles are relative to vertical; positive values point left."""
        try:
            H, W = mask.shape[:2]
            roi_top = int(
                0.45 * H
            )  # Use the lower half of the mask for a more stable stripe estimate.
            m_roi = (mask[roi_top:H, :] > 0).astype(np.uint8)
            g_roi = gray[roi_top:H, :]

            g_blur = cv2.GaussianBlur(g_roi, (5, 5), 0)
            edges = cv2.Canny(g_blur, 50, 150)
            edges = cv2.bitwise_and(edges, edges, mask=m_roi * 255)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=max(30, int(0.03 * W)),
                minLineLength=int(0.15 * W),
                maxLineGap=20,
            )
            if lines is None:
                return None

            angles, weights = [], []
            all_lines = []
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx, dy = x2 - x1, y2 - y1
                length = float(np.hypot(dx, dy))
                if length < 8:
                    continue
                ang = float(np.degrees(np.arctan2(dy, dx)))
                if ang > 90:
                    ang -= 180
                if ang < -90:
                    ang += 180

                if abs(ang) > 65:
                    continue
                # Weight stripes near the lower image boundary more heavily.
                ymid = (y1 + y2) * 0.5 + roi_top
                w = length * (0.5 + 0.5 * (ymid / max(1.0, H)))
                angles.append(ang)
                weights.append(w)
                all_lines.append((int(x1), int(y1 + roi_top), int(x2), int(y2 + roi_top)))

            if len(angles) < 5:
                return None

            # Reject outliers using a weighted median and median absolute deviation.
            angs = np.array(angles, dtype=np.float32)
            wts = np.array(weights, dtype=np.float32)

            sort_idx = np.argsort(angs)
            angs_sorted = angs[sort_idx]
            wts_sorted = wts[sort_idx]
            cum = np.cumsum(wts_sorted)
            med_idx = np.searchsorted(cum, cum[-1] * 0.5)
            med = float(angs_sorted[min(max(med_idx, 0), len(angs_sorted) - 1)])

            dev = np.abs(angs - med)
            mad = float(np.median(dev) + 1e-6)
            deg_thr = max(12.0, 2.8 * mad)
            keep = dev <= deg_thr

            if keep.sum() >= 3:
                angs_keep = angs[keep]
                wts_keep = wts[keep]
                lines_keep = [all_lines[i] for i, k in enumerate(keep) if k]
            else:
                angs_keep = angs
                wts_keep = wts
                lines_keep = all_lines

            # Compute a weighted circular mean of the surviving angles.
            ang_rad = np.radians(angs_keep)
            C = float(np.sum(wts_keep * np.cos(ang_rad)))
            S = float(np.sum(wts_keep * np.sin(ang_rad)))
            norm = float(np.sum(wts_keep) + 1e-6)
            if abs(C) < 1e-6 and abs(S) < 1e-6:
                return None
            mean = float(np.degrees(np.arctan2(S, C)))
            confidence = float(np.hypot(C, S) / norm)

            return {
                "angle_deg": mean,
                "lines": lines_keep,
                "confidence": confidence,
                "count": len(lines_keep),
            }
        except Exception:
            return None

    def _get_crosswalk_guidance_features(self, mask: np.ndarray, image_shape: tuple) -> dict:
        """Fit a robust centerline and derive target, angle and offset."""
        try:
            height, width = image_shape[:2]
            min_run_px = max(12, int(width * 0.02))
            centerline_rows = []

            # Scan upward and use the largest contiguous span at each row.
            for y in range(height - 1, int(height * 0.4), -5):
                row = mask[y, :]
                xs = np.where(row > 0)[0]
                if xs.size <= min_run_px:
                    continue
                splits = np.where(np.diff(xs) > 1)[0] + 1
                segments = np.split(xs, splits) if xs.size else []
                if not segments:
                    continue
                seg = max(segments, key=lambda s: (s[-1] - s[0] + 1))
                if seg.size == 0 or (seg[-1] - seg[0] + 1) < min_run_px:
                    continue
                center_x = 0.5 * (seg[0] + seg[-1])
                centerline_rows.append([y, center_x])

            if len(centerline_rows) < 10:
                return None

            data = np.array(centerline_rows, dtype=np.float32)
            y_coords, x_coords = data[:, 0], data[:, 1]

            # Weight lower rows more heavily.
            w_base = y_coords / float(height)
            coeffs = np.polyfit(y_coords, x_coords, 2, w=w_base)
            poly = np.poly1d(coeffs)

            # Apply one robust reweighting pass to reduce bends and outliers.
            res = x_coords - poly(y_coords)
            mad = np.median(np.abs(res - np.median(res))) + 1e-6
            c = 2.5 * mad
            w_robust = 1.0 / (1.0 + (res / c) ** 2)
            w_total = w_base * w_robust
            coeffs = np.polyfit(y_coords, x_coords, 2, w=w_total)
            poly = np.poly1d(coeffs)

            lookahead_y = int(height * 0.6)
            target_x = float(poly(lookahead_y))
            plot_y = np.arange(int(height * 0.4), height, 5).astype(int)
            plot_x = poly(plot_y).astype(int)
            centerline_points = np.vstack((plot_x, plot_y)).T.tolist()

            # Derive heading from the derivative of x(y).
            dpoly = np.polyder(poly)
            dx_dy = float(dpoly(lookahead_y))
            angle_deg = float(np.degrees(np.arctan(dx_dy)))
            offset = float((target_x - (width / 2.0)) / max(1.0, width / 2.0))

            tx = int(np.clip(target_x, 0, width - 1))
            return {
                "target_point": (tx, lookahead_y),
                "centerline_points": centerline_points,
                "angle_deg": angle_deg,
                "offset": offset,
            }
        except Exception:
            return None

    def _get_edge_mask(self, mask, offset=10):
        """Extract an inner mask boundary for optical flow feature selection."""
        if mask is None:
            return None
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset * 2, offset * 2))
        inner = cv2.erode(mask, kernel, iterations=1)
        edge = cv2.subtract(mask, inner)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge = cv2.dilate(edge, kernel_small, iterations=1)
        return edge

    def _predict_mask_with_flow(self, prev_mask, prev_gray, curr_gray):
        """Predict the mask using Lucas-Kanade optical flow and affine fallback."""
        try:
            edge_mask = self._get_edge_mask(prev_mask, offset=10)
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=edge_mask, **FEATURE_PARAMS)
            if p0 is None or len(p0) < 8:
                return None
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **LK_PARAMS)
            if p1 is None or st is None:
                return None
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            if len(good_new) < 5:
                return None
            M, inliers = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=5.0
            )
            if M is None:
                return None
            H, W = curr_gray.shape[:2]
            flow_mask = cv2.warpAffine(
                prev_mask,
                M,
                (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return flow_mask
        except Exception:
            return None

    def _detect_obstacles(self, image, path_mask=None):
        """Detect obstacles and populate the fields required by navigation."""
        logger.info(
            f"[_detect_obstacles] Starting frame={self.frame_counter}, obstacle_detector={('loaded' if self.obstacle_detector else 'unavailable')}"
        )
        if self.obstacle_detector is None:
            logger.warning("[_detect_obstacles] Obstacle detector unavailable")
            return []

        try:
            logger.info(
                f"[_detect_obstacles] Calling ObstacleDetectorClient.detect(); image.shape={image.shape}"
            )
            detected_obstacles = self.obstacle_detector.detect(image, path_mask=path_mask)
            logger.info(f"[_detect_obstacles] Returned {len(detected_obstacles)} objects")

            H, W = image.shape[:2]
            for i, obj in enumerate(detected_obstacles):
                if "mask" in obj and obj["mask"] is not None:
                    y_coords, x_coords = np.where(obj["mask"] > 0)
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                        x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                        obj["box_coords"] = (x1, y1, x2, y2)
                        if "y_position_ratio" not in obj:
                            obj["y_position_ratio"] = obj.get("center_y", 0) / H
                        if "label" not in obj:
                            obj["label"] = obj.get("name", "unknown")
                        if "center" not in obj:
                            obj["center"] = (obj.get("center_x", 0), obj.get("center_y", 0))
                        if "confidence" not in obj:
                            obj["confidence"] = 0.5
            return detected_obstacles
        except Exception as e:
            logger.error(f"[_detect_obstacles] Detection failed: {e}", exc_info=True)
            return []

    def _stabilize_obstacle_list(
        self, obstacles, prev_obstacles, prev_gray, curr_gray, image_shape, threshold=0.5
    ):
        """Associate obstacles across frames and smooth their masks."""
        if not obstacles or prev_gray is None or curr_gray is None:
            return obstacles

        H, W = image_shape
        stabilized = []
        used_prev = set()
        for curr_obs in obstacles:
            if "mask" not in curr_obs or curr_obs["mask"] is None:
                stabilized.append(curr_obs)
                continue
            curr_mask = curr_obs["mask"]
            best_match = None
            best_iou = 0
            best_idx = -1

            if prev_obstacles:
                for idx, prev_obs in enumerate(prev_obstacles):
                    if idx in used_prev or "mask" not in prev_obs:
                        continue
                    flow_mask = self._predict_mask_with_flow(prev_obs["mask"], prev_gray, curr_gray)
                    if flow_mask is None:
                        flow_mask = prev_obs["mask"]
                    inter = np.logical_and(curr_mask > 0, flow_mask > 0).sum()
                    union = np.logical_or(curr_mask > 0, flow_mask > 0).sum()
                    iou = float(inter) / float(union) if union > 0 else 0.0
                    if iou > best_iou and iou > threshold:
                        best_iou = iou
                        best_match = flow_mask
                        best_idx = idx

            if best_match is not None and best_idx >= 0:
                used_prev.add(best_idx)
                fused_mask = ((0.8 * curr_mask + 0.2 * best_match) > 128).astype(np.uint8) * 255
                curr_obs["mask"] = fused_mask
                self._update_obstacle_properties(curr_obs, H, W)
            stabilized.append(curr_obs)
        return stabilized

    def _update_obstacle_properties(self, obs, H, W):
        """Update obstacle geometry derived from its mask."""
        if "mask" not in obs or obs["mask"] is None:
            return
        mask = obs["mask"]
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            obs["area"] = int(len(y_coords))
            obs["center_x"] = float(np.mean(x_coords))
            obs["center_y"] = float(np.mean(y_coords))
            obs["y_position_ratio"] = obs["center_y"] / H
            obs["area_ratio"] = obs["area"] / float(H * W)
            obs["bottom_y_ratio"] = np.max(y_coords) / float(H)
            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
            obs["box_coords"] = (x1, y1, x2, y2)

    def _parse_color(self, color_str):
        """Parse a color string into OpenCV BGR components."""
        try:
            if isinstance(color_str, tuple) and len(color_str) == 3:
                return color_str
            if color_str.startswith("rgba("):
                values = color_str[5:-1].split(",")
                r, g, b = int(values[0]), int(values[1]), int(values[2])
                return (b, g, r)  # OpenCV: BGR
            elif color_str == "yellow":
                return (0, 255, 255)
            elif color_str == "red":
                return (0, 0, 255)
            else:
                return (0, 0, 255)
        except:
            return (0, 0, 255)

    def _add_obstacle_visualization(self, obstacle, visualizations, pulse_effect=False):
        """Draw red contours for nearby obstacles and yellow contours otherwise."""
        try:
            bottom_y_ratio = obstacle.get("bottom_y_ratio", 0)
            area_ratio = obstacle.get("area_ratio", 0)
            is_near = bottom_y_ratio > 0.7 or area_ratio > 0.1

            if "mask" in obstacle and obstacle["mask"] is not None:
                mask = obstacle["mask"]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_contour = max(contours, key=cv2.contourArea)
                    points = max_contour.squeeze(1)[::5].tolist()

                    # Nearby contours are red; other contours are yellow.
                    if is_near:
                        outline_color = "rgba(255, 0, 0, 1.0)"
                        thickness = 3
                    else:
                        outline_color = "rgba(255, 255, 0, 0.8)"
                        thickness = 2

                    visualizations.append(
                        {
                            "type": "outline",
                            "points": points,
                            "color": outline_color,
                            "thickness": thickness,
                        }
                    )
        except Exception as e:
            logger.error(f"[_add_obstacle_visualization] Drawing failed: {e}")

    def _draw_command_button(self, image, text):
        """Draw the current guidance text at the bottom center."""
        try:
            H, W = image.shape[:2]
            full_text = localized_text("ui.current_instruction").format(
                value_1=text if text else localized_text("ui.unavailable")
            )

            font_px = 14
            pad_x, pad_y = 14, 8
            bottom_margin = 28

            if PIL_AVAILABLE:
                try:
                    from PIL import Image as PILImage, ImageDraw, ImageFont

                    font = None
                    for font_path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
                        if os.path.exists(font_path):
                            try:
                                font = ImageFont.truetype(font_path, font_px)
                                break
                            except:
                                continue
                    if font:
                        bbox = ImageDraw.Draw(PILImage.new("RGB", (1, 1))).textbbox(
                            (0, 0), full_text, font=font
                        )
                        tw = max(1, bbox[2] - bbox[0])
                        th = max(1, bbox[3] - bbox[1])
                    else:
                        scale = font_px / 24.0
                        (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                except:
                    scale = font_px / 24.0
                    (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            else:
                scale = font_px / 24.0
                (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)

            bw = tw + pad_x * 2
            bh = th + pad_y * 2
            radius = max(10, bh // 2)

            cx = W // 2
            left = max(8, cx - bw // 2)
            top = H - bottom_margin - bh
            right = min(W - 8, left + bw)
            bottom = top + bh

            overlay = image.copy()
            bg_color = (26, 32, 41)
            border_color = (60, 76, 102)

            cv2.rectangle(overlay, (left + radius, top), (right - radius, bottom), bg_color, -1)
            cv2.circle(overlay, (left + radius, (top + bottom) // 2), radius, bg_color, -1)
            cv2.circle(overlay, (right - radius, (top + bottom) // 2), radius, bg_color, -1)

            cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

            cv2.rectangle(image, (left + radius, top), (right - radius, bottom), border_color, 1)
            cv2.circle(image, (left + radius, (top + bottom) // 2), radius, border_color, 1)
            cv2.circle(image, (right - radius, (top + bottom) // 2), radius, border_color, 1)

            text_x = left + pad_x
            text_y = top + pad_y + th

            if PIL_AVAILABLE and font:

                pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((text_x, top + pad_y), full_text, font=font, fill=(255, 255, 255))
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:

                cv2.putText(
                    image,
                    full_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (255, 255, 255),
                    1,
                )

            return image
        except Exception as e:
            logger.error(f"Command button rendering failed: {e}")
            return image

    def _draw_data_panel_no_bg(self, image, data, position=(15, 15)):
        """Draw an outlined text panel without a background."""
        if not PIL_AVAILABLE:
            return image
        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, "RGBA")
            env_scale = float(os.getenv("NAVGUIDE_PANEL_SCALE", "0.7"))
            base_font_size = max(10, int(round(14 * env_scale)))
            font = None
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/System/Library/Fonts/PingFang.ttc",
            ]
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, base_font_size)
                        break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()

            y_offset = position[1]
            for key, value in data.items():
                text = f"{key}: {value}"
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text(
                                (position[0] + dx, y_offset + dy),
                                text,
                                font=font,
                                fill=(0, 0, 0, 255),
                            )
                draw.text((position[0], y_offset), text, font=font, fill=(255, 255, 255, 255))
                y_offset += base_font_size + 5
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"Data panel rendering failed: {e}")
            return image

    def _draw_visualizations(self, image, viz_elements):
        """Render translucent masks, contours, guidance and diagnostic overlays."""
        if not viz_elements:
            return image
        current_time = time.time()
        panel_elements = [v for v in viz_elements if v.get("type") == "data_panel"]
        standard_elements = [v for v in viz_elements if v.get("type") != "data_panel"]

        # Render translucent fills first.
        for element in standard_elements:
            elem_type = element.get("type")
            if elem_type in ["blind_path_mask", "obstacle_mask", "crosswalk_mask"]:
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 0.5)"))
                    if element.get("effect") == "pulse":
                        pulse_speed = element.get("pulse_speed", 1.0)
                        alpha = 0.3 + 0.3 * np.sin(current_time * pulse_speed * 2 * np.pi)
                    else:
                        alpha = 0.4
                    x, y, w, h = cv2.boundingRect(points)
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    if w > 0 and h > 0:
                        binary_mask = np.zeros((h, w), dtype=np.uint8)
                        local_points = points - np.array([x, y])
                        cv2.fillPoly(binary_mask, [local_points], 255)
                        local_region = image[y : y + h, x : x + w].copy()
                        color_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                        color_overlay[:] = color
                        for c in range(3):
                            local_region[:, :, c] = np.where(
                                binary_mask > 0,
                                (1 - alpha) * local_region[:, :, c]
                                + alpha * color_overlay[:, :, c],
                                local_region[:, :, c],
                            )
                        image[y : y + h, x : x + w] = local_region

        # Render outlines and labels over the fills.
        for element in standard_elements:
            elem_type = element.get("type")
            if elem_type == "outline":
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                    thickness = element.get("thickness", 3)
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
            elif elem_type == "polyline":
                points = np.array(element.get("points", []), dtype=np.int32)
                if points.size > 0:
                    color = self._parse_color(element.get("color", "rgba(255, 255, 0, 1.0)"))
                    thickness = element.get("width", 2)
                    cv2.polylines(image, [points], isClosed=False, color=color, thickness=thickness)
            elif elem_type == "circle":
                center = tuple(element.get("center", (0, 0)))
                radius = element.get("radius", 10)
                color = self._parse_color(element.get("color", "rgba(255, 0, 0, 1.0)"))
                thickness = -1 if element.get("filled", True) else 2
                cv2.circle(image, center, radius, color, thickness)
            elif elem_type == "arrow":
                start = tuple(element.get("start", (0, 0)))
                end = tuple(element.get("end", (100, 100)))
                color = self._parse_color(element.get("color", "rgba(0, 255, 255, 1.0)"))
                thickness = element.get("thickness", 2)
                tip_length = element.get("tip_length", 0.3)
                cv2.arrowedLine(image, start, end, color, thickness, tipLength=tip_length)
            elif elem_type == "text_with_bg":
                text = element.get("text", "")
                pos = element.get("position", [10, 30])
                font_scale = element.get("font_scale", 0.6)
                color = self._parse_color(element.get("color", "rgba(255, 255, 255, 1.0)"))
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            cv2.putText(
                                image,
                                text,
                                (pos[0] + dx, pos[1] + dy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                (0, 0, 0),
                                3,
                            )
                cv2.putText(image, text, tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            elif elem_type == "warning_icon":
                pos = element.get("position", (100, 100))
                level = element.get("level", "info")
                text = element.get("text", "")
                flash = element.get("flash", False)
                if level == "danger":
                    icon_color = (0, 0, 255)
                    text_color = (255, 255, 255)
                elif level == "warning":
                    icon_color = (0, 165, 255)
                    text_color = (255, 255, 255)
                else:
                    icon_color = (0, 255, 255)
                    text_color = (0, 0, 0)
                if flash:
                    alpha = 0.5 + 0.5 * np.sin(current_time * 4 * np.pi)
                    icon_color = tuple(int(c * alpha) for c in icon_color)
                triangle = np.array(
                    [[pos[0], pos[1] - 20], [pos[0] - 15, pos[1]], [pos[0] + 15, pos[1]]], np.int32
                )
                cv2.fillPoly(image, [triangle], icon_color)
                cv2.polylines(image, [triangle], True, (255, 255, 255), 2)
                cv2.putText(
                    image,
                    "!",
                    (pos[0] - 5, pos[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                if text:
                    font_scale = 0.5
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    text_pos = (pos[0] - tw // 2, pos[1] + 20)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                cv2.putText(
                                    image,
                                    text,
                                    (text_pos[0] + dx, text_pos[1] + dy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale,
                                    (0, 0, 0),
                                    2,
                                )
                    cv2.putText(
                        image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1
                    )
            elif elem_type == "text":
                text = element.get("text", "")
                pos = tuple(element.get("pos", (10, 30)))
                cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if PIL_AVAILABLE:
            for panel in panel_elements:
                image = self._draw_data_panel_no_bg(image, panel["data"], panel["position"])
        else:
            for panel in panel_elements:
                y_offset = panel["position"][1]
                for key, value in panel["data"].items():
                    text = f"{key}: {value}"
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                cv2.putText(
                                    image,
                                    text,
                                    (panel["position"][0] + dx, y_offset + dy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 0, 0),
                                    3,
                                )
                    cv2.putText(
                        image,
                        text,
                        (panel["position"][0], y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )
                    y_offset += 25
        return image

    def _speech_for_obstacle(self, name: str) -> str:
        """Build a localized obstacle warning."""
        k = (name or "").strip().lower()
        if k == "person":
            return localized_text("obstacle.person_ahead")
        if k == "car":
            return localized_text("obstacle.car_ahead")
        if k == "bicycle":
            return localized_text("obstacle.bicycle_ahead")
        if k == "motorcycle":
            return localized_text("obstacle.motorcycle_ahead")
        if k == "bus":
            return localized_text("obstacle.bus_ahead")
        if k == "truck":
            return localized_text("obstacle.truck_ahead")
        if k == "scooter":
            return localized_text("obstacle.scooter_ahead")
        if k == "stroller":
            return localized_text("obstacle.stroller_ahead")
        if k == "dog":
            return localized_text("obstacle.dog_ahead")
        if k == "animal":
            return localized_text("obstacle.animal_ahead")
        return localized_text("obstacle.ahead")

    def _read_traffic_signal(self, bgr_image, annotated) -> dict:
        """Read only current raw-frame signals and clear interrupted confirmation."""
        result = {"available": False, "detected_light": None, "stable_light": None}
        if TRAFFIC_LIGHT_AVAILABLE and traffic_lights:
            try:
                response = traffic_lights.process_single_frame(bgr_image)
                if isinstance(response, dict):
                    result = response
                if result.get("available"):
                    try:
                        traffic_lights.draw_detections(annotated, result.get("detections", []))
                    except Exception as error:
                        logger.warning("[CROSS_STREET] Signal visualization failed: %s", error)
            except Exception as error:
                logger.warning("[CROSS_STREET] Signal detection failed: %s", error)
                traffic_lights.reset_detection_state()
                result = {"available": False, "detected_light": None, "stable_light": None}

        current = result.get("detected_light") if result.get("available") else None
        stable = result.get("stable_light") if result.get("available") else None
        self.last_traffic_light = current
        if current != "go" or stable != "go":
            self.green_light_counter = 0
        return result

    def process_frame(self, bgr_image: np.ndarray) -> CrossStreetResult:
        """Segment a frame and use optical flow when current segmentation is unavailable."""
        self.frame_counter += 1
        current_time = time.time()

        try:
            annotated = bgr_image.copy()
            h, w = bgr_image.shape[:2]
            frame_visualizations = []

            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

            # Run segmentation at the configured frame interval.
            crosswalk_mask = None
            blindpath_mask = None
            det_area = 0

            if self.seg_model and self.frame_counter % self.CROSSWALK_DETECTION_INTERVAL == 0:

                base_thr = min(CROSSWALK_MIN_CONF, BLIND_MIN_CONF)
                detections = self.seg_model.detect(bgr_image, confidence_threshold=base_thr) or []

                # Accept matching class identifiers or category aliases.
                raw_cw, raw_bp = [], []
                for det in detections:
                    if not hasattr(det, "mask") or det.mask is None:
                        continue

                    cid = _cls_of(det)
                    name = str(getattr(det, "name", "")).lower()

                    if (cid == CW_ID) or _in_set(name, _CW):
                        raw_cw.append(det)

                    elif (cid == BP_ID) or _in_set(name, _BP):
                        raw_bp.append(det)

                cw_list = [d for d in raw_cw if _score_of(d) >= CROSSWALK_MIN_CONF]
                bp_list = [d for d in raw_bp if _score_of(d) >= BLIND_MIN_CONF]

                if cw_list:
                    cw_masks = []
                    for det in cw_list:
                        mask = det.mask
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_bin = (mask > 0.5).astype(np.uint8)
                        cw_masks.append(mask_bin)
                    if cw_masks:
                        crosswalk_mask = np.maximum.reduce(cw_masks)
                        det_area = int(crosswalk_mask.sum())
                        if det_area < CROSSWALK_MIN_AREA:
                            crosswalk_mask = None
                            det_area = 0

                if bp_list:
                    bp_masks = []
                    for det in bp_list:
                        mask = det.mask
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_bin = (mask > 0.5).astype(np.uint8)
                        bp_masks.append(mask_bin)
                    if bp_masks:
                        blindpath_mask = np.maximum.reduce(bp_masks)

                # Remove tactile paving regions from the crosswalk mask.
                if crosswalk_mask is not None and blindpath_mask is not None:
                    crosswalk_mask = crosswalk_mask.copy()
                    crosswalk_mask[blindpath_mask > 0] = 0

                if blindpath_mask is not None:
                    if not _looks_like_blind_path(blindpath_mask, crosswalk_mask, h, w):
                        blindpath_mask = None

                self.last_detected_crosswalk_mask = crosswalk_mask
                self.last_detected_blindpath_mask = blindpath_mask
                self.last_crosswalk_detection_frame = self.frame_counter

            else:

                crosswalk_mask = self.last_detected_crosswalk_mask
                blindpath_mask = self.last_detected_blindpath_mask

            # Recover missing segmentation from tracked features.
            used_tracking = False
            if crosswalk_mask is None:
                if (
                    self.old_gray is not None
                    and self.p0 is not None
                    and len(self.p0) >= TRACK_MIN_POINTS
                ):
                    try:
                        p1, st, err = cv2.calcOpticalFlowPyrLK(
                            self.old_gray, gray, self.p0, None, **LK_PARAMS
                        )
                        if p1 is not None and st is not None:
                            good_new = p1[st == 1]
                            if good_new is not None and len(good_new) >= TRACK_MIN_POINTS:
                                tracked_mask = self._hull_mask_from_points(good_new, (h, w))
                                if tracked_mask is not None and int(tracked_mask.sum()) >= (
                                    0.3
                                    * (self.prev_mask.sum() if self.prev_mask is not None else 1)
                                ):
                                    crosswalk_mask = tracked_mask
                                    used_tracking = True
                                    self.p0 = good_new.reshape(-1, 1, 2)
                                else:
                                    self.p0 = None
                                    self.old_gray = None
                    except Exception as e:
                        logger.warning(f"[CROSS_STREET] LK optical flow failed: {e}")
                        self.p0 = None
                        self.old_gray = None

            # Smooth the mask with an exponential average before cleanup.
            if crosswalk_mask is not None:
                m = crosswalk_mask.astype(np.float32)
                if self.prev_mask_float is not None and self.prev_mask_float.shape == m.shape:
                    self.prev_mask_float = (
                        MASK_EMA_ALPHA * m + (1.0 - MASK_EMA_ALPHA) * self.prev_mask_float
                    )
                else:
                    self.prev_mask_float = m
                crosswalk_mask = (self.prev_mask_float > 0.5).astype(np.uint8)
                crosswalk_mask = self._postprocess_mask(crosswalk_mask)
                self.prev_mask = crosswalk_mask
                self.prev_mask_ts = current_time

            # Reseed features after successful segmentation or tracking.
            if crosswalk_mask is not None:
                need_seed = (
                    self.p0 is None
                    or len(self.p0) < TRACK_MIN_POINTS
                    or (self.frame_counter - self.last_seed_frame) >= TRACK_RESEED_EVERY
                )
                if need_seed:
                    pts = self._seed_points_from_mask(gray, crosswalk_mask)
                    if pts is not None and len(pts) >= TRACK_MIN_POINTS:
                        self.p0 = pts
                        self.old_gray = gray.copy()
                        self.last_seed_frame = self.frame_counter
                else:
                    self.old_gray = gray.copy()
            else:
                self.crosswalk_detected = False
                self.p0 = None
                self.old_gray = None

            # Restrict obstacle detection to the crosswalk when available.
            detected_obstacles = []
            if self.obstacle_detector is not None:
                if self.frame_counter % self.OBSTACLE_DETECTION_INTERVAL == 0:
                    detected_obstacles = self._detect_obstacles(bgr_image, path_mask=crosswalk_mask)

                    if self.prev_gray is not None:
                        detected_obstacles = self._stabilize_obstacle_list(
                            detected_obstacles,
                            self.last_detected_obstacles,
                            self.prev_gray,
                            gray,
                            bgr_image.shape[:2],
                        )
                    self.last_detected_obstacles = detected_obstacles
                    self.last_obstacle_detection_frame = self.frame_counter
                else:
                    if (
                        self.frame_counter - self.last_obstacle_detection_frame
                        < self.OBSTACLE_CACHE_DURATION_FRAMES
                    ):
                        detected_obstacles = self.last_detected_obstacles
                    else:
                        detected_obstacles = []

                for obs in detected_obstacles:
                    self._add_obstacle_visualization(obs, frame_visualizations)

            guidance_text = ""

            if blindpath_mask is not None:
                # Blend colors only inside each mask.
                mask_area = (blindpath_mask > 0).astype(bool)
                green_color = np.array([0, 255, 0], dtype=np.float32)  # BGR

                for c in range(3):
                    annotated[:, :, c] = np.where(
                        mask_area,
                        (annotated[:, :, c] * 0.7 + green_color[c] * 0.3).astype(np.uint8),
                        annotated[:, :, c],
                    )

                bp_ct = self._largest_contour(blindpath_mask)
                if bp_ct is not None:
                    cv2.drawContours(annotated, [bp_ct], -1, (0, 255, 0), 2)

            if crosswalk_mask is not None:
                self.crosswalk_detected = True

                mask_area = (crosswalk_mask > 0).astype(bool)
                orange_color = np.array([0, 165, 255], dtype=np.float32)  # BGR

                for c in range(3):
                    annotated[:, :, c] = np.where(
                        mask_area,
                        (annotated[:, :, c] * 0.7 + orange_color[c] * 0.3).astype(np.uint8),
                        annotated[:, :, c],
                    )

            if self.state == STATE_SEEKING:
                # Seek and align with a distant crosswalk.
                if crosswalk_mask is not None:
                    is_near = self._is_crosswalk_near(crosswalk_mask, h, w)

                    if is_near:

                        self.state = STATE_WAIT_LIGHT
                        self.green_light_counter = 0
                        if TRAFFIC_LIGHT_AVAILABLE and traffic_lights:
                            traffic_lights.reset_detection_state()
                        guidance_text = localized_text("crossing.arrived_check_signal")
                        self.last_seeking_guidance = ""
                    else:

                        angle, offset = self._compute_far_distance_alignment(crosswalk_mask, h, w)

                        # Correct angle before lateral offset.
                        if abs(angle) >= SEEKING_ANGLE_THRESH_DEG:
                            direction = (
                                localized_text("crossing.turn_left_slightly")
                                if angle > 0
                                else localized_text("crossing.turn_right_slightly")
                            )
                        elif abs(offset) >= SEEKING_OFFSET_THRESH:
                            direction = (
                                localized_text("crossing.sidestep_right")
                                if offset > 0
                                else localized_text("crossing.sidestep_left")
                            )
                        else:
                            direction = localized_text("path.keep_straight")

                        frame_visualizations.append(
                            {
                                "type": "data_panel",
                                "data": {
                                    localized_text("ui.status"): localized_text(
                                        "crossing.align_crosswalk"
                                    ),
                                    localized_text("ui.angle"): f"{angle:.1f}°",
                                    localized_text("ui.offset"): f"{offset:.2f}",
                                },
                                "position": (w - 180, 20),
                            }
                        )

                        # Speak when alignment changes or the repeat interval elapses.
                        if current_time - self.last_guide_time > self.guide_interval:
                            if direction != self.last_seeking_guidance:
                                guidance_text = direction
                                self.last_seeking_guidance = direction
                            elif current_time - self.last_guide_time > self.guide_interval * 2:

                                guidance_text = direction
                else:

                    frame_visualizations.append(
                        {
                            "type": "data_panel",
                            "data": {
                                localized_text("ui.status"): localized_text("crossing.searching")
                            },
                            "position": (w - 180, 20),
                        }
                    )
                    self.last_seeking_guidance = ""

            elif self.state == STATE_WAIT_LIGHT:
                signal = self._read_traffic_signal(bgr_image, annotated)
                available = signal.get("available", False)
                current_light = signal.get("detected_light")
                stable_light = signal.get("stable_light")
                if available and current_light == stable_light == "go":
                    self.green_light_counter += 1
                    detection_text = localized_text("crossing.green_confirmation").format(
                        green_light_counter=self.green_light_counter,
                        GREEN_LIGHT_STABLE_FRAMES=GREEN_LIGHT_STABLE_FRAMES,
                    )
                    if self.green_light_counter >= GREEN_LIGHT_STABLE_FRAMES:
                        self.state = STATE_CROSSING
                        guidance_text = localized_text("crossing.green_stable_proceed")
                        self.green_light_counter = 0
                        self.crossing_end_announced = False
                        self.last_crosswalk_seen_time = current_time
                        self.last_blindpath_announce_time = 0
                elif not available:
                    detection_text = localized_text("crossing.detector_unavailable")
                elif current_light in ("stop", "countdown_stop"):
                    detection_text = localized_text("crossing.red_wait")
                else:
                    detection_text = localized_text("crossing.waiting_for_green")

                frame_visualizations.append(
                    {
                        "type": "data_panel",
                        "data": {
                            localized_text("ui.status"): localized_text("crossing.checking_signal"),
                            localized_text("ui.detection"): detection_text,
                        },
                        "position": (w - 180, 20),
                    }
                )
                if (
                    self.state == STATE_WAIT_LIGHT
                    and current_time - self.last_waiting_light_time > 3.0
                ):
                    guidance_text = localized_text(
                        "crossing.waiting_for_green"
                        if available
                        else "crossing.detector_unavailable"
                    )
                    self.last_waiting_light_time = current_time

            elif self.state == STATE_CROSSING:
                signal = self._read_traffic_signal(bgr_image, annotated)
                traffic_light_warning = None
                if (
                    signal.get("available")
                    and signal.get("detected_light") == signal.get("stable_light") == "countdown_go"
                    and current_time - self.last_guide_time > 2.0
                ):
                    traffic_light_warning = localized_text("crossing.green_ending")

                if crosswalk_mask is not None:

                    self.last_crosswalk_seen_time = current_time

                    # Resume crossing if the crosswalk reappears after an end announcement.
                    area = int(crosswalk_mask.sum())
                    area_ratio = float(area) / float(h * w)

                    if area_ratio > 0.1 and self.crossing_end_announced:
                        self.crossing_end_announced = False
                        self.blindpath_announced = False
                        logger.info(
                            "[CROSS_STREET] Crosswalk detected; clearing completion flag and resuming guidance"
                        )

                    panel_data = {
                        localized_text("ui.status"): localized_text("crossing.in_progress"),
                        localized_text("ui.area"): f"{area_ratio:.2f}",
                    }
                    if self.crossing_end_announced:
                        panel_data[localized_text("ui.hint")] = localized_text(
                            "crossing.completion_announced"
                        )
                    frame_visualizations.append(
                        {"type": "data_panel", "data": panel_data, "position": (w - 180, 20)}
                    )

                    # Use the stripe normal through the centroid to calculate lateral offset.
                    angle_deg, offset = 0.0, 0.0

                    # Prefer Hough stripe orientation, with PCA as a fallback.
                    angle_source = localized_text("crossing.stripes")
                    stripes = self._estimate_angle_by_stripes(crosswalk_mask, gray)
                    if stripes and ("angle_deg" in stripes):
                        angle_deg = -float(stripes["angle_deg"])
                        for x1, y1, x2, y2 in stripes.get("lines", []):
                            cv2.line(annotated, (x1, y1), (x2, y2), VIS_COLORS["stripes"], 2)

                        cx, cy = int(w * 0.5), int(h * 0.85)
                        length = int(60)
                        rad = np.radians(angle_deg)
                        dx = int(length * np.sin(rad))
                        dy = int(length * np.cos(rad))
                        cv2.arrowedLine(
                            annotated,
                            (cx, cy),
                            (cx + dx, cy - dy),
                            VIS_COLORS["heading"],
                            3,
                            tipLength=0.25,
                        )
                    else:
                        angle_source = "PCA"
                        angle_deg, _ = self._compute_angle_and_offset(crosswalk_mask)

                    center_pt = self._mask_center(crosswalk_mask)
                    if center_pt is not None and stripes and ("angle_deg" in stripes):
                        # Treat the stripe mean as the normal angle relative to vertical.
                        angle_blue = float(
                            stripes["angle_deg"]
                        )  # Keep the angle sign and do not add a 90-degree rotation.
                        self._draw_line_vertical_angle(
                            annotated,
                            center_pt,
                            angle_blue,
                            length_ratio=0.7,
                            color=VIS_COLORS["centerline"],
                            thickness=3,
                        )

                        # The dashed vertical line represents the assumed walking direction.
                        angle_white = 0.0
                        self._draw_dashed_line_vertical_angle(
                            annotated,
                            center_pt,
                            angle_white,
                            length_ratio=0.7,
                            dash=12,
                            gap=8,
                            color=(255, 255, 255),
                            thickness=2,
                        )

                        diff = angle_blue - 0.0  # = angle_blue
                        diff = (diff + 180.0) % 360.0 - 180.0
                        cv2.putText(
                            annotated,
                            f"{abs(diff):.1f}°",
                            (min(center_pt[0] + 12, w - 110), max(center_pt[1] - 12, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                        try:
                            # Offset and drawing helpers share the vertical-zero angle convention.
                            offset_new = self._offset_from_centerline(
                                center_pt, angle_blue, w, h, y_ratio=0.75
                            )
                            offset = float(offset_new)
                        except Exception:

                            pass

                    if abs(angle_deg) >= ANGLE_THRESH_DEG:
                        direction = (
                            localized_text("crossing.turn_left_slightly")
                            if angle_deg > 0
                            else localized_text("crossing.turn_right_slightly")
                        )
                    elif abs(offset) >= OFFSET_THRESH:
                        direction = (
                            localized_text("crossing.sidestep_right")
                            if offset > 0
                            else localized_text("crossing.sidestep_left")
                        )
                    else:
                        direction = localized_text("path.keep_straight")

                    # Nearby obstacles override direction guidance.
                    obstacle_override = None
                    if detected_obstacles:
                        NEAR_Y = 0.7
                        NEAR_AREA = 0.1
                        near_list = [
                            o
                            for o in detected_obstacles
                            if (
                                o.get("bottom_y_ratio", 0) > NEAR_Y
                                or o.get("area_ratio", 0) > NEAR_AREA
                            )
                        ]
                        if near_list:
                            name = near_list[0].get("name") or localized_text("object.obstacle")
                            obstacle_override = self._speech_for_obstacle(name)

                    src_text = (
                        localized_text("crossing.segmentation")
                        if not used_tracking
                        else localized_text("crossing.tracking")
                    )

                    if current_time - self.last_guide_time > self.guide_interval:

                        is_almost_done = self._is_crosswalk_almost_done(crosswalk_mask, h, w)

                        if self.frame_counter % 30 == 0:
                            ys = np.where(crosswalk_mask > 0)[0]
                            if ys.size > 0:
                                top_y, bottom_y = int(ys.min()), int(ys.max())
                                logger.info(
                                    f"[CROSS_STREET] area_ratio={area_ratio:.3f}, top_ratio={top_y/h:.3f}, bottom_ratio={bottom_y/h:.3f}, almost_done={is_almost_done}"
                                )

                        # Speech priority: signal warning, crossing end, tactile paving, obstacle, direction.
                        if traffic_light_warning:
                            guidance_text = traffic_light_warning
                            self.last_guide_time = current_time

                        elif is_almost_done and not self.crossing_end_announced:
                            guidance_text = localized_text("crossing.completed")
                            self.crossing_end_announced = True
                            self.last_guide_time = current_time

                        elif self.crossing_end_announced and blindpath_mask is not None:
                            if current_time - self.last_blindpath_announce_time > 4.0:
                                guidance_text = localized_text("crossing.paving_ahead")
                                self.last_blindpath_announce_time = current_time
                                self.last_guide_time = current_time

                        elif obstacle_override:
                            guidance_text = obstacle_override
                            self.last_guide_time = current_time

                        else:
                            guidance_text = direction
                            self.last_guide_time = current_time
                else:

                    no_crosswalk_duration = current_time - self.last_crosswalk_seen_time

                    frame_visualizations.append(
                        {
                            "type": "data_panel",
                            "data": {
                                localized_text("ui.status"): localized_text("crossing.in_progress"),
                                localized_text("ui.crosswalk"): localized_text(
                                    "crossing.not_detected_duration"
                                ).format(no_crosswalk_duration=f"{no_crosswalk_duration:.1f}"),
                            },
                            "position": (w - 180, 20),
                        }
                    )

                    # Require ten seconds without a crosswalk before announcing completion.
                    if no_crosswalk_duration > 10.0:
                        if not self.crossing_end_announced:
                            if current_time - self.last_guide_time > self.guide_interval:

                                if traffic_light_warning:
                                    guidance_text = traffic_light_warning
                                    self.last_guide_time = current_time

                                else:
                                    guidance_text = localized_text("crossing.completed")
                                    self.crossing_end_announced = True
                                    self.last_guide_time = current_time
                        # Repeat tactile paving guidance at a four-second interval after completion.
                        elif blindpath_mask is not None:
                            if current_time - self.last_blindpath_announce_time > 4.0:
                                guidance_text = localized_text("crossing.paving_ahead")
                                self.last_blindpath_announce_time = current_time
                                self.last_guide_time = current_time

            if guidance_text:
                current_instruction = guidance_text
            elif self.state == STATE_SEEKING:
                current_instruction = (
                    self.last_seeking_guidance
                    if self.last_seeking_guidance
                    else localized_text("crossing.searching_status")
                )
            elif self.state == STATE_WAIT_LIGHT:
                current_instruction = localized_text("crossing.waiting_green_status")
            elif self.state == STATE_CROSSING:
                current_instruction = localized_text("crossing.crossing_status")
            else:
                current_instruction = localized_text("ui.waiting")
            annotated = self._draw_command_button(annotated, current_instruction)

            if frame_visualizations:
                annotated = self._draw_visualizations(annotated, frame_visualizations)

            # The application owns playback; return guidance text without playing it here.

            # Save the grayscale frame for obstacle tracking.
            self.prev_gray = gray

            return CrossStreetResult(
                annotated_image=annotated,
                guidance_text=guidance_text,
                visualizations=frame_visualizations,
                should_switch_to_blindpath=False,
            )

        except Exception as e:
            self.green_light_counter = 0
            logger.error(f"[CROSS_STREET] Frame processing failed: {e}", exc_info=True)
            return CrossStreetResult(
                annotated_image=bgr_image,
                guidance_text="",
                visualizations=[],
                should_switch_to_blindpath=False,
            )


class YOLOModelWrapper:
    """Adapt a YOLO predict method to the navigation detect interface."""

    def __init__(self, yolo_model):
        self.model = yolo_model

    def detect(self, image, confidence_threshold=0.25):
        """Convert YOLO predictions to the navigation detection format."""
        try:
            results = self.model.predict(image, conf=confidence_threshold, verbose=False)
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, "masks") and result.masks is not None:
                    for i, mask in enumerate(result.masks.data):
                        if hasattr(result, "boxes") and result.boxes is not None:
                            cls = int(result.boxes.cls[i].cpu().numpy())
                            conf = float(result.boxes.conf[i].cpu().numpy())

                            class Detection:
                                def __init__(self):
                                    self.cls = cls
                                    self.conf = conf
                                    self.mask = mask.cpu().numpy()

                            detections.append(Detection())
            return detections
        except Exception as e:
            logger.error(f"[YOLO Wrapper] Detection failed: {e}")
            return []
