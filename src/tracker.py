"""
tracker.py

DeepSORT-based tracking (replaces ByteTrack).

Why DeepSORT over ByteTrack for this project:
  - DeepSORT keeps an appearance embedding (a small CNN feature vector)
    for each track, not just motion/IoU. When a person is briefly
    occluded (e.g. walks behind a shelf) and reappears, DeepSORT can
    re-match them to their ORIGINAL track ID using appearance similarity,
    not just position -- ByteTrack relies more heavily on motion
    continuity and tends to hand out a new ID more readily after a gap.
  - This directly reduces the "ghost -> reassigned ID" problem, since
    fewer real people end up needing a brand new ID after occlusion.

Note this does NOT eliminate missed detections themselves (that's a
detector/YOLO-nano limitation, not a tracking one) -- it only improves
how well we RE-CONNECT a person's identity once they're detected again.

The interface returned to the rest of the pipeline (task logic modules)
is unchanged: a list of dicts with track_id, bbox, confidence, is_ghost.
"""

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2

PERSON_CLASS_ID = 0


class PersonTracker:
    def __init__(
        self,
        model_path="models/yolov8s.pt",
        conf_threshold=0.45,
        device="cpu",
        fps=30,
        max_missed_seconds=2.0,
        n_init=2,
        max_cosine_distance=0.40,
        max_iou_distance=0.7,
        min_box_area=1200,
        exclusion_zones=None,
    ):
        
        self.exclusion_zones = [np.array(zone, dtype="int32") for zone in (exclusion_zones or [])]
        """
        Person tracker wrapper around YOLO + DeepSORT with tunable parameters.

        - conf_threshold: minimum detection confidence to feed into tracker
        - max_missed_seconds: how long (seconds) a track can be missed before being dropped
        - n_init: number of consecutive detections required to confirm a track (higher -> fewer false positives / ID jitter)
        - max_cosine_distance: appearance distance threshold for re-identification (lower = stricter matching)
        - max_iou_distance: IoU threshold used by the tracker for matching
        - min_box_area: ignore tiny detections (likely noise/false positives)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.fps = fps
        self.min_box_area = min_box_area

        # max_age: frames to keep a track alive while coasting
        max_age = max(1, int(max_missed_seconds * fps))

        self.deepsort = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            max_iou_distance=max_iou_distance,
            embedder="mobilenet",
            half=False,
        )
    def _is_in_exclusion_zone(self, point):
        for zone in self.exclusion_zones:
            if cv2.pointPolygonTest(zone, point, False) >= 0:
                return True
        return False
    
    def update(self, frame, frame_index):
        """
        Runs YOLO detection, filters detections, then passes them to DeepSORT.

        Returns a list of dicts:
            {"track_id": int, "bbox": (x1,y1,x2,y2), "confidence": float, "is_ghost": bool}
        """
        results = self.model.predict(
            source=frame, conf=self.conf_threshold, classes=[PERSON_CLASS_ID],
            device=self.device, verbose=False,
            iou =0.5,
        )

        raw_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                if self._is_in_exclusion_zone(center):
                    continue

                # Filter out low-confidence and tiny detections (common false positives)
                if conf < self.conf_threshold:
                    continue
                if area < self.min_box_area:
                    continue

                # deep_sort_realtime expects ([left, top, w, h], confidence, class)
                raw_detections.append(([x1, y1, w, h], conf, "person"))

        tracks = self.deepsort.update_tracks(raw_detections, frame=frame)

        current_tracks = []
        for tr in tracks:
            # Only return confirmed tracks (helps reduce rapid ID churn / false positives)
            if not tr.is_confirmed():
                continue
            l, t, r, b = tr.to_ltrb()
            tid = tr.track_id
            tid_int = int(tid) if str(tid).isdigit() else abs(hash(tid)) % 100000

            # Use the track's last detection confidence if available
            conf_out = getattr(tr, "last_detection_conf", None)
            if conf_out is None:
                # fall back to 1.0 if not provided by the tracker wrapper
                conf_out = 1.0

            current_tracks.append({
                "track_id": tid_int,
                "bbox": (int(l), int(t), int(r), int(b)),
                "confidence": float(conf_out),
                "is_ghost": tr.time_since_update > 0,
            })

        # Optional: remove near-duplicate tracks (very high IoU & different IDs)
        # This post-filter reduces cases where the tracker briefly assigns multiple IDs
        # to the same person due to overlapping detections. Keep this conservative.
        filtered = []
        for ct in current_tracks:
            keep = True
            for f in filtered:
                # compute IoU between boxes
                ax1, ay1, ax2, ay2 = ct["bbox"]
                bx1, by1, bx2, by2 = f["bbox"]
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    iou = 0.0
                else:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area_a = (ax2 - ax1) * (ay2 - ay1)
                    area_b = (bx2 - bx1) * (by2 - by1)
                    iou = inter_area / float(area_a + area_b - inter_area)
                if iou > 0.80:
                    # keep the one with higher confidence to reduce duplicate IDs
                    if ct["confidence"] <= f["confidence"]:
                        keep = False
                        break
                    else:
                        filtered.remove(f)
                        break
            if keep:
                filtered.append(ct)

        return filtered

    @staticmethod
    def get_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def get_foot_point(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, y2)