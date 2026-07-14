import os
import glob
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# EDIT THESE 3 LINES
# ---------------------------------------------------------------------------
SEQ_DIR = "data/VisDrone2019-MOT-split/sequences/uav0000305_00000_v"           # folder of frame images
MODEL_PATH = "outputs/runs/visdrone_baseline-6/weights/best.pt"    # trained YOLO weights
OUTPUT_PATH = "outputs/tracked/tracked_output_3.mp4"                # where to save the result

CONF_THRESHOLD = 0.25   # detection confidence cutoff
IOU_THRESHOLD = 0.05     # SORT matching threshold
MAX_AGE = 40            # frames a track survives with no matching detection
MIN_HITS = 3            # detections needed before a track is confirmed/drawn
FPS = 20

CLASS_NAMES = {0: "pedestrian", 1: "car", 2: "van", 3: "truck"}
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(1000, 3))


# ---------------------------------------------------------------------------
# Helpers: bounding box <-> Kalman filter state conversions
# ---------------------------------------------------------------------------

def bbox_to_state(bbox):
    """[x1,y1,x2,y2] -> [center_x, center_y, area, aspect_ratio]"""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = bbox[0] + w / 2, bbox[1] + h / 2
    return np.array([cx, cy, w * h, w / max(h, 1e-6)]).reshape((4, 1))


def state_to_bbox(state):
    """[center_x, center_y, area, aspect_ratio, ...] -> [x1,y1,x2,y2]"""
    w = np.sqrt(max(state[2] * state[3], 0))
    h = state[2] / max(w, 1e-6)
    cx, cy = state[0], state[1]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).reshape((1, 4))


def iou(boxes_a, boxes_b):
    """IOU between every box in boxes_a and every box in boxes_b."""
    a = np.expand_dims(boxes_a, 1)
    b = np.expand_dims(boxes_b, 0)
    ix1, iy1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    ix2, iy2 = np.minimum(a[..., 2], b[..., 2]), np.minimum(a[..., 3], b[..., 3])
    intersection = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-6)


# ---------------------------------------------------------------------------
# One tracked object = one Kalman filter (constant velocity model)
# ---------------------------------------------------------------------------

class Track:
    next_id = 1

    def __init__(self, bbox, cls_id):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # state: [cx, cy, area, aspect_ratio, vx, vy, v_area]
        self.kf.F = np.eye(7)
        self.kf.F[0, 4] = self.kf.F[1, 5] = self.kf.F[2, 6] = 1  # constant velocity
        self.kf.H = np.eye(4, 7)
        self.kf.R[2:, 2:] *= 10
        self.kf.P[4:, 4:] *= 1000   # low confidence in initial velocity
        self.kf.P *= 10
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = bbox_to_state(bbox)

        self.id = Track.next_id
        Track.next_id += 1
        self.cls_id = cls_id
        self.hits = 1
        self.age_since_last_match = 0
        self.total_age = 0

    def predict(self):
        self.kf.predict()
        self.total_age += 1
        self.age_since_last_match += 1
        return state_to_bbox(self.kf.x)[0]

    def update(self, bbox, cls_id):
        self.kf.update(bbox_to_state(bbox))
        self.cls_id = cls_id
        self.hits += 1
        self.age_since_last_match = 0

    def current_bbox(self):
        return state_to_bbox(self.kf.x)[0]

    def is_confirmed(self):
        return self.hits >= MIN_HITS or self.total_age <= MIN_HITS


# ---------------------------------------------------------------------------
# SORT tracker: keeps a list of Tracks and matches detections to them each frame
# ---------------------------------------------------------------------------

class Sort:
    def __init__(self):
        self.tracks = []

    def update(self, detections):
        """detections: list of [x1,y1,x2,y2,conf,cls_id]. Returns list of (bbox, id, cls_id)."""
        # 1. Predict where every existing track should be this frame
        predicted_boxes = np.array([t.predict() for t in self.tracks]) if self.tracks else np.empty((0, 4))
        det_boxes = np.array([d[:4] for d in detections]) if detections else np.empty((0, 4))

        # 2. Match detections to predicted track positions by IOU (Hungarian algorithm)
        matched_dets, matched_tracks = set(), set()
        if len(predicted_boxes) and len(det_boxes):
            iou_matrix = iou(det_boxes, predicted_boxes)
            det_idx, track_idx = linear_sum_assignment(-iou_matrix)
            for d, t in zip(det_idx, track_idx):
                if iou_matrix[d, t] >= IOU_THRESHOLD:
                    self.tracks[t].update(detections[d][:4], int(detections[d][5]))
                    matched_dets.add(d)
                    matched_tracks.add(t)

        # 3. Unmatched detections become new tracks
        for d, det in enumerate(detections):
            if d not in matched_dets:
                self.tracks.append(Track(det[:4], int(det[5])))

        # 4. Drop tracks that have been missing too long
        self.tracks = [t for t in self.tracks if t.age_since_last_match <= MAX_AGE]

        # 5. Return confirmed, currently-matched tracks for drawing
        results = []
        for i, t in enumerate(self.tracks):
            if t.age_since_last_match == 0 and t.is_confirmed():
                results.append((t.current_bbox(), t.id, t.cls_id))
        return results


# ---------------------------------------------------------------------------
# Main loop: read frames -> detect -> track -> draw -> write video
# ---------------------------------------------------------------------------

def main():
    model = YOLO(MODEL_PATH)
    tracker = Sort()

    frame_paths = sorted(glob.glob(os.path.join(SEQ_DIR, "*.jpg")) + glob.glob(os.path.join(SEQ_DIR, "*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {SEQ_DIR}")

    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))

    seen_ids = set()

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)

        # Run detector
        result = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []
        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy()):
            detections.append([box[0], box[1], box[2], box[3], conf, cls])

        # Update tracker and draw results
        for bbox, track_id, cls_id in tracker.update(detections):
            seen_ids.add(track_id)
            x1, y1, x2, y2 = bbox.astype(int)
            color = tuple(int(c) for c in COLORS[track_id % len(COLORS)])
            label = f"ID {track_id} {CLASS_NAMES.get(cls_id, '?')}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        if i % 50 == 0:
            print(f"frame {i}/{len(frame_paths)}")

    writer.release()
    print(f"Done. {len(seen_ids)} unique tracks. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()