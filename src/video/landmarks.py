# src/video/landmarks.py
# Minimal, reliable facial-landmark feature extractor for NeuroAid

from __future__ import annotations
import numpy as np

try:
    import cv2
    import mediapipe as mp
except Exception:
    cv2 = None
    mp = None


def _aspect_ratio(pts: np.ndarray) -> float:
    """
    Generic 6-point aspect ratio:
      pts order assumed: [0,1,2,3,4,5]
      A = |p1-p5| + |p2-p4|
      B = 2*|p0-p3|
    """
    A = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    B = 2.0 * np.linalg.norm(pts[0] - pts[3])
    return float(A / (B + 1e-6))


def extract_frame_features(frame) -> np.ndarray | None:
    """
    Returns [EAR, MAR, brow_y] or None if face not found / libs missing.
    EAR ~ Eye Aspect Ratio, MAR ~ Mouth Aspect Ratio,
    brow_y ~ vertical eyebrow tension proxy (larger negative -> raised brows).
    """
    if mp is None or cv2 is None:
        return None

    mp_face = mp.solutions.face_mesh

    # static_image_mode=True → robust per-frame; FaceMesh is light enough
    with mp_face.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as fm:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)

        # Landmark index sets (MediaPipe Face Mesh)
        # Left eye rough 6 points
        left_eye_id = [33, 160, 158, 133, 153, 144]
        # Mouth rough 6 points
        mouth_id = [78, 81, 13, 311, 308, 402]

        if pts.shape[0] <= max(left_eye_id + mouth_id + [107, 70]):
            return None

        left_eye = pts[left_eye_id]
        mouth = pts[mouth_id]

        ear = _aspect_ratio(left_eye)
        mar = _aspect_ratio(mouth)

        # Simple eyebrow metric (vertical delta between two brow points)
        # 70 (left brow) and 107 (under-brow/eye region) give a vertical distance
        brow_y = float(pts[70, 1] - pts[107, 1])

        return np.array([ear, mar, brow_y], dtype=np.float32)


__all__ = ["extract_frame_features"]
