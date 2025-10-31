# src/video/realtime.py
import time
import cv2
import numpy as np

try:
    from .landmarks import extract_frame_features
except Exception:
    # fallback relative import if module layout differs
    from src.video.landmarks import extract_frame_features


def capture_face_live(
    frames: int | None = None,
    seconds: float | None = None,
    device_index: int = 0,
    timeout_s: float = 20.0,
    fps: int = 15,
):
    """
    Capture webcam for given frames/seconds and return [EAR, MAR, brow_y] or None.
    Accepts either `frames` OR `seconds` (your app passes seconds).
    """
    # Normalize frames from seconds
    if frames is None and seconds is not None:
        frames = max(1, int(seconds * fps))
    if frames is None:
        frames = 15  # default 1 second at 15 fps

    cap = cv2.VideoCapture(int(device_index), cv2.CAP_DSHOW)  # DSHOW = stable on Windows
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {device_index}")

    feats = []
    start = time.time()

    try:
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok:
                # give the camera a moment and retry a few times
                time.sleep(0.03)
                continue

            f = extract_frame_features(frame)  # -> [ear, mar, brow_y] or None
            if f is not None:
                feats.append(f)

            # safety timeout
            if (time.time() - start) > timeout_s:
                break
    finally:
        cap.release()

    if not feats:
        return None

    # average across collected frames
    feat = np.mean(np.stack(feats, axis=0), axis=0)
    return feat  # [ear, mar, brow_y]
