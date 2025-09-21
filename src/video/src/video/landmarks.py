import cv2, numpy as np, mediapipe as mp
mp_face = mp.solutions.face_mesh

def extract_frame_features(frame):
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        pts = np.array([[p.x*w, p.y*h] for p in lm.landmark], dtype=np.float32)
        left_eye = pts[[33,160,158,133,153,144]]
        mouth    = pts[[78,81,13,311,308,402]]
        def _ar(pts):
            A = np.linalg.norm(pts[1]-pts[5]) + np.linalg.norm(pts[2]-pts[4])
            B = 2.0*np.linalg.norm(pts[0]-pts[3])
            return float(A / (B + 1e-6))
        ear = _ar(left_eye)
        mar = _ar(mouth)
        brow_y = float(pts[70,1] - pts[107,1]) if pts.shape[0] > 108 else 0.0
        return np.array([ear, mar, brow_y], dtype=np.float32)
