import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import cv2

MODEL_PATH = r"C:\Projects\omnisign-b\asl_model.pkl"
LABELS_PATH = r"C:\Projects\omnisign-b\asl_labels.pkl"
TASK_FILE = r"C:\Projects\omnisign-b\hand_landmarker.task"

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

class TrainedASLRecognizer:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.clf = pickle.load(f)
        with open(LABELS_PATH, "rb") as f:
            self.labels = pickle.load(f)

        base_options = python.BaseOptions(model_asset_path=TASK_FILE)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.text = ""
        self.current_sign = ""
        self.last_committed = ""
        self.last_sign = ""
        self.last_sign_time = 0
        self.committed = ""
        self.cooldown = 1.0
        self._last_confidence = 0.0
        self._last_result = None
        self._frame_count = 0
        self._last_commit_time = 0
        self.MAX_DISPLAY = 20

        # Emergency — triggered by signing H E L P in sequence
        self._sos_buffer = []
        self._emergency_triggered = False
        self._emergency_cooldown = 0

    def predict(self, frame_rgb):
        small = cv2.resize(frame_rgb, (320, 240))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
        result = self.detector.detect(mp_image)
        self._last_result = result

        if not result.hand_landmarks:
            return None, 0.0

        landmarks = result.hand_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        min_x, min_y = min(xs), min(ys)

        row = []
        for lm in landmarks:
            row.append(lm.x - min_x)
            row.append(lm.y - min_y)

        X = np.array(row).reshape(1, -1)
        pred = self.clf.predict(X)[0]
        proba = self.clf.predict_proba(X).max()

        self._last_confidence = float(proba)
        self.current_sign = pred
        return pred, float(proba)

    def step(self, frame_rgb):
        self.committed = ""
        self._frame_count += 1

        if self._frame_count % 2 != 0:
            return self.text.strip(), ""

        sign, confidence = self.predict(frame_rgb)

        if not sign:
            self.current_sign = ""
            return self.text.strip(), ""

        now = time.time()

        # Commit if same sign held for cooldown duration
        if sign == self.last_sign and (now - self.last_sign_time) > self.cooldown:
            self.text += sign + " "
            self.committed = sign
            self.last_committed = sign
            self.last_sign_time = now

            # Keep only last MAX_DISPLAY signs
            words = self.text.strip().split()
            if len(words) > self.MAX_DISPLAY:
                self.text = " ".join(words[-self.MAX_DISPLAY:]) + " "

            # ─── HELP → Emergency trigger ─────────────────
            self._sos_buffer.append(sign)
            self._sos_buffer = self._sos_buffer[-4:]  # keep last 4
            if self._sos_buffer == ['H', 'E', 'L', 'P']:
                self._sos_buffer = []
                return self.text.strip(), "EMERGENCY"
            # ──────────────────────────────────────────────

        elif sign != self.last_sign:
            self.last_sign = sign
            self.last_sign_time = now

        return self.text.strip(), self.committed

    def draw(self, frame, w, h):
        if self._last_result is None or not self._last_result.hand_landmarks:
            return frame

        for hand in self._last_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 140, 255), 2, lineType=cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(frame, (x, y), 4, (0, 220, 255), -1, lineType=cv2.LINE_AA)

        if self.current_sign:
            label = f"Sign: {self.current_sign} ({self._last_confidence:.0%})"
            cv2.putText(frame, label, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2, cv2.LINE_AA)

        return frame