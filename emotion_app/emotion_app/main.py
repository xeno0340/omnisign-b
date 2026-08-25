import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _project_root)
sys.path.insert(0, _here)

import math
import argparse
import threading
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

from asl_trained import TrainedASLRecognizer
from controls import BrightnessVolumeController

def send_to_voice(signs: list, emotion: str) -> None:
    try:
        requests.post("http://localhost:8001/signs-to-speech",
            json={"signs": signs, "emotion": emotion, "language": "English"}, timeout=5)
    except Exception:
        pass

def trigger_emergency() -> None:
    try:
        requests.post("http://localhost:8001/emergency", timeout=5)
    except Exception:
        pass

@dataclass
class Metrics:
    eye_open: float
    mouth_open: float
    smile: float
    brow_raise: float
    brow_furrow: float

def _l2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _pt(lm, w, h):
    return (lm.x * w, lm.y * h)

class FaceEmotionHeuristic:
    L_EYE_TOP=159; L_EYE_BOTTOM=145; L_EYE_LEFT=33; L_EYE_RIGHT=133
    R_EYE_TOP=386; R_EYE_BOTTOM=374; R_EYE_LEFT=362; R_EYE_RIGHT=263
    MOUTH_LEFT=61; MOUTH_RIGHT=291; MOUTH_TOP=13; MOUTH_BOTTOM=14
    UPPER_LIP=0; L_BROW=105; R_BROW=334; L_EYE_UP=159; R_EYE_UP=386
    BROW_INNER_L=107; BROW_INNER_R=336

    def compute_metrics(self, landmarks, w, h):
        pts = lambda idx: _pt(landmarks[idx], w, h)
        iod = max(_l2(pts(self.L_EYE_LEFT), pts(self.R_EYE_RIGHT)), 1.0)
        l_eye_open = _l2(pts(self.L_EYE_TOP), pts(self.L_EYE_BOTTOM))
        l_eye_w = _l2(pts(self.L_EYE_LEFT), pts(self.L_EYE_RIGHT))
        r_eye_open = _l2(pts(self.R_EYE_TOP), pts(self.R_EYE_BOTTOM))
        r_eye_w = _l2(pts(self.R_EYE_LEFT), pts(self.R_EYE_RIGHT))
        eye_open = 0.5*((l_eye_open/max(l_eye_w,1.0))+(r_eye_open/max(r_eye_w,1.0)))
        mouth_open = _l2(pts(self.MOUTH_TOP), pts(self.MOUTH_BOTTOM))
        mouth_w = _l2(pts(self.MOUTH_LEFT), pts(self.MOUTH_RIGHT))
        mouth_open_n = mouth_open/max(mouth_w,1.0)
        mouth_w_n = mouth_w/iod
        upper_lip_y = pts(self.UPPER_LIP)[1]
        mouth_center_y = 0.5*(pts(self.MOUTH_TOP)[1]+pts(self.MOUTH_BOTTOM)[1])
        smile = (mouth_w_n*1.2)+((mouth_center_y-upper_lip_y)/iod)
        l_brow_raise = (pts(self.L_EYE_UP)[1]-pts(self.L_BROW)[1])/iod
        r_brow_raise = (pts(self.R_EYE_UP)[1]-pts(self.R_BROW)[1])/iod
        brow_raise = 0.5*(l_brow_raise+r_brow_raise)
        brow_inner_dist = _l2(pts(self.BROW_INNER_L), pts(self.BROW_INNER_R))/iod
        brow_furrow = 1.0-np.clip(brow_inner_dist/0.35,0.0,1.0)
        return Metrics(float(eye_open),float(mouth_open_n),float(smile),float(brow_raise),float(brow_furrow))

    def score_emotions(self, m):
        happy = np.clip((m.smile-0.55)/0.35,0,1)+np.clip((0.25-m.mouth_open)/0.25,0,1)*0.25+np.clip((0.30-m.brow_furrow)/0.30,0,1)*0.15
        surprised = np.clip((m.mouth_open-0.30)/0.30,0,1)+np.clip((m.eye_open-0.28)/0.20,0,1)*0.7+np.clip((m.brow_raise-0.065)/0.060,0,1)*0.6
        angry = np.clip((m.brow_furrow-0.35)/0.40,0,1)+np.clip((0.24-m.eye_open)/0.20,0,1)*0.45+np.clip((0.55-m.smile)/0.35,0,1)*0.25
        sad = np.clip((0.60-m.smile)/0.40,0,1)*0.5+np.clip((m.brow_raise-0.06)/0.06,0,1)*0.45+np.clip((0.26-m.eye_open)/0.18,0,1)*0.35
        upset = np.clip((m.brow_furrow-0.25)/0.40,0,1)*0.55+np.clip((0.55-m.smile)/0.35,0,1)*0.35+np.clip((m.mouth_open-0.20)/0.35,0,1)*0.15
        scores = {"happy":float(happy),"surprised":float(surprised),"angry":float(angry),"sad":float(sad),"upset":float(upset)}
        scores["neutral"] = float(np.clip(1.0-max(scores.values()),0.0,1.0))
        return {k:float(np.clip(v,0.0,1.5)) for k,v in scores.items()}

    def predict(self, m):
        scores = self.score_emotions(m)
        label = max(scores.items(), key=lambda kv: kv[1])[0]
        if label != "neutral" and scores[label] < 0.55:
            label = "neutral"
        return label, scores

class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = float(alpha)
        self._prev = None
    def __call__(self, m):
        if self._prev is None:
            self._prev = m
            return m
        a = self.alpha
        sm = Metrics(a*m.eye_open+(1-a)*self._prev.eye_open,a*m.mouth_open+(1-a)*self._prev.mouth_open,
                     a*m.smile+(1-a)*self._prev.smile,a*m.brow_raise+(1-a)*self._prev.brow_raise,
                     a*m.brow_furrow+(1-a)*self._prev.brow_furrow)
        self._prev = sm
        return sm

def draw_points(frame, landmarks, w, h, indices, show_labels):
    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x*w), int(lm.y*h)
        cv2.circle(frame, (x,y), 2, (0,255,0), -1, lineType=cv2.LINE_AA)
        if show_labels:
            cv2.putText(frame, str(idx), (x+3,y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

def draw_asl_text(frame, text, raw, confidence=0.0):
    if not text and not raw:
        return
    h, w = frame.shape[:2]
    pad = 10
    line1 = f"ISL: {text}" if text else "ISL:"
    line2 = f"Detected: {raw} ({confidence:.0%})" if raw else ""
    box_h = 56
    cv2.rectangle(frame, (pad-6, h-box_h-pad), (min(w-pad,820), h-pad), (20,20,26), -1)
    cv2.rectangle(frame, (pad-6, h-box_h-pad), (min(w-pad,820), h-pad), (60,60,70), 1)
    cv2.putText(frame, line1, (pad, h-pad-28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    if line2:
        cv2.putText(frame, line2, (pad, h-pad-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,190), 1, cv2.LINE_AA)

def draw_emergency_overlay(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 12)
    cv2.putText(frame, "EMERGENCY ALERT", (w//2-160, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)

def make_side_panel(h, w, label, scores, metrics, fps, last_sentence=""):
    panel = np.zeros((h,w,3), dtype=np.uint8)
    panel[:] = (22,22,28)
    def put(y, text, scale=0.6, color=(230,230,235), thickness=1):
        cv2.putText(panel, text, (18,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)
    put(40, "Expression", 0.75, (245,245,250), 2)
    put(78, label.upper(), 1.2, (120,210,255) if label != "neutral" else (210,210,210), 3)
    put(120, f"FPS: {fps:.1f}", 0.6, (190,190,200), 1)
    if last_sentence:
        put(148, "Last spoken:", 0.55, (180,180,190), 1)
        display = last_sentence[:35]+"..." if len(last_sentence)>35 else last_sentence
        put(168, display, 0.5, (100,220,150), 1)
    y0 = 200
    put(y0-18, "Scores", 0.65, (245,245,250), 2)
    bar_x, bar_w, bar_h = 18, w-36, 14
    y = y0
    for k in ["happy","sad","angry","surprised","upset","neutral"]:
        s = float(np.clip(scores.get(k,0.0),0.0,1.0))
        cv2.rectangle(panel, (bar_x,y), (bar_x+bar_w, y+bar_h), (40,40,48), -1)
        cv2.rectangle(panel, (bar_x,y), (bar_x+int(bar_w*s), y+bar_h), (80,170,110), -1)
        cv2.putText(panel, k, (bar_x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,225), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{s:.2f}", (bar_x+bar_w-52,y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,225), 1, cv2.LINE_AA)
        y += 34
    y += 8
    put(y, "Landmark metrics", 0.65, (245,245,250), 2); y += 26
    for name, val in [("eye_open",metrics.eye_open),("mouth_open",metrics.mouth_open),("smile",metrics.smile),("brow_raise",metrics.brow_raise),("brow_furrow",metrics.brow_furrow)]:
        put(y, f"{name:11s}: {val:.3f}", 0.55, (210,210,220), 1); y += 22
    put(h-22, "q=quit | d=dots | l=labels | e=emergency", 0.45, (160,160,170), 1)
    return panel

def open_camera(preferred_index=0):
    indices = [preferred_index]+[i for i in [0,1,2,3] if i != preferred_index]
    backends = []
    if hasattr(cv2,"CAP_DSHOW"): backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2,"CAP_MSMF"): backends.append(cv2.CAP_MSMF)
    backends.append(None)
    for idx in indices:
        for be in backends:
            cap = cv2.VideoCapture(idx) if be is None else cv2.VideoCapture(idx, be)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok: return cap
            cap.release()
    raise RuntimeError("Could not open a webcam.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--controls", action="store_true")
    args = ap.parse_args()

    cap = open_camera(preferred_index=int(args.camera))

    face_options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
                        model_asset_path=os.path.join(_project_root, "face_landmarker.task")
        ),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_detector = FaceLandmarker.create_from_options(face_options)

    classifier = FaceEmotionHeuristic()
    smoother = LandmarkSmoother(alpha=0.55)

    print("Loading trained ASL model (99.0% accuracy)...")
    asl = TrainedASLRecognizer()
    print("Ready. Sign H-E-L-P to trigger emergency alert.")

    show_dots = True
    show_labels = False
    emergency_flash = 0
    last_sentence = ""
    label = "no face"
    scores = {"neutral": 1.0}
    metrics = Metrics(0.0,0.0,0.0,0.0,0.0)

    major_points = sorted(set([33,133,159,145,362,263,386,374,105,66,107,55,334,296,336,285,
                                1,2,4,5,6,197,168,61,291,13,14,0,17,78,308,152,377,400,378,176,148]))
    last_t = cv2.getTickCount()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Face emotion
        res = face_detector.detect(mp_image)
        if res.face_landmarks:
            lms = res.face_landmarks[0]
            metrics = classifier.compute_metrics(lms, w, h)
            metrics = smoother(metrics)
            label, scores = classifier.predict(metrics)
            if show_dots:
                draw_points(frame, lms, w, h, major_points, show_labels)
        else:
            label = "no face"
            scores = {"neutral": 1.0}

        # ASL recognition
        asl_text, asl_committed = asl.step(rgb)
        frame = asl.draw(frame, w, h)
        last_confidence = getattr(asl, '_last_confidence', 0.0)
        draw_asl_text(frame, asl_text, asl_committed, last_confidence)

        # Handle committed sign
        if asl_committed == "EMERGENCY":
            # H-E-L-P sequence detected → trigger emergency
            emergency_flash = 60  # 2 seconds of red border
            threading.Thread(target=trigger_emergency).start()
            last_sentence = "🚨 EMERGENCY ALERT!"
            print("EMERGENCY TRIGGERED by H-E-L-P sequence!")
        elif asl_committed:
            emotion_label = label if label != "no face" else "neutral"
            threading.Thread(target=send_to_voice, args=([asl_committed], emotion_label)).start()
            last_sentence = asl_committed

        # Emergency overlay
        if emergency_flash > 0:
            draw_emergency_overlay(frame)
            emergency_flash -= 1

        now = cv2.getTickCount()
        dt = (now-last_t)/cv2.getTickFrequency()
        last_t = now
        if dt > 0:
            fps = 0.9*fps+0.1*(1.0/dt) if fps > 0 else (1.0/dt)

        panel_w = max(320, int(w*0.42))
        panel = make_side_panel(h, panel_w, label, scores, metrics, fps, last_sentence)
        out = np.concatenate([frame, panel], axis=1)
        cv2.imshow("OmniSign — Real-time ISL Translator", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord("d"): show_dots = not show_dots
        if key == ord("l"): show_labels = not show_labels
        if key == ord("e"):
            # Manual emergency trigger with E key
            emergency_flash = 60
            threading.Thread(target=trigger_emergency).start()
            last_sentence = "🚨 EMERGENCY ALERT!"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()