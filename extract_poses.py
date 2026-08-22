import os
import cv2
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR = r"D:\Projects\omnisign-b\data\asl_alphabet\asl_alphabet_train\asl_alphabet_train"
POSES_OUT = r"D:\Projects\omnisign-b\hand_poses.json"
TASK_FILE = r"D:\Projects\omnisign-b\hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=TASK_FILE)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

SKIP = {"del", "nothing", "space"}
poses = {}
classes = sorted([d for d in os.listdir(DATA_DIR) if d not in SKIP])

print("Extracting poses...")

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    images = os.listdir(class_dir)
    
    # Try images until we get a good detection
    for img_file in images[:50]:
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            pose = [[lm.x, lm.y, lm.z] for lm in landmarks]
            poses[class_name] = pose
            print(f"  {class_name}: extracted")
            break
    
    if class_name not in poses:
        print(f"  {class_name}: FAILED")

with open(POSES_OUT, "w") as f:
    json.dump(poses, f)

print(f"\nSaved {len(poses)} poses to {POSES_OUT}")
print("Done!")