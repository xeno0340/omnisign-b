import os
import cv2
import numpy as np
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── CONFIG ───────────────────────────────────────────
DATA_DIR = r"D:\Projects\omnisign-b\data\asl_alphabet\asl_alphabet_train\asl_alphabet_train"
MODEL_OUT = r"D:\Projects\omnisign-b\asl_model.pkl"
LABELS_OUT = r"D:\Projects\omnisign-b\asl_labels.pkl"
TASK_FILE = r"D:\Projects\omnisign-b\hand_landmarker.task"

SKIP = {"del", "nothing", "space"}
IMAGES_PER_CLASS = 500
# ──────────────────────────────────────────────────────

# Setup MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=TASK_FILE)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

data = []
labels = []
classes = sorted([d for d in os.listdir(DATA_DIR) if d not in SKIP])

print(f"Found {len(classes)} classes: {classes}")
print(f"Processing {IMAGES_PER_CLASS} images per class...")

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    images = os.listdir(class_dir)[:IMAGES_PER_CLASS]
    count = 0

    for img_file in images:
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            min_x, min_y = min(xs), min(ys)

            row = []
            for lm in landmarks:
                row.append(lm.x - min_x)
                row.append(lm.y - min_y)

            data.append(row)
            labels.append(class_name)
            count += 1

    print(f"  {class_name}: {count} samples")

print(f"\nTotal samples: {len(data)}")

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

print(f"\nTraining Random Forest on {len(X_train)} samples...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.1f}%")

with open(MODEL_OUT, "wb") as f:
    pickle.dump(clf, f)

with open(LABELS_OUT, "wb") as f:
    pickle.dump(classes, f)

print(f"\nModel saved to: {MODEL_OUT}")
print(f"Labels saved to: {LABELS_OUT}")
print("Done!")