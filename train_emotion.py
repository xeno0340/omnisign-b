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

DATA_DIR = r"D:\Projects\omnisign-b\data\fer2013\train"
MODEL_OUT = r"D:\Projects\omnisign-b\emotion_model.pkl"
LABELS_OUT = r"D:\Projects\omnisign-b\emotion_labels.pkl"
TASK_FILE = r"D:\Projects\omnisign-b\face_landmarker.task"

IMAGES_PER_CLASS = 500

base_options = python.BaseOptions(model_asset_path=TASK_FILE)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.FaceLandmarker.create_from_options(options)

data = []
labels = []
classes = sorted(os.listdir(DATA_DIR))
print(f"Found classes: {classes}")

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    images = os.listdir(class_dir)[:IMAGES_PER_CLASS]
    count = 0

    for img_file in images:
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # FER2013 is grayscale — convert to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
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

if len(data) == 0:
    print("No data found — check DATA_DIR path")
    exit()

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

print(f"Training Random Forest on {len(X_train)} samples...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.1f}%")

with open(MODEL_OUT, "wb") as f:
    pickle.dump(clf, f)
with open(LABELS_OUT, "wb") as f:
    pickle.dump(classes, f)

print(f"Model saved to: {MODEL_OUT}")
print("Done!")