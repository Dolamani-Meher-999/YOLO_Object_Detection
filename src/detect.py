from ultralytics import YOLO
import cv2
import time
import os
from datetime import datetime

# ---------------- SETTINGS ----------------

MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.5
SAVE_INTERVAL = 3  # seconds between saved images

# ------------------------------------------

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

width = int(cap.get(3))
height = int(cap.get(4))

# Create folders
os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("outputs/images", exist_ok=True)

# Video filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"outputs/videos/output_{timestamp}.avi"

out = cv2.VideoWriter(
    video_filename,
    cv2.VideoWriter_fourcc(*"XVID"),
    20,
    (width, height)
)

prev_time = 0
last_save_time = 0

print("Detection started...")
print("Video saving to:", video_filename)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    # Run detection
    results = model(frame, conf=CONF_THRESHOLD)

    annotated_frame = results[0].plot()

    # -------- PERSON COUNT --------
    person_count = 0

    if results[0].boxes is not None:
        classes = results[0].boxes.cls

        for cls in classes:
            if int(cls) == 0:
                person_count += 1

    # -------- SAVE IMAGE --------
    if person_count > 0:
        if current_time - last_save_time > SAVE_INTERVAL:

            image_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            image_filename = (
                f"outputs/images/person_{image_timestamp}.jpg"
            )

            cv2.imwrite(image_filename, annotated_frame)

            print("Saved image:", image_filename)

            last_save_time = current_time

    # -------- FPS --------
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # -------- DISPLAY --------

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Persons: {person_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Conf: {CONF_THRESHOLD}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    # Save video
    out.write(annotated_frame)

    # Show window
    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved to:", video_filename)