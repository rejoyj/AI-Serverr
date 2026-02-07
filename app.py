from flask import Flask
import cv2
import numpy as np
import requests
import threading
import time
from ultralytics import YOLO

# ---------------- CONFIG ----------------

MOBILE_CAMERA_URL = "http://192.168.137.136:8080/video"  # CHANGE IP
NODE_BACKEND = "https://smartparking-lyla.onrender.com/update-slot"

CONFIDENCE_THRESHOLD = 0.4
CAR_CLASS_ID = 2  # COCO class for "car"

# Parking slot regions (x1, y1, x2, y2)
SLOTS = {
    1: (0, 0, 213, 480),
    2: (213, 0, 426, 480),
    3: (426, 0, 640, 480),
}

# ----------------------------------------

app = Flask(__name__)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(MOBILE_CAMERA_URL)

if not cap.isOpened():
    print("❌ Cannot open mobile camera")
    exit()

last_sent = {}

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return {
        "status": "AI Server Running",
        "slots": last_sent
    }

# ---------------- FUNCTIONS ----------------

def check_slots(car_boxes):
    status = {1: False, 2: False, 3: False}

    for slot_id, (x1, y1, x2, y2) in SLOTS.items():
        for (bx1, by1, bx2, by2) in car_boxes:
            cx = int((bx1 + bx2) / 2)
            cy = int((by1 + by2) / 2)

            if x1 < cx < x2 and y1 < cy < y2:
                status[slot_id] = True

    return status


last_sent = {}
last_post_time = 0   # ADD THIS GLOBAL TIMER


def send_to_backend(status):
    global last_sent, last_post_time

    now = time.time()

    # send every 5 seconds
    if now - last_post_time >= 5:
        try:
            requests.post(NODE_BACKEND, json=status, timeout=2)
            last_sent = status.copy()
            last_post_time = now
            print("📡 Sent to backend:", status)
        except:
            print("⚠ Backend not reachable")



# ---------------- AI LOOP ----------------

def ai_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, verbose=False)[0]
        car_boxes = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == CAR_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_boxes.append((x1, y1, x2, y2))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    "CAR",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

        slot_status = check_slots(car_boxes)
        send_to_backend(slot_status)

        # Draw slots
        for slot_id, (x1, y1, x2, y2) in SLOTS.items():
            occupied = slot_status[slot_id]
            color = (0, 0, 255) if occupied else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                f"Slot {slot_id}",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

        cv2.imshow("Smart Parking AI (Mobile Camera)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- START SERVER ----------------

if __name__ == "__main__":
    threading.Thread(target=ai_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
