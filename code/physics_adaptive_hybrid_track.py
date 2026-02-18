import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

INPUT_FOLDER = "/home/tihan40909/Desktop/HSI/Courses/Hasini/EdgeAiFleet/25_nov_2025/25_nov_2025/"
OUTPUT_FOLDER = "./outputs_adaptive_physics_hybrid/"

CONF_THRESHOLD = 0.15
BALL_CLASS_ID = 32

MAX_DISTANCE = 200
MIN_SPEED = 4
MAX_SPEED = 250
STATIC_FRAME_LIMIT = 5
AREA_TOLERANCE = 2.0

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def detect_in_roi(model, frame, roi_coords):
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]

    results = model.predict(
        roi,
        imgsz=1280,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == BALL_CLASS_ID and conf > CONF_THRESHOLD:
            bx1, by1, bx2, by2 = box.xyxy[0]
            w = int(bx2 - bx1)
            h = int(by2 - by1)
            area = w * h

            if area > 8000:
                continue

            return (
                int(bx1 + x1),
                int(by1 + y1),
                int(bx2 + x1),
                int(by2 + y1),
                area
            )

    return None


def valid_motion(trajectory, x, y):
    if len(trajectory) < 2:
        return True

    prev_x, prev_y = trajectory[-1]
    dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

    if dist < MIN_SPEED:
        return False

    if dist > MAX_SPEED:
        return False

    if len(trajectory) >= 2:
        prev2_x, prev2_y = trajectory[-2]

        v1 = np.array([prev_x - prev2_x, prev_y - prev2_y])
        v2 = np.array([x - prev_x, y - prev_y])

        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cosine = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2)
            )

            if cosine < -0.5:
                return False

    return True


def process_video(video_path, model):

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    output_video_path = os.path.join(
        OUTPUT_FOLDER, f"{video_name}_processed.mp4"
    )
    output_csv_path = os.path.join(
        OUTPUT_FOLDER, f"{video_name}_annotations.csv"
    )

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    trajectory = []
    records = []

    tracker = None
    tracking_mode = False
    last_area = None
    static_counter = 0

    frame_index = 0

    for _ in tqdm(range(total_frames), desc=f"Processing {video_name}"):

        ret, frame = cap.read()
        if not ret:
            break

        ball_detected = False
        x_centroid, y_centroid = -1, -1

        if tracking_mode and tracker is not None:

            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = bbox
                area = w * h

                x_centroid = int(x + w / 2)
                y_centroid = int(y + h / 2)

                if not valid_motion(trajectory, x_centroid, y_centroid):
                    success = False

                if last_area and area > last_area * AREA_TOLERANCE:
                    success = False

                if success:
                    if len(trajectory) > 0:
                        prev_x, prev_y = trajectory[-1]
                        if abs(x_centroid - prev_x) < 2 and abs(y_centroid - prev_y) < 2:
                            static_counter += 1
                        else:
                            static_counter = 0

                    if static_counter > STATIC_FRAME_LIMIT:
                        success = False

                if success:
                    ball_detected = True
                    trajectory.append((x_centroid, y_centroid))
                    last_area = area

                    cv2.rectangle(frame,
                                  (int(x), int(y)),
                                  (int(x + w), int(y + h)),
                                  (0, 255, 0), 2)
                else:
                    tracking_mode = False
                    tracker = None
                    static_counter = 0

            else:
                tracking_mode = False
                tracker = None

        if not tracking_mode:

            detection = detect_in_roi(
                model, frame, (0, 0, width, height)
            )

            if detection:
                x1, y1, x2, y2, area = detection

                x_centroid = int((x1 + x2) / 2)
                y_centroid = int((y1 + y2) / 2)

                if valid_motion(trajectory, x_centroid, y_centroid):

                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame,
                                 (x1, y1, x2 - x1, y2 - y1))

                    tracking_mode = True
                    last_area = area
                    trajectory.append((x_centroid, y_centroid))
                    ball_detected = True

                    cv2.rectangle(frame,
                                  (x1, y1),
                                  (x2, y2),
                                  (0, 255, 0), 2)

        for i in range(1, len(trajectory)):
            cv2.line(frame,
                     trajectory[i - 1],
                     trajectory[i],
                     (0, 0, 255), 2)

        if ball_detected:
            cv2.circle(frame,
                       (x_centroid, y_centroid),
                       5,
                       (255, 0, 0),
                       -1)

        visibility_flag = 1 if ball_detected else 0

        records.append({
            "frame_index": frame_index,
            "x_centroid": x_centroid,
            "y_centroid": y_centroid,
            "visibility_flag": visibility_flag
        })

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)

    print(f"Finished {video_name}")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    model = YOLO("yolov8l.pt")

    video_files = [
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    print(f"Found {len(video_files)} videos.")

    for video_path in video_files:
        process_video(video_path, model)

    print("All videos processed.")


if __name__ == "__main__":
    main()
