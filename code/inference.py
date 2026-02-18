import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

VIDEO_PATH = "/home/tihan40909/Desktop/HSI/Courses/Hasini/EdgeAiFleet/25_nov_2025/25_nov_2025/15.mov"
OUTPUT_VIDEO_PATH = "outputs/processed_video_15.mp4"
OUTPUT_CSV_PATH = "outputs/annotations_15.csv"

CONF_THRESHOLD = 0.01
BALL_CLASS_ID = 32  # COCO class id for sports ball

def main():
    model = YOLO("yolov8l.pt")  # lightweight & fast

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    trajectory = []
    records = []

    frame_index = 0

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # results = model(frame, verbose=False)[0]
        results = model.track(frame, persist=True, imgsz=1280, conf=CONF_THRESHOLD, verbose=False)[0]


        ball_detected = False
        x_centroid, y_centroid = -1, -1

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == BALL_CLASS_ID and conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0]
                x_centroid = int((x1 + x2) / 2)
                y_centroid = int((y1 + y2) / 2)

                ball_detected = True
                trajectory.append((x_centroid, y_centroid))

                # Draw bounding box
                cv2.rectangle(frame,
                              (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              (0, 255, 0), 2)

                break

        # Draw trajectory
        for i in range(1, len(trajectory)):
            cv2.line(frame,
                     trajectory[i - 1],
                     trajectory[i],
                     (0, 0, 255), 2)

        # Draw centroid
        if ball_detected:
            cv2.circle(frame, (x_centroid, y_centroid), 5, (255, 0, 0), -1)

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
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Processing complete.")
    print("Saved video to:", OUTPUT_VIDEO_PATH)
    print("Saved annotations to:", OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
