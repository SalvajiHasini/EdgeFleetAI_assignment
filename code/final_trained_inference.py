import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# ===================== CONFIG =====================

MODEL_PATH = "./runs/detect/cricket_ball_finetune3/weights/best.pt"  
INPUT_FOLDER = "/home/tihan40909/Desktop/HSI/Courses/Hasini/EdgeAiFleet/25_nov_2025/25_nov_2025/"                    
OUTPUT_FOLDER = "./outputs_final_final_2"


#---------------------------------version 2-------------------------------------------------#
CONF_THRESHOLD = 0.35
MIN_AREA = 50
MAX_AREA = 2000
MAX_DISTANCE = 200
SMOOTHING_WINDOW = 3

ROI_WIDTH_RATIO = 0.6
ROI_HEIGHT_RATIO = 0.6

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

# ==================================================


def smooth_point(trajectory, window=3):
    if len(trajectory) < window:
        return trajectory[-1]
    xs = [p[0] for p in trajectory[-window:]]
    ys = [p[1] for p in trajectory[-window:]]
    return (int(np.mean(xs)), int(np.mean(ys)))


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

    # Define center ROI
    roi_w = int(width * ROI_WIDTH_RATIO)
    roi_h = int(height * ROI_HEIGHT_RATIO)

    roi_x1 = (width - roi_w) // 2
    roi_y1 = (height - roi_h) // 2
    roi_x2 = roi_x1 + roi_w
    roi_y2 = roi_y1 + roi_h

    trajectory = []
    records = []

    print(f"\nProcessing {video_name}...")

    for frame_index in tqdm(range(total_frames)):

        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        results = model.predict(
            roi,
            imgsz=1280,
            conf=CONF_THRESHOLD,
            verbose=False
        )[0]

        best_conf = 0
        best_box = None

        for box in results.boxes:

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]

            area = float((x2 - x1) * (y2 - y1))

            if area < MIN_AREA or area > MAX_AREA:
                continue

            if conf > best_conf:
                best_conf = conf
                best_box = box

        ball_detected = False
        x_centroid, y_centroid = -1, -1

        if best_box is not None:

            x1, y1, x2, y2 = best_box.xyxy[0]

            # Convert ROI coordinates back to full frame
            x1_full = int(x1 + roi_x1)
            y1_full = int(y1 + roi_y1)
            x2_full = int(x2 + roi_x1)
            y2_full = int(y2 + roi_y1)

            cx = int((x1_full + x2_full) / 2)
            cy = int((y1_full + y2_full) / 2)

            if len(trajectory) > 0:
                prev_x, prev_y = trajectory[-1]
                dist = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)

                if dist < MAX_DISTANCE:
                    trajectory.append((cx, cy))
                    ball_detected = True
            else:
                trajectory.append((cx, cy))
                ball_detected = True

            if ball_detected:
                cx, cy = smooth_point(trajectory, SMOOTHING_WINDOW)
                x_centroid, y_centroid = cx, cy

                # Draw bounding box
                cv2.rectangle(frame,
                              (x1_full, y1_full),
                              (x2_full, y2_full),
                              (0, 255, 0), 2)

                # Draw centroid
                cv2.circle(frame,
                           (cx, cy),
                           5,
                           (255, 0, 0),
                           -1)

        # Draw trajectory
        for i in range(1, len(trajectory)):
            cv2.line(frame,
                     trajectory[i - 1],
                     trajectory[i],
                     (0, 0, 255),
                     2)

        visibility_flag = 1 if ball_detected else 0

        records.append({
            "frame": frame_index,
            "x": x_centroid,
            "y": y_centroid,
            "visible": visibility_flag
        })

        out.write(frame)

    cap.release()
    out.release()

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)

    print(f"Saved video to: {output_video_path}")
    print(f"Saved annotations to: {output_csv_path}")


def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    model = YOLO(MODEL_PATH)

    video_files = [
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    print(f"Found {len(video_files)} videos.")

    for video_path in video_files:
        process_video(video_path, model)

    print("\nAll videos processed successfully.")


if __name__ == "__main__":
    main()

