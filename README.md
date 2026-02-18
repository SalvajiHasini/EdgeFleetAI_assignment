Overview

This project implements an end-to-end system for detecting the centroid of a cricket ball in broadcast videos captured from a single static camera.

The system performs the following tasks:

Detects the cricket ball centroid in each visible frame

Generates a per-frame CSV annotation file

Produces a processed video with the ball trajectory overlayed

Provides fully reproducible training and inference pipelines

Multiple modelling strategies were explored during development:

Baseline – Pretrained YOLO (COCO sports ball class)

Spatial Filtering using Center ROI

Hybrid Detection + Tracking (CSRT + motion constraints)

Domain-Specific Fine-Tuning (Final Model)

The final submitted system uses a fine-tuned YOLOv8 model combined with center ROI filtering and temporal constraints for stable trajectory generation.

Key Challenges

Very small object size (10–25 pixels in broadcast frames)

Motion blur during fast bowling

White ball detection under bright lighting

False positives from scoreboard logos and pitch markings

Occlusion by batsman pads and bat

Domain shift between static datasets and broadcast footage

Video compression artifacts

Repository Structure

code/
Training, inference, tracking, and utility scripts

annotations/
Generated per-frame CSV files

results/
Processed output videos with trajectory overlays

models/
Trained model weights (if file size allows)

report.pdf
Detailed technical documentation

requirements.txt
Dependency list

README.md
Project overview and usage instructions

Environment Setup

Create Conda Environment

conda create -n cricket python=3.10
conda activate cricket

Install Dependencies

pip install -r requirements.txt

Required packages include:

ultralytics

opencv-contrib-python

numpy

pandas

tqdm

Training – Fine-Tuned Model

Training was performed using YOLOv8l initialized from pretrained weights.

Configuration:

Model: YOLOv8l

Image size: 1280

Epochs: 50

Batch size: 8

Dataset: Cricket-specific annotated dataset

To train:

python code/train.py

Best model weights are generated at:

runs/detect/cricket_ball_finetune/weights/best.pt

Move the best model to:

models/best_finetuned.pt

Inference – Final Submission Version

Run the final pipeline:

python code/infer_final_center_roi.py

Pipeline steps:

Extract center pitch ROI

Run fine-tuned YOLO detection

Apply bounding box area filtering

Apply distance gating between consecutive frames

Apply moving average smoothing

Draw centroid and trajectory

Save processed video and CSV annotations

Annotation Output Format

CSV format:

frame,x,y,visible
0,512,298,1
1,518,305,1
2,-1,-1,0

Where:

x, y → centroid coordinates

visible → 1 if detected, 0 if not detected

Modelling Evolution

Phase 1 – Direct Pretrained YOLO

Used COCO sports ball class

High false positives from logos

Weak white ball detection

Inconsistent small object performance

Phase 2 – Center ROI Filtering

Detection limited to pitch area

Reduced scoreboard false positives

Still struggled with occlusion

Phase 3 – Hybrid Detection + Tracking

YOLO for initialization

CSRT tracker for temporal continuity

Distance gating for motion consistency

Reduced trajectory jumps

Still limited by detection quality

Phase 4 – Fine-Tuning (Final Model)

Trained YOLOv8l on cricket-specific dataset

Improved detection for red and white balls

Reduced logo confusion

Stabilized centroid tracking

Training Performance Summary

Validation Metrics:

mAP@0.5 ≈ 0.97

High precision-recall performance

Optimal confidence threshold ≈ 0.35–0.45

Fine-tuning significantly improved detection robustness compared to the pretrained baseline.

See report.pdf for:

Confusion matrix

Precision-recall curves

F1-confidence curves

Example validation predictions

Final System Architecture

Input Video
↓
Center ROI Extraction
↓
Fine-Tuned YOLO Detection
↓
Area Filtering
↓
Distance Gating
↓
Trajectory Smoothing
↓
CSV Export + Processed Video

Large Files (Model & Videos)

Due to GitHub file size limitations, large files are hosted externally.

Model Weights:
Insert your Google Drive link here

Processed Output Videos:
Insert your Google Drive link here

Known Limitations

Performance may degrade for extremely low-resolution footage

Severe occlusion can cause missed detections

Extreme motion blur frames remain challenging

Conclusion

Through iterative experimentation and domain-specific fine-tuning, the final system significantly improves robustness compared to the pretrained baseline.

The final pipeline is:

Modular

Reproducible

Stable

Suitable for sports analytics applications
