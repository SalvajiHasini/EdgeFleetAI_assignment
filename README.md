# EdgeFleetAI_assignment
# Cricket Ball Centroid Detection & Trajectory Tracking

## 📌 Overview

This project implements an end-to-end system for detecting the centroid of a cricket ball in broadcast videos captured from a single static camera. The system outputs:

- Per-frame annotation file (CSV)
- Processed video with centroid and trajectory overlay
- Fully reproducible training and inference pipeline

Two approaches are implemented:

1. **Baseline (Pretrained YOLO + Hybrid Tracking)**
2. **Fine-Tuned YOLO + Tracking (Improved Performance)**

---

## 🏗 Repository Structure

code/ # Training, inference, tracking modules
annotations/ # Per-frame CSV files
results/ # Processed output videos
models/ # Trained model weights
README.md
requirements.txt
report.pdf


## ⚙️ Environment Setup

### Using Conda

```bash
conda create -n cricket python=3.10
conda activate cricket
pip install -r requirements.txt
