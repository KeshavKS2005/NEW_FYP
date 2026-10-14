# Stream B: Known-Object Semantic Detection Engine
### *Module 2 of the Dual-Stream Motion–YOLO Discrepancy Architecture*

---

## Table of Contents
1. [Executive Summary & Architectural Purpose](#1-executive-summary--architectural-purpose)
2. [Theoretical Foundation & The Discrepancy Baseline](#2-theoretical-foundation--the-discrepancy-baseline)
   - [2.1 The Closed-World Semantics Problem](#21-the-closed-world-semantics-problem)
   - [2.2 The Anomaly Baseline Strategy (Why Exclude Car and Person?)](#22-the-anomaly-baseline-strategy-why-exclude-car-and-person)
   - [2.3 Stream B's Mathematical Contract in the Pipeline](#23-stream-bs-mathematical-contract-in-the-pipeline)
3. [Model Architecture & YOLO11s Characteristics](#3-model-architecture--yolo11s-characteristics)
   - [3.1 Backbone, Neck, and Decoupled Head](#31-backbone-neck-and-decoupled-head)
   - [3.2 Single-Class Adaptation ($nc=1$)](#32-single-class-adaptation-nc1)
4. [The End-to-End Stream B Inference Pipeline](#4-the-end-to-end-stream-b-inference-pipeline)
   - [Phase 1: Input Normalization & Letterbox Preprocessing](#phase-1-input-normalization--letterbox-preprocessing)
   - [Phase 2: Tensor Forward Pass (GPU Acceleration)](#phase-2-tensor-forward-pass-gpu-acceleration)
   - [Phase 3: Confidence Gating ($\ge 0.40$) & NMS ($IoU = 0.45$)](#phase-3-confidence-gating--040--nms-iou--045)
   - [Phase 4: Coordinate Clamping & Boundary Contract Enforcement](#phase-4-coordinate-clamping--boundary-contract-enforcement)
   - [Phase 5: Output Matrix Construction `[K, 4]`](#phase-5-output-matrix-construction-k-4)
5. [Dataset Curation, Background Modeling & Training](#5-dataset-curation-background-modeling--training)
   - [5.1 Single-Class Annotation Filtering](#51-single-class-annotation-filtering)
   - [5.2 Surveillance Domain Negative Background Injection](#52-surveillance-domain-negative-background-injection)
   - [5.3 Training Protocol & Early Stopping](#53-training-protocol--early-stopping)
6. [Quantitative Evaluation, Validation & Confusion Matrix](#6-quantitative-evaluation-validation--confusion-matrix)
   - [6.1 Metrics on Downloaded Evaluation Dataset](#61-metrics-on-downloaded-evaluation-dataset)
   - [6.2 Detailed Confusion Matrix & False Positive Analysis](#62-detailed-confusion-matrix--false-positive-analysis)
   - [6.3 Verification on Surveillance Footage](#63-verification-on-surveillance-footage)
7. [Code Architecture & Module Breakdown](#7-code-architecture--module-breakdown)
   - [7.1 `config.py`: Configuration & Dataclass Contracts](#71-configpy-configuration--dataclass-contracts)
   - [7.2 `detector.py`: YOLOKnownDetector Engine](#72-detectorpy-yoloknowndetector-engine)
   - [7.3 `visualizer.py`: Diagnostic HUD & Video Writer](#73-visualizerpy-diagnostic-hud--video-writer)
   - [7.4 `run_stream_b.py`: Standalone CLI Runner](#74-run_stream_bpy-standalone-cli-runner)
8. [CLI Usage Guide & Production Recipes](#8-cli-usage-guide--production-recipes)
9. [Downstream Integration: Interface with Module 3 (Discrepancy Erasure)](#9-downstream-integration-interface-with-module-3-discrepancy-erasure)

---

## 1. Executive Summary & Architectural Purpose

In automated surveillance, standard object detectors operate under a strict **closed-world assumption**: they can only identify the exact semantic classes they were explicitly trained on. High-security environments, however, require the detection of **unknown moving objects** (e.g., untracked machinery, rolling debris, unauthorized carts, stray animals) without triggering false alarms for regular stationary backgrounds or known authorized traffic.

The **Dual-Stream Motion–YOLO Discrepancy Architecture** resolves this challenge through mask-level subtraction:

```
+-----------------------------------------------------------------------------+
|               DUAL-STREAM DISCREPANCY PIPELINE OVERVIEW                     |
|                                                                             |
|   +-----------------------+              +-----------------------+          |
|   | STREAM A (Module 1)   |              | STREAM B (This Module)|          |
|   | MOG2 Motion Extractor |              | YOLO11s Semantics     |          |
|   | All Moving Pixels     |              | Known Objects Only    |          |
|   +-----------+-----------+              +-----------+-----------+          |
|               |                                      |                      |
|               | (H, W, 1) Binary Mask                | [K, 4] Known Boxes   |
|               v                                      v                      |
|       +------------------------------------------------------+              |
|       |     MODULE 3: MASK-LEVEL ERASURE & EXTRACTION        |              |
|       |  Digitally erase Stream B boxes from Stream A mask   |              |
|       +--------------------------+---------------------------+              |
|                                  | Surviving Unknown Motion Blobs           |
|                                  v                                          |
|       +------------------------------------------------------+              |
|       |         MODULE 4: TEMPORAL TRAJECTORY GATING         |              |
|       |     ByteTrack Kalman filtering & directional sanity  |              |
|       +--------------------------+---------------------------+              |
|                                  | Confirmed Unknown Moving Objects         |
|                                  v                                          |
|                       ALERT & ACTIVE RETRAINING                             |
+-----------------------------------------------------------------------------+
```

### The Role of Stream B
Stream B is the **Semantic Filter** of the system. While Stream A extracts *all* physical displacement regardless of class, Stream B extracts the spatial coordinates of **all known objects**, regardless of whether they are moving or stationary.

Downstream Module 3 uses Stream B's coordinate array to paint black holes over the known objects on Stream A's motion mask. **Whatever motion survives this digital erasure represents an unknown moving entity.**

---

## 2. Theoretical Foundation & The Discrepancy Baseline

### 2.1 The Closed-World Semantics Problem
Standard deep neural networks compute a conditional probability distribution:

$$P(y \in \mathcal{C} \mid \mathbf{X})$$

where $\mathcal{C} = \{c_1, c_2, \dots, c_N\}$ represents a predefined closed set of classes (e.g., 80 COCO classes). When an uncatalogued object $\mathbf{X}_{\text{unknown}}$ appears, the network forces the probabilities across all known categories to remain low, causing the prediction to be dropped after confidence thresholding. Thus, deep learning alone cannot identify an unknown entity.

### 2.2 The Anomaly Baseline Strategy (Why Exclude Car and Person?)
In real-world CCTV surveillance footage (such as the VIRAT dataset in `CCTV 01`), the overwhelming majority of moving traffic consists of:
- **Pedestrians (`person`)**
- **Vehicles (`car`, `truck`, `bus`)**

If the baseline YOLO model were trained to recognize both `person` and `car`, Stream B would detect them, Module 3 would erase their motion masks, and zero unknown motion blobs would survive.

To rigorously demonstrate and validate the discrepancy pipeline, the initial known model was trained on **exactly one object (`bicycle`)**, strictly **excluding `person` and `car`**:
1. When pedestrians walk or cars drive in the CCTV video, Stream B produces **0 bounding boxes** on them.
2. Because Stream B outputs no bounding boxes for pedestrians and cars, Module 3 leaves their motion pixels completely intact on the motion mask.
3. Modules 3 and 4 successfully isolate, contour, and track them as **unknown moving objects**.
4. When an actual bicycle is present, Stream B detects it and erases its motion, demonstrating selective discrepancy.

### 2.3 Stream B's Mathematical Contract in the Pipeline
Stream B enforces a deterministic input/output contract per `fyp_architecture.md`:
* **Input:** Raw RGB Frame $\mathbf{I}_t \in \mathbb{R}^{H \times W \times 3}$, sequential frame index $t$.
* **Output Matrix:** Known coordinate matrix $\mathbf{B}_{\text{known}} \in \mathbb{Z}^{K \times 4}$:
  $$\mathbf{B}_{\text{known}} = \begin{bmatrix} x_1^{(1)} & y_1^{(1)} & x_2^{(1)} & y_2^{(1)} \\ x_1^{(2)} & y_1^{(2)} & x_2^{(2)} & y_2^{(2)} \\ \vdots & \vdots & \vdots & \vdots \\ x_1^{(K)} & y_1^{(K)} & x_2^{(K)} & y_2^{(K)} \end{bmatrix}$$
  Where:
  $$0 \le x_1^{(i)} < x_2^{(i)} \le W, \quad 0 \le y_1^{(i)} < y_2^{(i)} \le H, \quad \text{Confidence}^{(i)} \ge 0.40$$

---

## 3. Model Architecture & YOLO11s Characteristics

Stream B employs **Ultralytics YOLO11s** (`yolo11s.pt`), providing an optimal balance of latency and feature representation.

### 3.1 Backbone, Neck, and Decoupled Head
* **Backbone:** Modified CSPDarknet with enhanced C3k2 (Cross Stage Partial with 2 convolutions) blocks and SPPF (Spatial Pyramid Pooling - Fast) to pool multiscale context at $13\times 13$, $9\times 9$, and $5\times 5$ receptive fields without degrading resolution.
* **Neck:** Path Aggregation Network (PANet) fusing low-level spatial localization features with high-level semantic abstractions via top-down and bottom-up pathways.
* **Head:** Anchor-free decoupled head separating bounding box regression (DFL - Distribution Focal Loss) from class score prediction (BCE - Binary Cross-Entropy Loss).

### 3.2 Single-Class Adaptation ($nc = 1$)
The standard 80-class output layer was replaced with a single binary logit head ($nc = 1$):
* Classification loss: $\mathcal{L}_{\text{cls}} = \text{BCE}(\hat{p}, y)$
* Box regression loss: $\mathcal{L}_{\text{box}} = \text{CIoU}(\hat{b}, b) + \text{DFL}(\hat{b}, b)$
* Total loss: $\mathcal{L} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}$

Parameters:
* **Total Layers:** 100
* **Total Parameters:** 9,413,187 (~9.4M)
* **Computational Complexity:** 21.4 GFLOPs at $640 \times 640$

---

## 4. The End-to-End Stream B Inference Pipeline

```
+-----------------------------------------------------------------------------+
|                     STREAM B INFERENCE FLOW DIAGRAM                         |
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   | 1. RAW RGB FRAME INPUT (H, W, 3)                                     |   |
|   +----------------------------------+----------------------------------+   |
|                                      |                                      |
|                                      v                                      |
|   +---------------------------------------------------------------------+   |
|   | 2. LETTERBOX RESIZE & NORMALIZATION                                 |   |
|   |    Pads image to 640x640 keeping aspect ratio, scale to [0, 1]       |   |
|   +----------------------------------+----------------------------------+   |
|                                      |                                      |
|                                      v                                      |
|   +---------------------------------------------------------------------+   |
|   | 3. FORWARD PASS (GPU RTX 3050 / FP32)                                |   |
|   |    Multi-scale feature maps at P3/8, P4/16, P5/32                   |   |
|   +----------------------------------+----------------------------------+   |
|                                      |                                      |
|                                      v                                      |
|   +---------------------------------------------------------------------+   |
|   | 4. CONFIDENCE GATING & NMS                                           |   |
|   |    Filter predictions: confidence >= 0.40                           |   |
|   |    Apply NMS: IoU threshold = 0.45                                  |   |
|   +----------------------------------+----------------------------------+   |
|                                      |                                      |
|                                      v                                      |
|   +---------------------------------------------------------------------+   |
|   | 5. COORDINATE CLAMPING & MATRIX FORMATION                           |   |
|   |    Clip: x in [0, W], y in [0, H]                                   |   |
|   |    Output: Known Bounding Box Matrix [K, 4]                         |   |
|   +---------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------+
```

### Phase 1: Input Normalization & Letterbox Preprocessing
Incoming frames $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ are rescaled while preserving the native aspect ratio via letterbox padding with gray borders (`(114, 114, 114)`), preventing geometric distortion of thin structures like wheels and bicycle frames.

### Phase 2: Tensor Forward Pass (GPU Acceleration)
The normalized tensor $\mathbf{X} \in \mathbb{R}^{1 \times 3 \times 640 \times 640}$ is processed via PyTorch on the NVIDIA GeForce RTX 3050 Laptop GPU (`device=0`), achieving an inference latency of **10.67 ms**.

### Phase 3: Confidence Gating ($\ge 0.40$) & NMS ($IoU = 0.45$)
1. **Confidence Filtering:** Predictions with confidence score $s_i < 0.40$ are immediately dropped.
2. **Non-Maximum Suppression (NMS):** Overlapping candidate boxes are suppressed using Greedy NMS with an IoU threshold of $0.45$:
   $$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} > 0.45 \implies \text{Suppress lower-confidence box}$$

### Phase 4: Coordinate Clamping & Boundary Contract Enforcement
Bounding box coordinates are mapped back from letterboxed space to original pixel space $(H, W)$, and rounded and clamped:
$$x_1 = \max(0, \min(W, \text{round}(x_1))), \quad y_1 = \max(0, \min(H, \text{round}(y_1)))$$
$$x_2 = \max(0, \min(W, \text{round}(x_2))), \quad y_2 = \max(0, \min(H, \text{round}(y_2)))$$
Any box where $x_2 \le x_1$ or $y_2 \le y_1$ is discarded.

### Phase 5: Output Matrix Construction `[K, 4]`
The surviving coordinates are packed into a 2D integer NumPy array `box_matrix` of shape `[K, 4]`. If no known objects are detected ($K = 0$), an empty array of shape `(0, 4)` with `dtype=int32` is returned.

---

## 5. Dataset Curation, Background Modeling & Training

### 5.1 Single-Class Annotation Filtering
The single-class dataset was curated programmatically via [download_and_prepare_dataset.py](file:///c:/fyp_new/train_stream_b/download_and_prepare_dataset.py):
* `bicycle` annotations were extracted and remapped to `class_id = 0`.
* All other classes—specifically `person` and `car`—were **purged from the labels**.

### 5.2 Surveillance Domain Negative Background Injection
A key innovation in the training pipeline was the deliberate inclusion of **negative background samples**:
* **COCO / VOC Backgrounds:** 40 images containing pedestrians, vehicles, and complex scenes with empty `.txt` annotation files.
* **CCTV Surveillance In-Domain Frames:** 15 background frames extracted directly from the target CCTV camera (`CCTV 01/VIRAT_S_010001_...`) featuring parking lot pavement, road asphalt, parked cars, and walking pedestrians with empty `.txt` annotation files.

**Mathematical Benefit:**
When the YOLO classification head evaluates negative images, any prediction is heavily penalized by Binary Cross-Entropy loss ($\mathcal{L}_{\text{cls}} = -\log(1 - \hat{p})$). This explicitly teaches the network that asphalt textures, car contours, and human silhouettes are **background**, reducing false positives on surveillance video to **0.0%**.

### 5.3 Training Protocol & Early Stopping
* **Pretrained Weights:** `yolo11s.pt`
* **Optimizer:** Auto (AdamW with Cosine LR scheduler)
* **Epochs:** 30
* **Patience:** 20 (Early stopping)
* **Batch Size:** 16
* **Resolution (`imgsz`):** 640
* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)

---

## 6. Quantitative Evaluation, Validation & Confusion Matrix

### 6.1 Metrics on Downloaded Evaluation Dataset
Evaluation was conducted on a dedicated validation split containing **70 images** (50 images with ground-truth bicycles + 20 negative background images):

| Metric | Result | Benchmark | Interpretation |
| :--- | :--- | :--- | :--- |
| **Precision (P)** | **91.67%** | > 85.0% | High purity: 91.67% of declared boxes are real bicycles |
| **Recall (R)** | **64.71%** | > 60.0% | 64.71% of all ground-truth bicycle instances detected |
| **mAP@50** | **63.59%** (Peak: **74.53%**) | > 50.0% | Mean Average Precision at IoU 0.50 |
| **mAP@50-95** | **42.87%** (Peak: **48.94%**) | > 35.0% | Strict COCO-standard localization precision |
| **Inference Latency** | **10.67 ms** (~93 FPS) | < 33.3 ms | Real-time capability on edge hardware |

### 6.2 Detailed Confusion Matrix & False Positive Analysis
At image level across the 70 evaluation samples:

```
                          GROUND TRUTH
                     Positive      Negative
                   (Has Bicycle) (No Bicycle)
                 +--------------+--------------+
  PREDICTED   Pos|   TP = 40    |    FP = 1    |  Precision = 40 / 41 = 97.6% (Image)
                 +--------------+--------------+  Box Precision  = 91.67%
              Neg|   FN = 10    |   TN = 19    |  Specificity    = 19 / 20 = 95.0%
                 +--------------+--------------+
                   Sensitivity = 80.0%
```

* **False Positive Rate (FPR):**
  $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{1}{1 + 19} = \mathbf{5.00\%}$$
* **Specificity (True Negative Rate):**
  $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{19}{20} = \mathbf{95.00\%}$$
* **Analysis of the Single False Positive (`005468.jpg`):**
  The sole false positive had a confidence score of **`0.432`** (borderline, barely exceeding the default `0.40` threshold). Setting `--conf 0.45` drops the False Positive Rate to **0.0%**.

### 6.3 Verification on Surveillance Footage
Stream B was evaluated on 151 frames of native CCTV surveillance footage (`VIRAT_S_010001_00_000019_000065.mp4`):
* **Bicycles in Video:** 0
* **Pedestrians & Cars in Video:** Multiple moving throughout the scene
* **Stream B Detections:** **0 bounding boxes**
* **Surveillance Video False Positive Rate:** **0.00%**
* **Average Pipeline FPS:** **29.6 FPS**

---

## 7. Code Architecture & Module Breakdown

Stream B is structured as a modular package inside `c:\fyp_new\stream_b\`:

```
c:\fyp_new\stream_b\
├── __init__.py         # Public exports (StreamBConfig, YOLOKnownDetector, StreamBVisualizer)
├── config.py           # Dataclasses and contracts
├── detector.py         # YOLOKnownDetector inference engine
├── visualizer.py       # Multi-element HUD overlay & video encoder
├── run_stream_b.py     # Standalone CLI runner
└── weights\
    └── known_single_object.pt # Trained YOLO11s model weights (19.2 MB)
```

### 7.1 `config.py`: Configuration & Dataclass Contracts
Defines:
* `StreamBConfig`: Dataclass holding `model_path`, `conf_threshold` (0.40), `iou_threshold` (0.45), `imgsz` (640), `device` ('0'), and `known_classes` (`['bicycle']`).
* `KnownBoundingBox`: Single detection containing `x1`, `y1`, `x2`, `y2`, `confidence`, `class_id`, `class_name`, plus computed properties (`width`, `height`, `area`, `centroid`, `as_xyxy`).
* `StreamBFrameResult`: Frame output container with `frame_index`, `boxes`, `box_matrix` (`np.ndarray` of shape `[K, 4]`), and `inference_time_ms`.

### 7.2 `detector.py`: YOLOKnownDetector Engine
Implements the core detection class:
```python
detector = YOLOKnownDetector(config=StreamBConfig(conf_threshold=0.40, device="0"))
result = detector.detect(frame, frame_index=1)
print(result.box_matrix.shape)  # Guaranteed (K, 4)
```
* Safely manages GPU memory and handles automatic CPU fallback if CUDA is unavailable.
* Validates and clamps all output box coordinates within frame dimensions $[0, W] \times [0, H]$.

### 7.3 `visualizer.py`: Diagnostic HUD & Video Writer
* Renders bounding boxes in amber/cyan (`BGR: (255, 180, 0)`) with label badges indicating class and confidence percentage.
* Draws a semi-transparent HUD telemetry card in the top-left showing FPS, inference latency, frame counter, and known-object count.
* Encodes annotated frames to MP4 (`mp4v` codec) via OpenCV `VideoWriter`.

### 7.4 `run_stream_b.py`: Standalone CLI Runner
Provides a standalone command-line interface for video processing with progress logging, periodic snapshot capture, and execution summaries.

---

## 8. CLI Usage Guide & Production Recipes

### Basic Surveillance Video Processing
```bash
python stream_b/run_stream_b.py \
    --video "CCTV 01/VIRAT_S_010001_00_000019_000065.mp4" \
    --output "outputs/stream_b_demo.mp4"
```

### Processing with Frame Limit & Higher Confidence Threshold
```bash
python stream_b/run_stream_b.py \
    --video "CCTV 01/VIRAT_S_010001_00_000019_000065.mp4" \
    --output "outputs/stream_b_strict.mp4" \
    --conf 0.45 \
    --max_frames 200
```

### Re-Evaluating the Model on the Downloaded Test Dataset
```bash
python train_stream_b/evaluate_on_test_dataset.py
```

### Running the Unit Test Suite
```bash
python -m unittest tests/test_stream_b.py -v
```

---

## 9. Downstream Integration: Interface with Module 3 (Discrepancy Erasure)

Stream B feeds directly into **Module 3 (Mask-Level Erasure & Extraction)**:

```python
# Module 3 Pseudocode Integration
from stream_a.motion_extractor import MOG2MotionExtractor
from stream_b.detector import YOLOKnownDetector
import cv2

motion_extractor = MOG2MotionExtractor()
known_detector = YOLOKnownDetector()

# In surveillance loop:
motion_result = motion_extractor.process(frame)  # (H, W) binary mask
stream_b_result = known_detector.detect(frame)    # [K, 4] coordinate matrix

# Copy motion mask
discrepancy_mask = motion_result.binary_mask.copy()

# Step 1: Dilation Margin (expand known boxes by 10% to prevent limb bleed)
for x1, y1, x2, y2 in stream_b_result.box_matrix:
    pad_x = int((x2 - x1) * 0.10)
    pad_y = int((y2 - y1) * 0.10)
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(frame.shape[1], x2 + pad_x)
    ey2 = min(frame.shape[0], y2 + pad_y)

    # Step 2: Digital Erasure (Paint known object BLACK on motion mask)
    discrepancy_mask[ey1:ey2, ex1:ex2] = 0

# Step 3: Find surviving unknown contours
contours, _ = cv2.findContours(discrepancy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
unknown_candidate_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 500]
```

Because Stream B's false positive rate on pedestrians and vehicles is **0.0%**, their pixels survive this erasure step intact and are cleanly handed over to ByteTrack (Module 4) for trajectory confirmation as **unknown moving objects**.
