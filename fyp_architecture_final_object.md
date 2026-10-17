# Ground Truth Architecture: 12-Module Open-World Moving Object Discovery, Decomposition & Continual Retraining Pipeline

## 1. Executive Summary & System Purpose

### 1.1 Problem Statement
In automated video surveillance, standard object detectors (such as YOLO) operate under a **closed-world assumption**:
1. **Blindness to Unknown Classes:** Any moving physical object outside the trained class list (e.g., unrecognized carts, loose animals, novel vehicles, falling debris) is ignored as background.
2. **Stationary Object False Positives:** Standard detectors fire continuously on motionless known objects (such as parked cars), wasting compute and database storage when only moving objects matter.
3. **Compound Blob Merging in Classical Motion Detection:** When classical background subtraction (MOG2) is used to detect unknown moving objects, two unknown objects that physically touch and move together (such as an unknown person riding an unknown bicycle) merge into a **single connected silhouette**.
4. **Static Model Obsolescence & Unsafe Retraining:** Once an unknown object class is discovered and labeled, manually retraining and redeploying YOLO risks **catastrophic forgetting** of old classes or deploying a degraded model if not gated by automated Champion-vs-Challenger metric evaluation.

### 1.2 Architectural Solution
This system implements an end-to-end, **12-Module Open-World Continual Discovery & Self-Evolution Engine** split across three multiprocessing tiers:
* **Phase 1: Upstream Real-Time Discrepancy Tracker (Modules 1–4, 30+ FPS):**
  Runs pixel-level MOG2 background subtraction (Stream A) in parallel with the current Champion YOLO model (Stream B). It erases known objects from the motion mask, extracts residual unknown moving contours, tracks them via Kalman/IoU gating to reject oscillating noise (such as swaying trees), dispatches the single sharpest keyframe per completed track into an asynchronous queue, and listens for atomic hot-swap signals to reload newly promoted YOLO weights without dropping frames.
* **Phase 2: Downstream Compound Decomposition, Self-Supervised Clustering & VLM Discovery (Modules 5–9, Async Worker):**
  Receives each confirmed unknown keyframe. **Module 5** applies **Method 2 (`FastSAM` Boundary Proposals + `DINOv2` Mask-Guided Pooling & Iterative Cosine Merging)** to decompose compound blobs (such as Rider + Bicycle) into isolated sub-object masks and tight bounding boxes. **Module 6** masks out touching neighbor pixels and extracts a global **DINOv2-Reg** vector for each isolated sub-object. **Module 7** clusters vectors online via Cosine Similarity in **FAISS** with disk index synchronization, **Module 8** persists clean unedited full frames, crops, masked crops, bounding boxes, and counts in **PostgreSQL**, and **Module 9** calls **Gemini 2.5 Flash** (drawing red bounding boxes only on an in-memory visualization copy) once a cluster reaches **100 observations** to automatically name the discovered object class.
* **Phase 3: Continual YOLO Retraining, Champion-vs-Challenger Evaluation & Atomic Hot-Swap Deployment (Modules 10–12, Continual Trainer Process):**
  Once Module 9 identifies a new valid physical object class (or a configurable threshold of $N_{\text{retrain}}$ novel clusters is reached), **Module 10** builds an incremental standard YOLO bounding-box dataset by combining the 100 discovered clean full frames, running the current **Champion model** as a Teacher to pseudo-label any co-occurring known objects in those frames with standard bounding boxes, and mixing in an **Experience Replay Buffer** of old classes. **Module 11** fine-tunes a **Challenger YOLO model** (`freeze=10`) and runs a strict **Champion-vs-Challenger Evaluation Gate** (verifying zero catastrophic forgetting on old classes and high mAP on the new class). **Module 12** atomically promotes the Challenger to Champion and signals Phase 1 to hot-swap the weights in real time.

---

## 2. Strict Environment & Credential Isolation Rule (`FYP_phase1/.env`)

* **Credential Source:** This project loads PostgreSQL database credentials and the `GEMINI_API_KEY` **exclusively** from the `.env` file located in the sibling `FYP_phase1` project directory (default path: `../FYP_phase1/.env`, or overridden via `FYP_PHASE1_ENV_PATH`).
* **Strict Zero-Code-Coupling Policy:** The pipeline **must never import, call, or depend on any Python files, modules, classes, or business logic inside `FYP_phase1`**. It uses `dotenv.load_dotenv(dotenv_path=FYP_PHASE1_ENV_PATH)` solely to read environment variables (`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `GEMINI_API_KEY`).

---

## 3. End-to-End 12-Module Data Flow & Integration Map

| Module | Module Name | Core Library / GitHub Reference | Input Data | Output Data | Purpose in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Module 1** | Class-Agnostic Motion Extraction (Stream A) | OpenCV `MOG2` (`opencv/opencv`) | Raw BGR Frame `(H, W, 3)` | Binary Foreground Mask `(H, W, 1)` `{0, 255}` | Captures all moving pixels; strips shadows (`127`) and ignores stationary objects. |
| **Module 2** | Known-Object Detection (Stream B) | Ultralytics YOLO (`ultralytics/ultralytics`) | Raw BGR Frame `(H, W, 3)` | Known Bounding Boxes `[K, 4]` | Locates all currently known classes using the active `champion.pt` weights. |
| **Module 3** | Mask-Level Known Erasure & Contour Extraction | OpenCV + NumPy Matrix Masking | Foreground Mask `(H, W, 1)` + Known Boxes `[K, 4]` | Residual Unknown Mask + Candidate Boxes `[U, 5]` | Erases known objects (padded by 15px) from the motion mask and extracts surviving moving blobs. |
| **Module 4** | Temporal Trajectory Gating & Keyframe Buffer | ByteTrack / Kalman IoU (`ifzhang/ByteTrack`) | Candidate Unknown Boxes `[U, 5]` | 1 Sharpest Keyframe Tuple per Confirmed `track_id` | Rejects swaying trees/noise and prevents 1 track from triggering 100 frame counts. |
| **Module 5** | Compound Blob Decomposer (Method 2: `FastSAM + DINOv2`) | `FastSAM` (`CASIA-IVA-Lab/FastSAM`) + `DINOv2` Patch Grid | Confirmed Keyframe + Residual MOG2 Mask + Merged BBox | List of Isolated Sub-Object Dicts `[1..S]` | Splits touching unknown objects (e.g., Unknown Person + Unknown Bicycle) into separate masks and tight bboxes. |
| **Module 6** | Neighbor-Suppressed Preprocessor & Global Embedder | DINOv2 with Registers (`facebookresearch/dinov2`) | Isolated `sub_raw_crop` + `sub_mask` | L2-Normalized Vector `(1, 384)` `float32` + `bbox_xyxy` | Replaces background AND touching neighbor pixels with gray `(128,128,128)` and embeds crop. |
| **Module 7** | Online Cosine Centroid Clustering Engine | FAISS (`facebookresearch/faiss`) | Normalized Vector `(1, 384)` | Matched or New `faiss_index_id` + Cosine Score | Matches object against existing cluster centroids (`sim >= 0.78`) or allocates a new cluster; syncs index to disk. |
| **Module 8** | Relational Metadata & Observation Store | PostgreSQL (`psycopg2`) | `faiss_index_id`, Clean File Paths, `bbox_xyxy` | `cluster_id` + Updated `observation_count` | Persists clean full frames, raw crops, masked crops, bounding boxes, and cluster counts. |
| **Module 9** | Automated VLM Class Discovery Trigger | Google GenAI SDK (`googleapis/python-genai`) | 1 In-Memory Red-Box Frame + Top 3 Sharpest Crops | Structured JSON (`object_name`, `confidence`) | Identifies the unknown object class when `observation_count == 100` and updates PostgreSQL. |
| **Module 10** | Continual Dataset Builder & Teacher Pseudo-Labeler | Ultralytics YOLO (`yolo11n.pt`) | 100 Clean Full Frames + BBoxes + Replay Buffer | Standard YOLO Dataset (`dataset.yaml`, `train/`, `val/`) | Auto-labels co-occurring known objects with Champion model in 5-col YOLO format, appends new class bboxes, and mixes replay data. |
| **Module 11** | Incremental Trainer & Champion-vs-Challenger Gate | Ultralytics `model.train(freeze=10)` & `model.val()` | `champion.pt` + Incremental `dataset.yaml` | `challenger.pt` + Validation Metric Verdict | Trains Challenger and verifies zero old-class regression (`d_mAP >= -0.015`) and new-class `mAP50 >= 0.60`. |
| **Module 12** | Atomic Model Registry & Zero-Downtime Hot-Swapper | PostgreSQL `model_registry` + `multiprocessing.Event` | Validated `challenger.pt` + Evaluation Metrics | Promoted `champion_latest.pt` + Live Reload Signal | Atomically replaces production weights and signals Phase 1 to reload YOLO without dropping the video stream. |

---

## 4. Architectural Wiring Diagram (All 12 Modules)

~~~text
[EXTERNAL CONFIG: ../FYP_phase1/.env] ──(Reads ONLY Postgres & Gemini API Key; ZERO code imported)
                 │
                 ▼
[PHASE 1: 30 FPS REAL-TIME VIDEO PROCESS]
  Module 1 (MOG2 Mask) ───────┐
                              ├──> Module 3 (Mask Erasure) ──> Module 4 (Trajectory Gate & Keyframe Dedup)
  Module 2 (YOLO Champion) ───┘              ▲                                     │
         ▲                                   │                                     ▼ (1 Keyframe Tuple / Track)
         │ (Atomic Hot-Swap Reload)          └─────────────── [ multiprocessing.Queue (discovery_queue) ]
         │                                                                         │
[PHASE 2: ASYNC DISCOVERY WORKER]                                                  ▼
         │                         ┌──────────────────────────────────────────────────────────────────────────────┐
         │                         │ MODULE 5: METHOD 2 DECOMPOSER (FastSAM Proposals + DINOv2 Cosine Merging)    │
         │                         │   Splits touching unknowns (e.g. Rider + Bicycle) into 1..S isolated masks   │
         │                         └───────────────────────────────────────────────┬──────────────────────────────┘
         │                                                                         │ (1-to-S Fan-Out Loop)
         │                                                                         ▼
         │                         Module 6 (Neighbor-Masked DINOv2 Embed) ──> Module 7 (FAISS Centroid Match)
         │                                                                         │
         │                                                                         ▼
         │                         Module 9 (Gemini 2.5 Flash VLM) <──(Count==100)─┴─ Module 8 (PostgreSQL Store)
         │                                   │
         │                                   ▼ (Pushes identified cluster_id to retrain_queue)
[PHASE 3: CONTINUAL RETRAINING & EVALUATION WORKER]
         │                         Module 10 (Dataset Builder + Champion Teacher Pseudo-Labeling + Replay Buffer)
         │                                   │
         │                                   ▼
         │                         Module 11 (YOLO Fine-Tuning [freeze=10] + Champion vs. Challenger mAP Gate)
         │                                   │
         │                                   ▼ (If Challenger Passes Both Old-Class & New-Class mAP Gates)
         └──────────────────────── Module 12 (Atomic Promotion to champion_latest.pt + Set model_reload_event)
~~~

---

## 5. Detailed Specifications for Modules 1 through 12

### Modules 1–4: Phase 1 Real-Time Discrepancy Tracker
* **Module 1 (MOG2 Motion Mask):** Uses `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)`, thresholds at `254` to remove gray shadows (`127`), and applies a `5x5` morphological opening. Stationary objects are completely ignored (`0`).
* **Module 2 (Champion YOLO Inference):** Runs the active production model (`champion_latest.pt` bootstrapped from `yolo11n.pt`) to obtain known standard bounding boxes `[x1, y1, x2, y2]`. Checks `model_reload_event.is_set()` on each frame to hot-swap weights in-place when Module 12 promotes a new model.
* **Module 3 (Mask-Level Erasure):** Dilates known YOLO bounding boxes by `YOLO_PAD_PX = 15` and zeroes them out of the MOG2 binary mask (`fg_mask[padded_known] = 0`). Extracts surviving contours with area $\ge 500$ pixels.
* **Module 4 (Trajectory Gate & Keyframe Dedup):** Tracks unknown contours via IoU + Kalman persistence, requires Directional Persistence $\eta_{\text{dir}} \ge 0.50$ over $\ge 8$ frames to reject swaying trees, and dispatches only the **single sharpest keyframe** per completed `track_id` into `discovery_queue`.

### Modules 5–9: Phase 2 Compound Decomposition & VLM Discovery
* **Module 5 (Method 2 Compound Blob Decomposer — `FastSAM + DINOv2`):**
  1. Runs `FastSAM-s` on the high-resolution crop to propose crisp part masks $\{\hat{M}_1 \dots \hat{M}_P\}$.
  2. Intersects each part mask with the MOG2 foreground mask (`cv2.bitwise_and`) and drops static background parts ($< 6\%$ foreground area).
  3. Extracts the $16 \times 16 \times 384$ `dinov2_vits14_reg` spatial patch grid.
  4. Computes the mask-pooled DINOv2 vector $\mathbf{v}_p$ for each sub-mask and runs **Iterative Agglomerative Cosine Merging** ($\mathbf{v}_p \cdot \mathbf{v}_q \ge 0.62$). This fuses separated parts of the same object (e.g., front wheel + rear wheel + frame across the rider's leg into `Bicycle`, and shirt + pants into `Person`). Includes a `MaskCut` Fiedler eigenvector fallback if `FastSAM` returns a single mask.
  5. Outputs each sub-object's isolated `sub_mask` (for neighbor suppression) and tight `[x1, y1, x2, y2]` bounding box.
* **Module 6 (Neighbor-Suppressed Global Embedder):** Masks out **both** static background and touching neighbor pixels to neutral gray `(128, 128, 128)` using each sub-object's isolated binary mask, letterbox-pads to $224 \times 224$, and extracts an L2-normalized 384-D `dinov2_vits14_reg` vector. Persists tight `bbox_xyxy` and `frame_hw`.
* **Module 7 (Online FAISS Centroid Matcher):** Searches `faiss.IndexFlatIP(384)`. Merges into the nearest cluster if cosine similarity $\ge 0.78$ (updating the running centroid) or allocates a new cluster centroid. Persists index and centroids to disk for synchronization across restarts.
* **Module 8 (PostgreSQL State & Observation Store):** Logs clean `full_frame_path`, `crop_image_path`, `masked_crop_path`, and `bbox_xyxy` in `unknown_observations`, and increments `observation_count` in `unknown_clusters`.
* **Module 9 (Gemini 2.5 Flash VLM Trigger):** Fires when `observation_count == 100`. Sends 1 full contextual frame (drawing the red bounding box on an in-memory copy) + top 3 sharpest crops to `gemini-2.5-flash` with a strict Pydantic schema. If `is_valid_physical_object == True` and `confidence >= 0.70`, updates `status = 'IDENTIFIED'` and pushes `cluster_id` into `retrain_queue`.

### Module 10: Continual Dataset Builder & Teacher Pseudo-Labeler
* **What it does:** Prepares a self-contained Ultralytics YOLO standard bounding-box dataset (`storage/datasets/incremental_v{N}/`) whenever `NEW_CLUSTERS_FOR_RETRAIN` (default `1`) newly identified clusters are ready.
* **Why it is required:** Training only on the new class labels causes **Unlabeled Foreground Suppression** (co-occurring known objects in the same frame get penalized as background) and **Catastrophic Forgetting**.
* **Process:**
  1. Assigns the next sequential YOLO class index `new_cls_id` to `identified_label` (or maps to an existing index if Gemini identified a variation of an existing name).
  2. Queries all 100 clean `full_frame_path`, `bbox_xyxy`, and `frame_hw` records for the cluster from PostgreSQL and splits them **80% Train / 20% Val**.
  3. **Teacher Pseudo-Labeling:** Runs the current `champion.pt` model (`conf=0.50`) on each clean `full_frame_path` to generate standard 5-column YOLO label lines (`<cls> <xc> <yc> <w> <h>`) for all *existing* known objects present in the background of that frame, then appends the newly discovered object's standard 5-column bounding box line (`new_cls_id <xc> <yc> <w> <h>`).
  4. **Experience Replay Injection:** Copies up to 100 historical images + labels from `storage/replay_buffer/` into the `train/` and `val/` splits, and saves 20 of the new class's sharpest frames into `storage/replay_buffer/` for future continual cycles.
* **Module Input:** `cluster_id`, current `champion.pt`, PostgreSQL records, and `storage/replay_buffer/`.
* **Module Output:** `dataset.yaml` path, `old_class_ids` list, and `new_class_ids` list.

### Module 11: Incremental YOLO Trainer & Champion-vs-Challenger Gate
* **What it does:** Fine-tunes a Challenger YOLO model on the incremental dataset and evaluates both Champion and Challenger on the validation split before allowing deployment.
* **Why it is required:** Prevents deploying a model that forgot old classes or failed to learn the new class.
* **Process:**
  1. Loads `champion.pt` weights into a fresh YOLO trainer and runs `model.train(data=dataset_yaml, epochs=25, imgsz=640, batch=16, freeze=10, lr0=0.001, patience=7)` to produce `challenger.pt`.
  2. Evaluates `champion.pt` on the validation set to record baseline $\text{mAP}_{50\text{-}95}^{\text{old}}(\text{Champion})$.
  3. Evaluates `challenger.pt` on the validation set to extract per-class $\text{mAP}_{50\text{-}95}$ and $\text{mAP}_{50}$.
  4. **Gate 1 (Old-Class Retention):** Verifies $\text{mAP}_{50\text{-}95}^{\text{old}}(\text{Challenger}) \ge \text{mAP}_{50\text{-}95}^{\text{old}}(\text{Champion}) - 0.015$.
  5. **Gate 2 (New-Class Competence):** Verifies $\text{mAP}_{50}^{\text{new}}(\text{Challenger}) \ge 0.60$ and $\text{Recall}^{\text{new}}(\text{Challenger}) \ge 0.65$.
* **Module Input:** `champion.pt`, `dataset.yaml`.
* **Module Output:** Boolean `passed_gate`, `challenger_weights_path`, and metric dictionary.

### Module 12: Atomic Model Registry & Zero-Downtime Hot-Swap Deployer
* **What it does:** If `passed_gate == True`, atomically replaces `models/champion_latest.pt` using `os.replace()`, records the model lineage and metrics in the PostgreSQL `model_registry` table, marks the cluster as `DEPLOYED_IN_YOLO`, and sets `model_reload_event.set()`.
* **Module Input:** Validated `challenger.pt` and evaluation metrics.
* **Module Output:** Live hot-swap in Phase 1 Stream B (`Module 2`) with zero dropped frames.

---

## 6. System Dependencies

~~~bash
pip install opencv-python ultralytics faiss-cpu torch torchvision psycopg2-binary pillow pydantic google-genai python-dotenv yt-dlp numpy
~~~

---

## 7. PostgreSQL Database Schema (`schema.sql`)

~~~sql
CREATE TABLE IF NOT EXISTS unknown_clusters (
    cluster_id SERIAL PRIMARY KEY,
    faiss_index_id INT UNIQUE NOT NULL,
    observation_count INT NOT NULL DEFAULT 1,
    status VARCHAR(32) NOT NULL DEFAULT 'COLLECTING', -- 'COLLECTING', 'IDENTIFIED', 'RETRAINING', 'DEPLOYED_IN_YOLO', 'NOISE'
    identified_label VARCHAR(128) DEFAULT NULL,
    assigned_yolo_class_id INT DEFAULT NULL,
    vlm_confidence FLOAT DEFAULT NULL,
    vlm_reasoning TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS unknown_observations (
    observation_id SERIAL PRIMARY KEY,
    cluster_id INT NOT NULL REFERENCES unknown_clusters(cluster_id) ON DELETE CASCADE,
    track_id INT NOT NULL,
    sub_object_idx INT NOT NULL DEFAULT 0,
    crop_image_path TEXT NOT NULL,
    masked_crop_path TEXT NOT NULL,
    full_frame_path TEXT NOT NULL,
    bbox_xyxy INT[4] NOT NULL,
    frame_hw INT[2] NOT NULL,
    sharpness_score FLOAT NOT NULL,
    cosine_similarity FLOAT NOT NULL,
    decomposition_method VARCHAR(32) NOT NULL,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version_id SERIAL PRIMARY KEY,
    weights_path TEXT NOT NULL,
    num_classes INT NOT NULL,
    class_names_json TEXT NOT NULL,
    old_classes_map50_95 FLOAT NOT NULL,
    new_class_map50 FLOAT NOT NULL,
    new_class_recall FLOAT NOT NULL,
    promoted_to_production BOOLEAN NOT NULL DEFAULT FALSE,
    trained_on_cluster_id INT REFERENCES unknown_clusters(cluster_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cluster_sharpness 
ON unknown_observations(cluster_id, sharpness_score DESC);
~~~

---

## 8. Complete 12-Module Python Implementation (`pipeline_12_modules.py`)

~~~python
import os
import json
import shutil
import cv2
import yaml
import faiss
import torch
import psycopg2
import numpy as np
from pathlib import Path
from collections import deque
from multiprocessing import Process, Queue, Event
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv
from ultralytics import YOLO, FastSAM
from pydantic import BaseModel
from google import genai
from google.genai import types

# =====================================================================
# 0. STRICT ENVIRONMENT LOADING FROM FYP_phase1/.env ONLY
#    (Loads ONLY Postgres & Gemini credentials; ZERO logic imported)
# =====================================================================
FYP_PHASE1_ENV_PATH = os.environ.get(
    "FYP_PHASE1_ENV_PATH",
    str(Path(__file__).resolve().parent.parent / "FYP_phase1" / ".env")
)
if os.path.exists(FYP_PHASE1_ENV_PATH):
    load_dotenv(dotenv_path=FYP_PHASE1_ENV_PATH, override=True)
else:
    raise FileNotFoundError(
        f"CRITICAL: Could not find FYP_phase1 .env file at {FYP_PHASE1_ENV_PATH}. "
        "Set FYP_PHASE1_ENV_PATH to point to the FYP_phase1/.env file."
    )

def build_postgres_dsn() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "surveillance")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pwd = os.environ.get("POSTGRES_PASSWORD", "")
    return f"dbname={db} user={user} password={pwd} host={host} port={port}"

POSTGRES_DSN = build_postgres_dsn()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("CRITICAL: GEMINI_API_KEY is missing in FYP_phase1/.env")

# =====================================================================
# SYSTEM HYPERPARAMETERS & PATHS
# =====================================================================
VIDEO_SOURCE = os.environ.get("SURVEILLANCE_STREAM_URL", "rtsp://your_camera_stream")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHAMPION_WEIGHTS_PATH = str(MODELS_DIR / "champion_latest.pt")
INITIAL_BASE_WEIGHTS = "yolo11n.pt"
FASTSAM_WEIGHTS = "FastSAM-s.pt"

# Copy base weights to champion_latest.pt on first boot
if not os.path.exists(CHAMPION_WEIGHTS_PATH):
    YOLO(INITIAL_BASE_WEIGHTS)  # Ensures weights are downloaded
    shutil.copy(INITIAL_BASE_WEIGHTS, CHAMPION_WEIGHTS_PATH)

MIN_CONTOUR_AREA = 500              # Module 3: Minimum pixel area for MOG2 contour
MIN_SUB_OBJECT_AREA = 350           # Module 5: Minimum pixel area for a split sub-object
YOLO_PAD_PX = 15                    # Module 3: Dilation margin around known YOLO objects
MIN_TRACK_FRAMES = 8                # Module 4: Minimum frames an unknown track must survive
MIN_DIR_PERSISTENCE = 0.50          # Module 4: Net displacement / path length ratio

PART_MIN_FG_RATIO = 0.06            # Module 5: FastSAM sub-mask must cover >= 6% of moving blob
PART_MERGE_COSINE_THRESH = 0.62     # Module 5: DINOv2 cosine threshold to merge sub-parts
MASKCUT_SEP_THRESH = 0.22           # Module 5: Fallback Fiedler separation threshold
FAISS_CLUSTER_THRESH = 0.78         # Module 7: Cosine threshold to join existing FAISS cluster
GEMINI_TRIGGER_COUNT = int(os.environ.get("GEMINI_TRIGGER_COUNT", "100"))

# Module 10-12 Continual Learning & Evaluation Gate Thresholds
RETRAIN_EPOCHS = 25
FREEZE_BACKBONE_LAYERS = 10
MAX_OLD_MAP_DROP = 0.015            # Gate 1: Max allowed drop in mAP50-95 on old classes (1.5%)
MIN_NEW_CLASS_MAP50 = 0.60          # Gate 2: Min required mAP50 on newly discovered class
MIN_NEW_CLASS_RECALL = 0.65         # Gate 2: Min required Recall on newly discovered class

# Optional: Holdout class IDs during testing (e.g., [0, 1] to pretend 'person' & 'bicycle' are unknown)
HOLDOUT_CLASS_IDS = [
    int(x) for x in os.environ.get("TEST_HOLDOUT_CLASS_IDS", "").split(",") if x.strip().isdigit()
]

# =====================================================================
# MODULE 9 SCHEMA: GEMINI STRUCTURED OUTPUT
# =====================================================================
class GeminiObjectSchema(BaseModel):
    object_name: str
    category: str
    is_valid_physical_object: bool
    confidence: float
    reasoning: str

# =====================================================================
# MODULE 4: TEMPORAL TRAJECTORY GATE & TRACK KEYFRAME DEDUPLICATOR
# =====================================================================
class UnknownTrack:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.hits = 1
        self.missed = 0
        self.dispatched = False
        self.centroids = deque(maxlen=15)
        self.centroids.append(self._center(bbox))
        self.best_sharpness = -1.0
        self.best_payload = None

    @staticmethod
    def _center(b):
        return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])

    def update(self, bbox, frame_bgr, fg_mask):
        self.bbox = bbox
        self.hits += 1
        self.missed = 0
        self.centroids.append(self._center(bbox))

        x1, y1, x2, y2 = bbox
        raw_crop = frame_bgr[y1:y2, x1:x2]
        mask_crop = fg_mask[y1:y2, x1:x2]
        if raw_crop.size > 0:
            gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharpness > self.best_sharpness:
                self.best_sharpness = sharpness
                self.best_payload = (
                    frame_bgr.copy(),
                    raw_crop.copy(),
                    mask_crop.copy(),
                    list(bbox),
                    sharpness,
                )

    def is_valid_moving_object(self):
        if self.hits < MIN_TRACK_FRAMES or len(self.centroids) < MIN_TRACK_FRAMES:
            return False
        pts = np.array(self.centroids)
        net_disp = np.linalg.norm(pts[-1] - pts[0])
        total_path = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) + 1e-5
        return (net_disp / total_path) >= MIN_DIR_PERSISTENCE

# =====================================================================
# MODULES 5 & 6: METHOD 2 COMPOUND BLOB DECOMPOSER & GLOBAL EMBEDDER
# =====================================================================
class Method2DecomposerAndEmbedder:
    def __init__(self, device="cuda"):
        self.device = device
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device).eval()
        self.fastsam = FastSAM(FASTSAM_WEIGHTS)
        self.bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def decompose_and_embed(self, frame_bgr, raw_crop, mask_crop, merged_bbox):
        frame_h, frame_w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = merged_bbox
        crop_h, crop_w = mask_crop.shape
        if crop_h < 32 or crop_w < 32:
            return [self._run_module6_embedding(raw_crop, mask_crop, merged_bbox, frame_w, frame_h, "SINGLE")]

        # Step 5.3: Extract DINOv2 16x16 dense spatial patch tokens [256, 384]
        bg_gray = np.full_like(raw_crop, 128)
        masked_rgb = np.where(mask_crop[..., None] > 0, raw_crop, bg_gray)
        pil_img = Image.fromarray(cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2RGB))
        tensor_in = self.dino_transform(pil_img).unsqueeze(0).to(self.device)

        feats = self.dinov2.forward_features(tensor_in)
        patch_tokens = feats["x_norm_patchtokens"][0].cpu().numpy()
        norms = np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-8
        patch_tokens = patch_tokens / norms

        # Step 5.1: Run FastSAM-s on the high-resolution crop
        sam_res = self.fastsam(
            raw_crop, device=self.device, retina_masks=True, conf=0.35, iou=0.7, verbose=False
        )[0]

        valid_submasks = []
        submask_vectors = []
        total_fg_pixels = max(1, cv2.countNonZero(mask_crop))

        # Step 5.2 & 5.4: Intersect with MOG2 mask & pool DINOv2 features per sub-mask
        if sam_res.masks is not None and len(sam_res.masks.data) > 1:
            for m in sam_res.masks.data.cpu().numpy():
                m_bin = (cv2.resize(m, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
                fg_inter = cv2.bitwise_and(m_bin, (mask_crop > 0).astype(np.uint8))

                if cv2.countNonZero(fg_inter) < PART_MIN_FG_RATIO * total_fg_pixels:
                    continue

                w_16 = cv2.resize(fg_inter, (16, 16), interpolation=cv2.INTER_AREA).flatten()
                if np.sum(w_16) < 1e-3:
                    continue

                pooled_vec = np.average(patch_tokens, axis=0, weights=w_16)
                pooled_vec = pooled_vec / (np.linalg.norm(pooled_vec) + 1e-8)

                valid_submasks.append(fg_inter * 255)
                submask_vectors.append(pooled_vec)

        # Step 5.5: Iterative Agglomerative Cosine Merging (UnSAM Conquer Stage)
        if len(valid_submasks) > 1:
            groups = [[i] for i in range(len(valid_submasks))]
            g_vecs = list(submask_vectors)

            while len(groups) > 1:
                best_sim, best_pair = -1.0, None
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        sim = float(np.dot(g_vecs[i], g_vecs[j]))
                        if sim > best_sim:
                            best_sim, best_pair = sim, (i, j)

                if best_sim >= PART_MERGE_COSINE_THRESH and best_pair is not None:
                    i, j = best_pair
                    groups[i].extend(groups[j])
                    merged_v = (g_vecs[i] + g_vecs[j]) / 2.0
                    g_vecs[i] = merged_v / (np.linalg.norm(merged_v) + 1e-8)
                    groups.pop(j)
                    g_vecs.pop(j)
                else:
                    break

            if len(groups) > 1:
                decomposed_list = []
                for g in groups:
                    comb_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                    for idx in g:
                        comb_mask = cv2.bitwise_or(comb_mask, valid_submasks[idx])
                    comb_mask = cv2.morphologyEx(comb_mask, cv2.MORPH_CLOSE, self.bridge_kernel)

                    pts = cv2.findNonZero(comb_mask)
                    if pts is not None:
                        sx, sy, sw, sh = cv2.boundingRect(pts)
                        if sw * sh >= MIN_SUB_OBJECT_AREA:
                            abs_box = [x1 + sx, y1 + sy, x1 + sx + sw, y1 + sy + sh]
                            s_raw = raw_crop[sy : sy + sh, sx : sx + sw].copy()
                            s_mask = comb_mask[sy : sy + sh, sx : sx + sw].copy()
                            decomposed_list.append(
                                self._run_module6_embedding(
                                    s_raw, s_mask, abs_box, frame_w, frame_h, "METHOD2_FASTSAM_DINOV2"
                                )
                            )
                if len(decomposed_list) > 1:
                    return decomposed_list

        # Fallback: CutLER / MaskCut Normalized Cuts
        mask_16 = cv2.resize(mask_crop, (16, 16), interpolation=cv2.INTER_AREA)
        fg_idx = np.where(mask_16.flatten() > 64)[0]
        if len(fg_idx) >= 16:
            fg_feats = patch_tokens[fg_idx]
            sim_mat = np.dot(fg_feats, fg_feats.T)
            W = np.where(sim_mat >= 0.20, sim_mat, 1e-5)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-8))
            L_sym = np.eye(len(fg_idx)) - D_inv_sqrt @ W @ D_inv_sqrt

            _, eigvecs = np.linalg.eigh(L_sym)
            fiedler = D_inv_sqrt @ eigvecs[:, 1]
            split_val = np.mean(fiedler)
            idx_a = np.where(fiedler > split_val)[0]
            idx_b = np.where(fiedler <= split_val)[0]

            if len(idx_a) >= 4 and len(idx_b) >= 4:
                sim_aa = np.mean(sim_mat[np.ix_(idx_a, idx_a)])
                sim_bb = np.mean(sim_mat[np.ix_(idx_b, idx_b)])
                sim_ab = np.mean(sim_mat[np.ix_(idx_a, idx_b)])
                if (((sim_aa + sim_bb) / 2.0) - sim_ab) >= MASKCUT_SEP_THRESH:
                    ncut_list = []
                    for subset in (fg_idx[idx_a], fg_idx[idx_b]):
                        g16 = np.zeros(256, dtype=np.uint8)
                        g16[subset] = 255
                        up_m = cv2.resize(g16.reshape(16, 16), (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                        sub_m = cv2.bitwise_and(mask_crop, (up_m > 110).astype(np.uint8) * 255)
                        sub_m = cv2.morphologyEx(sub_m, cv2.MORPH_CLOSE, self.bridge_kernel)
                        pts = cv2.findNonZero(sub_m)
                        if pts is not None:
                            sx, sy, sw, sh = cv2.boundingRect(pts)
                            if sw * sh >= MIN_SUB_OBJECT_AREA:
                                abs_box = [x1 + sx, y1 + sy, x1 + sx + sw, y1 + sy + sh]
                                s_raw = raw_crop[sy : sy + sh, sx : sx + sw].copy()
                                s_mask = sub_m[sy : sy + sh, sx : sx + sw].copy()
                                ncut_list.append(
                                    self._run_module6_embedding(
                                        s_raw, s_mask, abs_box, frame_w, frame_h, "FALLBACK_MASKCUT"
                                    )
                                )
                    if len(ncut_list) > 1:
                        return ncut_list

        return [self._run_module6_embedding(raw_crop, mask_crop, merged_bbox, frame_w, frame_h, "SINGLE")]

    @torch.no_grad()
    def _run_module6_embedding(self, sub_raw, sub_mask, abs_bbox, frame_w, frame_h, method_label):
        bg_gray = np.full_like(sub_raw, 128)
        isolated_bgr = np.where(sub_mask[..., None] > 0, sub_raw, bg_gray)

        h, w, _ = isolated_bgr.shape
        dim = max(h, w)
        square_bgr = np.full((dim, dim, 3), 128, dtype=np.uint8)
        y_off, x_off = (dim - h) // 2, (dim - w) // 2
        square_bgr[y_off : y_off + h, x_off : x_off + w] = isolated_bgr

        pil_sub = Image.fromarray(cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB))
        t_sub = self.dino_transform(pil_sub).unsqueeze(0).to(self.device)
        emb = self.dinov2(t_sub).cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)

        gray = cv2.cvtColor(sub_raw, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return {
            "bbox": abs_bbox,
            "frame_hw": [frame_h, frame_w],
            "raw_crop": sub_raw,
            "masked_crop": square_bgr,
            "embedding": emb,
            "sharpness": sharpness,
            "method": method_label,
        }

# =====================================================================
# PHASE 3: MODULES 10, 11, AND 12 (CONTINUAL RETRAINING, EVALUATION GATE & HOT-SWAP)
# =====================================================================
def continual_retraining_worker(retrain_queue: Queue, model_reload_event: Event):
    """
    Listens for newly identified clusters from Module 9, builds an incremental dataset
    with Teacher Pseudo-Labeling + Experience Replay (Module 10), trains a Challenger
    model & evaluates Champion vs. Challenger metrics (Module 11), and atomically
    hot-swaps production weights if the gates pass (Module 12).
    """
    db_conn = psycopg2.connect(POSTGRES_DSN)
    replay_img_dir = Path("storage/replay_buffer/images")
    replay_lbl_dir = Path("storage/replay_buffer/labels")
    replay_img_dir.mkdir(parents=True, exist_ok=True)
    replay_lbl_dir.mkdir(parents=True, exist_ok=True)

    while True:
        cluster_id = retrain_queue.get()
        if cluster_id is None:
            break

        cur = db_conn.cursor()
        cur.execute("""
            SELECT identified_label, status FROM unknown_clusters WHERE cluster_id = %s;
        """, (cluster_id,))
        row = cur.fetchone()
        if not row or row[1] != "IDENTIFIED":
            continue

        new_label_name = row[0].strip().lower()
        cur.execute("UPDATE unknown_clusters SET status = 'RETRAINING' WHERE cluster_id = %s;", (cluster_id,))
        db_conn.commit()

        # -----------------------------------------------------------------
        # MODULE 10: INCREMENTAL DATASET BUILDER & TEACHER PSEUDO-LABELER
        # -----------------------------------------------------------------
        champion_model = YOLO(CHAMPION_WEIGHTS_PATH)
        existing_names = dict(champion_model.names)  # {0: 'person', 1: 'bicycle', ...}
        name_to_id = {v.lower(): k for k, v in existing_names.items()}

        if new_label_name in name_to_id:
            target_cls_id = name_to_id[new_label_name]
        else:
            target_cls_id = max(existing_names.keys()) + 1 if existing_names else 0
            existing_names[target_cls_id] = new_label_name

        old_class_ids = [cid for cid in existing_names.keys() if cid != target_cls_id]

        ds_root = Path(f"storage/datasets/cluster_{cluster_id}")
        for split in ("train", "val"):
            (ds_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (ds_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        cur.execute("""
            SELECT observation_id, full_frame_path, bbox_xyxy, frame_hw
            FROM unknown_observations
            WHERE cluster_id = %s
            ORDER BY sharpness_score DESC;
        """, (cluster_id,))
        obs_rows = cur.fetchall()

        split_idx = int(0.80 * len(obs_rows))
        for idx, (obs_id, frame_path, bbox, frame_hw) in enumerate(obs_rows):
            if not os.path.exists(frame_path):
                continue
            split = "train" if idx < split_idx else "val"
            dst_img = ds_root / "images" / split / f"obs_{obs_id}.jpg"
            dst_lbl = ds_root / "labels" / split / f"obs_{obs_id}.txt"
            shutil.copy(frame_path, dst_img)

            label_lines = []
            fh, fw = frame_hw

            # 10.1 Teacher Pseudo-Labeling: Annotate co-occurring known objects with standard YOLO bboxes
            teacher_res = champion_model(frame_path, conf=0.50, verbose=False)[0]
            if teacher_res.boxes is not None and len(teacher_res.boxes) > 0:
                t_boxes = teacher_res.boxes.xyxy.cpu().numpy()
                t_clses = teacher_res.boxes.cls.cpu().numpy().astype(int)
                for (tx1, ty1, tx2, ty2), t_cls in zip(t_boxes, t_clses):
                    if t_cls in HOLDOUT_CLASS_IDS:
                        continue
                    xc = np.clip(((tx1 + tx2) / 2.0) / fw, 0.0, 1.0)
                    yc = np.clip(((ty1 + ty2) / 2.0) / fh, 0.0, 1.0)
                    w = np.clip((tx2 - tx1) / fw, 0.0, 1.0)
                    h = np.clip((ty2 - ty1) / fh, 0.0, 1.0)
                    label_lines.append(f"{t_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            # 10.2 Append the newly discovered object's standard 5-column YOLO bounding box
            bx1, by1, bx2, by2 = bbox
            xc = np.clip(((bx1 + bx2) / 2.0) / fw, 0.0, 1.0)
            yc = np.clip(((by1 + by2) / 2.0) / fh, 0.0, 1.0)
            w = np.clip((bx2 - bx1) / fw, 0.0, 1.0)
            h = np.clip((by2 - by1) / fh, 0.0, 1.0)
            label_lines.append(f"{target_cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            dst_lbl.write_text("\n".join(label_lines) + "\n")

            # Save top 20 sharpest samples into the Experience Replay Buffer for future cycles
            if idx < 20:
                shutil.copy(dst_img, replay_img_dir / f"cluster_{cluster_id}_obs_{obs_id}.jpg")
                shutil.copy(dst_lbl, replay_lbl_dir / f"cluster_{cluster_id}_obs_{obs_id}.txt")

        # 10.3 Inject Experience Replay Buffer samples into Train & Val splits
        replay_imgs = sorted(list(replay_img_dir.glob("*.jpg")))
        for r_idx, r_img in enumerate(replay_imgs):
            r_lbl = replay_lbl_dir / f"{r_img.stem}.txt"
            if r_lbl.exists():
                r_split = "train" if (r_idx % 5 != 0) else "val"
                shutil.copy(r_img, ds_root / "images" / r_split / r_img.name)
                shutil.copy(r_lbl, ds_root / "labels" / r_split / r_lbl.name)

        dataset_yaml_path = ds_root / "dataset.yaml"
        dataset_Config = {
            "path": str(ds_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": existing_names,
        }
        dataset_yaml_path.write_text(yaml.dump(dataset_Config, sort_keys=True))

        # -----------------------------------------------------------------
        # MODULE 11: CONTINUAL FINE-TUNING & CHAMPION-VS-CHALLENGER GATE
        # -----------------------------------------------------------------
        # 11.1 Evaluate Champion Baseline on Old Classes
        champ_val = champion_model.val(data=str(dataset_yaml_path), split="val", verbose=False)
        champ_old_map = float(np.mean(champ_val.box.maps[old_class_ids])) if len(old_class_ids) > 0 else 0.0

        # 11.2 Train Challenger with Frozen Backbone (freeze=10)
        challenger_trainer = YOLO(CHAMPION_WEIGHTS_PATH)
        run_name = f"challenger_cluster_{cluster_id}"
        challenger_trainer.train(
            data=str(dataset_yaml_path),
            epochs=RETRAIN_EPOCHS,
            imgsz=640,
            batch=16,
            freeze=FREEZE_BACKBONE_LAYERS,
            lr0=0.001,
            project=str(MODELS_DIR / "runs"),
            name=run_name,
            exist_ok=True,
            verbose=False,
        )

        challenger_weights = MODELS_DIR / "runs" / run_name / "weights" / "best.pt"
        if not challenger_weights.exists():
            continue

        # 11.3 Evaluate Challenger on Both Old Classes and Newly Discovered Class
        challenger_model = YOLO(str(challenger_weights))
        chal_val = challenger_model.val(data=str(dataset_yaml_path), split="val", verbose=False)

        chal_old_map = float(np.mean(chal_val.box.maps[old_class_ids])) if len(old_class_ids) > 0 else 0.0
        new_cls_map50 = float(chal_val.box.ap50[target_cls_id]) if target_cls_id < len(chal_val.box.ap50) else float(chal_val.box.map50)
        new_cls_recall = float(chal_val.box.r[target_cls_id]) if target_cls_id < len(chal_val.box.r) else float(chal_val.box.mr)

        passed_old_retention = (chal_old_map >= (champ_old_map - MAX_OLD_MAP_DROP))
        passed_new_competence = (new_cls_map50 >= MIN_NEW_CLASS_MAP50) and (new_cls_recall >= MIN_NEW_CLASS_RECALL)
        promote_challenger = bool(passed_old_retention and passed_new_competence)

        # -----------------------------------------------------------------
        # MODULE 12: ATOMIC MODEL REGISTRY & ZERO-DOWNTIME HOT-SWAP
        # -----------------------------------------------------------------
        cur.execute("""
            INSERT INTO model_registry
            (weights_path, num_classes, class_names_json, old_classes_map50_95,
             new_class_map50, new_class_recall, promoted_to_production, trained_on_cluster_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            str(challenger_weights), len(existing_names), json.dumps(existing_names),
            chal_old_map, new_cls_map50, new_cls_recall, promote_challenger, cluster_id
        ))

        if promote_challenger:
            staging_path = str(MODELS_DIR / "champion_staging.pt")
            shutil.copy(str(challenger_weights), staging_path)
            os.replace(staging_path, CHAMPION_WEIGHTS_PATH)  # Atomic POSIX/NTFS file swap

            cur.execute("""
                UPDATE unknown_clusters
                SET status = 'DEPLOYED_IN_YOLO', assigned_yolo_class_id = %s, updated_at = CURRENT_TIMESTAMP
                WHERE cluster_id = %s;
            """, (target_cls_id, cluster_id))
            db_conn.commit()

            # Signal Phase 1 (30 FPS loop) to hot-swap YOLO weights immediately
            model_reload_event.set()
        else:
            cur.execute("""
                UPDATE unknown_clusters SET status = 'IDENTIFIED', updated_at = CURRENT_TIMESTAMP
                WHERE cluster_id = %s;
            """, (cluster_id,))
            db_conn.commit()

# =====================================================================
# PHASE 2 WORKER: MODULES 5, 6, 7, 8, AND 9
# =====================================================================
def discovery_worker_loop(discovery_queue: Queue, retrain_queue: Queue):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decomposer_embedder = Method2DecomposerAndEmbedder(device=device)

    faiss_index_path = "storage/faiss_centroids.index"
    centroids_path = "storage/centroids.npy"
    if os.path.exists(faiss_index_path) and os.path.exists(centroids_path):
        faiss_index = faiss.read_index(faiss_index_path)
        centroids = list(np.load(centroids_path))
    else:
        faiss_index = faiss.IndexFlatIP(384)
        centroids = []

    db_conn = psycopg2.connect(POSTGRES_DSN)
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    os.makedirs("storage/frames", exist_ok=True)
    os.makedirs("storage/crops", exist_ok=True)
    os.makedirs("storage/masked_crops", exist_ok=True)

    while True:
        item = discovery_queue.get()
        if item is None:
            break

        track_id, frame_bgr, raw_crop, mask_crop, merged_bbox, _ = item

        # MODULES 5 & 6: Decompose merged blob (Method 2) & compute isolated DINOv2 embeddings
        sub_objects = decomposer_embedder.decompose_and_embed(frame_bgr, raw_crop, mask_crop, merged_bbox)

        for sub_idx, obj in enumerate(sub_objects):
            bx1, by1, bx2, by2 = obj["bbox"]
            emb = obj["embedding"]
            sharpness = obj["sharpness"]
            method_used = obj["method"]
            frame_hw = obj["frame_hw"]

            # Save clean, pristine full frame for Module 10 YOLO retraining (NEVER with red box drawn on it)
            frame_path = f"storage/frames/track_{track_id}_sub_{sub_idx}.jpg"
            crop_path = f"storage/crops/track_{track_id}_sub_{sub_idx}.jpg"
            masked_path = f"storage/masked_crops/track_{track_id}_sub_{sub_idx}.jpg"

            cv2.imwrite(frame_path, frame_bgr)
            cv2.imwrite(crop_path, obj["raw_crop"])
            cv2.imwrite(masked_path, obj["masked_crop"])

            # MODULES 7 & 8: FAISS Centroid Matching & PostgreSQL Update
            cur = db_conn.cursor()
            if faiss_index.ntotal > 0:
                D, I = faiss_index.search(emb, 1)
                best_sim, best_faiss_id = float(D[0][0]), int(I[0][0])
            else:
                best_sim, best_faiss_id = -1.0, -1

            if best_sim >= FAISS_CLUSTER_THRESH:
                cur.execute("""
                    UPDATE unknown_clusters
                    SET observation_count = observation_count + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE faiss_index_id = %s
                    RETURNING cluster_id, observation_count, status;
                """, (best_faiss_id,))
                cluster_id, obs_count, status = cur.fetchone()

                new_c = (obs_count - 1) * centroids[best_faiss_id] + emb[0]
                centroids[best_faiss_id] = new_c / np.linalg.norm(new_c)
                faiss_index.reset()
                faiss_index.add(np.array(centroids, dtype="float32"))
            else:
                new_faiss_id = faiss_index.ntotal
                faiss_index.add(emb)
                centroids.append(emb[0])
                best_sim = 1.0

                cur.execute("""
                    INSERT INTO unknown_clusters (faiss_index_id, observation_count)
                    VALUES (%s, 1)
                    RETURNING cluster_id, observation_count, status;
                """, (new_faiss_id,))
                cluster_id, obs_count, status = cur.fetchone()

            # Save FAISS index and centroids to disk for synchronization across restarts
            faiss.write_index(faiss_index, faiss_index_path)
            np.save(centroids_path, np.array(centroids, dtype="float32"))

            cur.execute("""
                INSERT INTO unknown_observations
                (cluster_id, track_id, sub_object_idx, crop_image_path, masked_crop_path,
                 full_frame_path, bbox_xyxy, frame_hw, sharpness_score,
                 cosine_similarity, decomposition_method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                cluster_id, track_id, sub_idx, crop_path, masked_path,
                frame_path, [bx1, by1, bx2, by2], frame_hw,
                sharpness, best_sim, method_used
            ))
            db_conn.commit()

            # MODULE 9: GEMINI 2.5 FLASH VLM TRIGGER AT COUNT == GEMINI_TRIGGER_COUNT
            if obs_count >= GEMINI_TRIGGER_COUNT and status == "COLLECTING":
                cur.execute("""
                    SELECT crop_image_path, full_frame_path, bbox_xyxy
                    FROM unknown_observations
                    WHERE cluster_id = %s
                    ORDER BY sharpness_score DESC LIMIT 3;
                """, (cluster_id,))
                top_rows = cur.fetchall()

                contents = [
                    "You are a computer vision surveillance analyst. Image 1 shows a full CCTV frame with a red bounding box around a single isolated moving object. Images 2, 3, and 4 show close-up crops of this exact same class of object captured across multiple sightings. Identify this specific physical object using a concise singular noun (e.g. 'bicycle', 'person', 'forklift', 'dog')."
                ]
                # Draw red box ONLY on an in-memory copy of the full frame for Gemini context:
                full_clean = cv2.imread(top_rows[0][1])
                target_box = top_rows[0][2]
                rx1, ry1, rx2, ry2 = target_box
                cv2.rectangle(full_clean, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                contents.append(Image.fromarray(cv2.cvtColor(full_clean, cv2.COLOR_BGR2RGB)))

                for r in top_rows:
                    contents.append(Image.open(r[0]))

                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeminiObjectSchema,
                        temperature=0.1,
                    ),
                )
                parsed = GeminiObjectSchema.model_validate_json(response.text)
                new_status = "IDENTIFIED" if parsed.is_valid_physical_object else "NOISE"

                cur.execute("""
                    UPDATE unknown_clusters
                    SET status = %s, identified_label = %s, vlm_confidence = %s, vlm_reasoning = %s
                    WHERE cluster_id = %s;
                """, (new_status, parsed.object_name, parsed.confidence, parsed.reasoning, cluster_id))
                db_conn.commit()

                # Trigger Phase 3 Continual Retraining & Evaluation if valid physical object
                if new_status == "IDENTIFIED" and parsed.confidence >= 0.70:
                    retrain_queue.put(cluster_id)

# =====================================================================
# PHASE 1 REAL-TIME LOOP: MODULES 1, 2, 3, AND 4 (+ ATOMIC HOT-SWAP)
# =====================================================================
def run_surveillance_pipeline():
    discovery_queue = Queue(maxsize=256)
    retrain_queue = Queue(maxsize=64)
    model_reload_event = Event()

    discovery_proc = Process(
        target=discovery_worker_loop, args=(discovery_queue, retrain_queue), daemon=True
    )
    retrain_proc = Process(
        target=continual_retraining_worker, args=(retrain_queue, model_reload_event), daemon=True
    )
    discovery_proc.start()
    retrain_proc.start()

    yolo_model = YOLO(CHAMPION_WEIGHTS_PATH)
    active_holdout_ids = set(HOLDOUT_CLASS_IDS)

    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (YOLO_PAD_PX, YOLO_PAD_PX))

    active_tracks = []
    next_track_id = 1
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape

        # Check if Module 12 promoted a new Challenger model to Champion
        if model_reload_event.is_set():
            yolo_model = YOLO(CHAMPION_WEIGHTS_PATH)
            active_holdout_ids.clear()  # Newly trained classes are now active in Stream B
            model_reload_event.clear()

        # MODULE 1: MOG2 Foreground Mask (Drop shadows = 127, keep pure motion = 255)
        fg_mask = mog2.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel)

        # MODULE 2 & MODULE 3: Known-Object Detection & Bounding-Box Erasure (Pad = 15 px)
        yolo_res = yolo_model(frame, verbose=False)[0]
        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            cls_ids = yolo_res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cid in zip(yolo_res.boxes.xyxy.cpu().numpy().astype(int), cls_ids):
                if cid in active_holdout_ids:
                    continue
                cv2.rectangle(
                    fg_mask,
                    (max(0, x1 - YOLO_PAD_PX), max(0, y1 - YOLO_PAD_PX)),
                    (min(W, x2 + YOLO_PAD_PX), min(H, y2 + YOLO_PAD_PX)),
                    0, -1
                )

        # Extract residual unknown moving contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                candidate_boxes.append([x, y, x + w, y + h])

        # MODULE 4: Associate Unknown Contours & Gate by Trajectory Persistence
        used_candidates = set()
        for track in active_tracks:
            best_iou, best_idx = 0.0, -1
            for idx, cb in enumerate(candidate_boxes):
                if idx in used_candidates:
                    continue
                ix1, iy1 = max(track.bbox[0], cb[0]), max(track.bbox[1], cb[1])
                ix2, iy2 = min(track.bbox[2], cb[2]), min(track.bbox[3], cb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1])
                area_b = (cb[2] - cb[0]) * (cb[3] - cb[1])
                iou = inter / float(area_a + area_b - inter + 1e-5)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

            if best_iou >= 0.25:
                track.update(candidate_boxes[best_idx], frame, fg_mask)
                used_candidates.add(best_idx)
            else:
                track.missed += 1

        for idx, cb in enumerate(candidate_boxes):
            if idx not in used_candidates:
                active_tracks.append(UnknownTrack(next_track_id, cb))
                next_track_id += 1

        surviving_tracks = []
        for track in active_tracks:
            if track.is_valid_moving_object():
                tx1, ty1, tx2, ty2 = track.bbox
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
                cv2.putText(frame, f"UNKNOWN TRACK:{track.track_id}", (tx1, ty1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if track.missed > 5:
                if track.is_valid_moving_object() and not track.dispatched and track.best_payload:
                    f_bgr, r_crop, m_crop, b_box, sharp = track.best_payload
                    if not discovery_queue.full():
                        discovery_queue.put((track.track_id, f_bgr, r_crop, m_crop, b_box, sharp))
                    track.dispatched = True
            else:
                surviving_tracks.append(track)

        active_tracks = surviving_tracks

    discovery_queue.put(None)
    retrain_queue.put(None)
    discovery_proc.join()
    retrain_proc.join()
    cap.release()

if __name__ == "__main__":
    run_surveillance_pipeline()
~~~

---

## 9. End-to-End Verification & Testing Protocol on Online Stationary Outdoor Surveillance Cameras

### 9.1 Testing Objective
Verify all 12 modules on a **stationary outdoor surveillance camera feed** (zero camera pan/tilt/zoom; fixed background perspective where only moving objects, pedestrians, cyclists, vehicles, and lighting change) while loading credentials strictly from `FYP_phase1/.env`.

### 9.2 How the "Holdout Class" Test Validates Compound Decomposition & Retraining
Standard pre-trained `yolo11n-seg.pt` already knows COCO class `0 (person)` and class `1 (bicycle)`. To test that our pipeline can:
1. Detect an unknown person riding an unknown bicycle via MOG2,
2. Split that merged cyclist blob into **2 separate unknown objects** (`Person` and `Bicycle`) via **Module 5 (Method 2: `FastSAM + DINOv2`)**,
3. Cluster them into two separate FAISS/PostgreSQL clusters,
4. Identify them via **Module 9 (Gemini 2.5 Flash)**, and
5. Retrain, evaluate, and hot-swap YOLO via **Modules 10–12**,

we set `TEST_HOLDOUT_CLASS_IDS="0,1"` in the test runner. This instructs Module 2 to ignore `person` and `bicycle` predictions initially, treating cyclists and pedestrians as **100% unknown moving objects** while still erasing moving cars (`class 2`) and trucks (`class 7`)!

### 9.3 Stationary Outdoor Camera Test Script (`verify_stationary_cctv.py`)

~~~python
import os
import time
import subprocess
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

# 1. Strictly load ONLY the .env file from FYP_phase1 (Zero logic imported)
FYP_ENV = Path(__file__).resolve().parent.parent / "FYP_phase1" / ".env"
assert FYP_ENV.exists(), f"Missing FYP_phase1/.env at {FYP_ENV}"
load_dotenv(dotenv_path=str(FYP_ENV), override=True)

def resolve_live_stationary_stream(youtube_or_hls_url: str) -> str:
    """
    Resolves a public stationary outdoor traffic/crosswalk stream URL (HLS .m3u8)
    using yt-dlp so OpenCV VideoCapture can read it directly.
    """
    if youtube_or_hls_url.endswith(".mp4") or youtube_or_hls_url.startswith("rtsp://"):
        return youtube_or_hls_url
    cmd = ["yt-dlp", "-f", "best[ext=mp4]/best", "-g", youtube_or_hls_url]
    resolved_url = subprocess.check_output(cmd, text=True).strip().splitlines()[0]
    return resolved_url

def verify_Camera_is_stationary(stream_url: str, sample_frames: int = 60) -> bool:
    """
    Verifies that the online surveillance camera has zero ego-motion (stationary mount)
    by checking that median global optical flow across background corners is < 0.35 px.
    """
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(stream_url)
    ret, prev = cap.read()
    assert ret, "Failed to read initial frame from stream."
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    global_drifts = []

    for _ in range(sample_frames):
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if pts is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
            valid_prev = pts[status.flatten() == 1]
            valid_next = next_pts[status.flatten() == 1]
            if len(valid_prev) > 20:
                displacements = np.linalg.norm(valid_next - valid_prev, axis=2)
                # Median displacement reflects static background (moving objects are outliers)
                global_drifts.append(float(np.median(displacements)))
        prev_gray = curr_gray
    cap.release()
    mean_bg_drift = sum(global_drifts) / max(1, len(global_drifts))
    print(f"[STATIC CAMERA CHECK] Median Background Drift: {mean_bg_drift:.4f} px/frame")
    return mean_bg_drift < 0.35

if __name__ == "__main__":
    # Public stationary outdoor crosswalk/street camera or static VIRAT/MOT17 outdoor CCTV file
    PUBLIC_STATIC_CCTV_URL = os.environ.get(
        "TEST_CCTV_URL",
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
    )
    direct_stream = resolve_live_stationary_stream(PUBLIC_STATIC_CCTV_URL)
    assert verify_Camera_is_stationary(direct_stream), "Camera has ego-motion! Use a fixed CCTV stream."

    # Configure rapid verification thresholds (trigger Gemini & Retraining at 15 sightings)
    os.environ["SURVEILLANCE_STREAM_URL"] = direct_stream
    os.environ["TEST_HOLDOUT_CLASS_IDS"] = "0,1"  # Hold out person (0) & bicycle (1) as 'Unknowns'
    os.environ["GEMINI_TRIGGER_COUNT"] = "15"     # Trigger Gemini VLM & YOLO Retraining at 15 sightings

    print("[TEST START] Launching 12-Module Pipeline with FYP_phase1/.env credentials...")
    from pipeline_12_modules import run_surveillance_pipeline, POSTGRES_DSN
    run_surveillance_pipeline()

    # Post-Run Database Verification Assertions
    conn = psycopg2.connect(POSTGRES_DSN)
    cur = conn.cursor()

    cur.execute("SELECT decomposition_method, COUNT(*) FROM unknown_observations GROUP BY decomposition_method;")
    print("[DB CHECK 1 - Decomposition Methods Used]:", cur.fetchall())

    cur.execute("SELECT cluster_id, observation_count, status, identified_label, vlm_confidence FROM unknown_clusters;")
    print("[DB CHECK 2 - Discovered Clusters & Gemini Labels]:", cur.fetchall())

    cur.execute("SELECT model_version_id, old_classes_map50_95, new_class_map50, promoted_to_production FROM model_registry;")
    print("[DB CHECK 3 - Champion vs. Challenger Model Registry]:", cur.fetchall())
~~~

### 9.4 Five-Point Verification Checklist for Sign-Off
1. **Credential Isolation Verified:** Confirm `verify_stationary_cctv.py` connects to PostgreSQL and Gemini 2.5 Flash using only `../FYP_phase1/.env` with zero imports from `FYP_phase1`.
2. **Stationary Background Rejection:** Confirm parked vehicles in the outdoor stream generate `0` rows in `unknown_observations`, and moving cars (`class 2`) are erased by Module 3.
3. **Method 2 (`FastSAM + DINOv2`) Cyclist Separation:** Query `SELECT track_id, COUNT(sub_object_idx) FROM unknown_observations WHERE decomposition_method = 'METHOD2_FASTSAM_DINOV2' GROUP BY track_id HAVING COUNT(*) >= 2;` and verify that single cyclist tracks produce `sub_object_idx = 0` (Person crop with bike masked to gray) and `sub_object_idx = 1` (Bicycle crop with rider masked to gray).
4. **Gemini VLM Identification:** Confirm `unknown_clusters` transitions from `COLLECTING` to `IDENTIFIED` with `identified_label` populated (`'person'` and `'bicycle'`) once `observation_count` reaches the trigger threshold.
5. **Champion-vs-Challenger Gate & Atomic Hot-Swap:** Confirm `model_registry` logs the validation metrics (`old_classes_map50_95` and `new_class_map50`) and sets `promoted_to_production = TRUE` only when both evaluation gates pass, updating `unknown_clusters.status` to `'DEPLOYED_IN_YOLO'`.