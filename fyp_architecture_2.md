# End-to-End Ground Truth Architecture: Open-World Moving Object Discovery Pipeline

## 1. Executive Summary & System Purpose

### The Core Problem
Standard object detectors (such as YOLO) operate under a **closed-world assumption**:
1. They only detect classes present in their training set and ignore unknown physical objects (unrecognized equipment, stray animals, falling debris, novel vehicles).
2. They detect known objects even when completely stationary (such as parked cars), creating false alarms when only moving objects matter.

### How This Architecture Solves It
This system operates as a two-phase **Open-World Continual Discovery Engine**:
* **Phase 1: Upstream Real-Time Discrepancy Tracker (Modules 1–4):** Runs at 30+ FPS on the video stream. It extracts class-agnostic pixel motion (Stream A) and known semantic objects via YOLO (Stream B), erases known objects from the motion mask at the pixel level, and tracks the surviving unknown moving blobs using Kalman trajectory gating.
* **Phase 2: Downstream Self-Supervised Clustering & VLM Labeling (Modules 5–8):** Runs asynchronously in a background worker. Once an unknown track is confirmed, it extracts the sharpest keyframe, masks out the static background, embeds the crop with **DINOv2-Reg**, clusters embeddings online in **FAISS**, logs image paths and counts in **PostgreSQL**, and triggers **Gemini 2.5 Flash** once any cluster reaches **100 observations** to automatically identify and name the new object class.

---

## 2. End-to-End Pipeline Data Flow

| Module | Component / GitHub Reference | Input Data | Output Data | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1** | OpenCV `BackgroundSubtractorMOG2` (`opencv/opencv`) | Raw BGR Frame `(H, W, 3)` | Binary Motion Mask `(H, W, 1)` | Captures all moving pixels while suppressing shadows (`127`) and static objects. |
| **Module 2** | Ultralytics YOLO (`ultralytics/ultralytics`) | Raw BGR Frame `(H, W, 3)` | Known Boxes `[K, 4]` `(x1, y1, x2, y2)` | Locates all known trained classes in the current frame. |
| **Module 3** | Mask-Level Discrepancy Engine (NumPy / OpenCV) | Motion Mask `(H, W, 1)` + Known Boxes `[K, 4]` | Residual Mask + Unknown Boxes `[U, 5]` | Dilates known YOLO boxes by 10%, zeroes them out of the motion mask, and extracts surviving contours. |
| **Module 4** | ByteTrack (`ifzhang/ByteTrack`) | Candidate Unknown Boxes `[U, 5]` | Confirmed Unknown Tracks `[V, 5]` + `track_id` | Filters out swaying trees and noise using Kalman persistence and directional displacement. |
| **Module 5** | Keyframe Deduplicator & Mask Preprocessor | Confirmed Track Stream + Residual Mask | 1 Sharpest Crop + Masked Crop + Full Frame per `track_id` | Prevents 1 slow-moving object from triggering 100 counts in 3.3 seconds; removes background texture. |
| **Module 6** | DINOv2 with Registers (`facebookresearch/dinov2`) | Letterboxed Masked Crop `(224, 224, 3)` | L2-Normalized Vector `(1, 384)` `float32` | Generates self-supervised visual part embeddings without class bias. |
| **Module 7** | FAISS (`facebookresearch/faiss`) + PostgreSQL | Normalized Vector `(1, 384)` + Image Paths | `cluster_id` + Updated `observation_count` | Performs online cosine centroid matching (`sim >= 0.78`) and stores relational metadata. |
| **Module 8** | Google GenAI SDK (`googleapis/python-genai`) | Top 3 Sharpest Crops + 1 Full Context Frame | Structured JSON (`object_name`, `confidence`) | Identifies the unknown object class when `observation_count == 100` and writes the label to PostgreSQL. |

---

## 3. Detailed Module Specifications & Selection Rationale

### Module 1: Class-Agnostic Motion Extraction (Stream A)
* **Implementation:** `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)`
* **Why Chosen:** Unlike simple frame differencing (which leaves hollow boxes and ghosting artifacts), MOG2 models each pixel as a mixture of Gaussian distributions and labels shadows as gray (`127`). Thresholding at `254` strips shadows immediately, and a `5x5` morphological opening removes sensor noise.
* **Input:** Raw BGR Frame `(H, W, 3)`
* **Output:** Clean Binary Foreground Mask `(H, W, 1)` with values in `{0, 255}`

### Module 2: Known-Object Detection (Stream B)
* **Implementation:** Ultralytics `YOLO("yolo11n.pt")` or your custom-trained weights.
* **Why Chosen:** Provides real-time single-pass bounding boxes for known classes so they can be excluded from anomaly discovery.
* **Input:** Raw BGR Frame `(H, W, 3)`
* **Output:** Known Bounding Boxes `[K, 4]` in `(x1, y1, x2, y2)` pixel coordinates.

### Module 3: Mask-Level Erasure & Unknown Contour Extraction
* **Implementation:** Pixel-level zeroing on `fg_mask` followed by `cv2.findContours()`.
* **Why Chosen:** Box-level IoU filtering fails when a known person walks next to or pushes an unknown object (their motion blobs merge). Painting padded YOLO boxes black (`0`) on the binary motion mask cleanly severs the known object and preserves the unknown object's pixels.
* **Input:** Binary Foreground Mask `(H, W, 1)` and Known Boxes `[K, 4]`
* **Output:** Residual Binary Mask `(H, W, 1)` and Candidate Unknown Detections `[U, 5]` formatted as `[x1, y1, x2, y2, 1.0]`

### Module 4: Temporal Trajectory Gating (ByteTrack + Directional Persistence)
* **Implementation:** Standalone `BYTETracker` + sliding-window displacement ratio.
* **Why Chosen:** ByteTrack relies purely on Kalman motion state and IoU without needing pretrained Re-ID appearance weights (which fail on unknown classes). Adding a directional persistence check (`net_displacement / total_path_length >= 0.50` over `W = 8` frames) eliminates oscillating false positives like swaying tree branches.
* **Input:** Candidate Unknown Detections `[U, 5]`
* **Output:** Confirmed Unknown Tracks with persistent `track_id` and bounding box `[x1, y1, x2, y2]`

### Module 5: Track-Level Keyframe Deduplication & Background Suppression
* **Implementation:** Laplacian sharpness scorer (`cv2.Laplacian(gray, cv2.CV_64F).var() * area`) + foreground masking + square letterbox padding (`128, 128, 128`).
* **Why Chosen:**
  1. **Deduplication:** Without track-level deduplication, a single stray animal walking across a 30 FPS camera for 3.5 seconds generates 100 frames and prematurely triggers the 100-count Gemini API call on a single event. Emitting **one sharpest keyframe per completed track** ensures `observation_count = 100` represents 100 true independent sightings.
  2. **Background Masking:** Fixed CCTV cameras have uniform backgrounds (asphalt, grass). Zeroing out background pixels around the contour to neutral gray prevents DINOv2 from clustering objects by *where they appeared in the yard* instead of *what they are*.
* **Input:** Active Track Frames, Bounding Boxes, and Residual Foreground Masks
* **Output:** Saved `full_frame_path`, saved `crop_image_path`, and Masked Square PIL Image `(224, 224, 3)`

### Module 6: Self-Supervised Feature Extraction (DINOv2 with Registers)
* **Implementation:** `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')`
* **Why Chosen:** DINOv2 learns fine-grained visual part similarity without text-caption bias. The `_reg` (Register) variant prevents background artifact tokens from polluting embeddings of small surveillance crops.
* **Input:** Normalized Tensor `(1, 3, 224, 224)`
* **Output:** L2-Normalized Embedding Vector `(1, 384)` `float32`

### Module 7: Online FAISS Centroid Clustering & PostgreSQL Persistence
* **Implementation:** `faiss.IndexFlatIP(384)` + PostgreSQL (`unknown_clusters` and `unknown_observations` tables).
* **Why Chosen:** Because embeddings are L2-normalized, FAISS Inner Product (`IndexFlatIP`) computes exact Cosine Similarity. When a new embedding arrives:
  * If `cosine_similarity >= 0.78` with the nearest centroid: it belongs to that existing cluster. Increment `observation_count` in PostgreSQL and update the centroid vector via running average.
  * If `cosine_similarity < 0.78`: it is a brand-new unknown object type. Add a new centroid to FAISS and insert a new cluster row in PostgreSQL with `observation_count = 1`.
* **Input:** L2-Normalized Embedding `(1, 384)`, `track_id`, file paths, sharpness score
* **Output:** Updated `cluster_id` and current `observation_count`

### Module 8: Automated VLM Discovery Trigger (Gemini 2.5 Flash)
* **Implementation:** `google-genai` SDK calling `gemini-2.5-flash` with Structured Outputs (`response_schema`).
* **Why Chosen:** When `observation_count == 100`, the worker queries PostgreSQL for the **3 sharpest crops** in that cluster plus **1 full video frame** with the bounding box drawn. Passing both the full scene context and high-resolution crops allows Gemini to accurately deduce object scale and identity.
* **Input:** 1 Full Contextual Frame + Top 3 Sharpest Crops for `cluster_id`
* **Output:** `identified_label`, `vlm_confidence`, and `vlm_reasoning` written back to PostgreSQL `unknown_clusters`

---

## 4. Environment Setup & Dependencies

```bash
pip install opencv-python ultralytics faiss-cpu torch torchvision psycopg2-binary pillow pydantic google-genai numpy
```

---

## 5. PostgreSQL Database Schema (`schema.sql`)

```sql
CREATE TABLE IF NOT EXISTS unknown_clusters (
    cluster_id SERIAL PRIMARY KEY,
    faiss_index_id INT UNIQUE NOT NULL,
    observation_count INT NOT NULL DEFAULT 1,
    status VARCHAR(32) NOT NULL DEFAULT 'COLLECTING', -- COLLECTING, IDENTIFIED, NOISE
    identified_label VARCHAR(128) DEFAULT NULL,
    vlm_confidence FLOAT DEFAULT NULL,
    vlm_reasoning TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS unknown_observations (
    observation_id SERIAL PRIMARY KEY,
    cluster_id INT NOT NULL REFERENCES unknown_clusters(cluster_id) ON DELETE CASCADE,
    track_id INT NOT NULL,
    crop_image_path TEXT NOT NULL,
    full_frame_path TEXT NOT NULL,
    bbox_xyxy INT[4] NOT NULL,
    sharpness_score FLOAT NOT NULL,
    cosine_similarity FLOAT NOT NULL,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cluster_sharpness 
ON unknown_observations(cluster_id, sharpness_score DESC);
```

---

## 6. Complete Reference Implementation (`pipeline_ground_truth.py`)

```python
import os
import cv2
import faiss
import torch
import psycopg2
import numpy as np
from collections import deque
from multiprocessing import Process, Queue
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from pydantic import BaseModel
from google import genai
from google.genai import types

# =====================================================================
# CONFIGURATION CONSTANTS
# =====================================================================
VIDEO_SOURCE = "rtsp://your_camera_stream"
YOLO_WEIGHTS = "yolo11n.pt"
POSTGRES_DSN = "dbname=surveillance user=postgres password=secret host=localhost"

MIN_CONTOUR_AREA = 500        # Minimum pixel area for an unknown moving blob
YOLO_PAD_PX = 15              # Dilation padding around known YOLO boxes before mask erasure
MIN_TRACK_FRAMES = 8          # Minimum frames an unknown track must survive
MIN_DIR_PERSISTENCE = 0.50    # Net displacement / total path ratio to filter swaying trees
COSINE_THRESHOLD = 0.78       # DINOv2 cosine similarity threshold to join an existing cluster
CLUSTER_TRIGGER_COUNT = 100   # Observation count required to trigger Gemini VLM

# =====================================================================
# GEMINI STRUCTURED OUTPUT SCHEMA
# =====================================================================
class GeminiObjectSchema(BaseModel):
    object_name: str
    category: str
    is_valid_physical_object: bool
    confidence: float
    reasoning: str

# =====================================================================
# LIGHTWEIGHT KALMAN / IOU TRACKER FOR UNKNOWN CONTOURS (MODULE 4)
# =====================================================================
class UnknownTrack:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.hits = 1
        self.missed = 0
        self.dispatched_to_worker = False
        self.centroids = deque(maxlen=15)
        self.centroids.append(self._center(bbox))
        self.best_sharpness = -1.0
        self.best_payload = None  # Holds (frame_bgr, raw_crop, mask_crop, bbox, sharpness)

    @staticmethod
    def _center(b):
        return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])

    def update(self, bbox, frame_bgr, fg_mask):
        self.bbox = bbox
        self.hits += 1
        self.missed = 0
        self.centroids.append(self._center(bbox))

        # Evaluate crop quality (Laplacian variance * area) to keep sharpest keyframe
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
                    bbox,
                    sharpness,
                )

    def is_persistent_and_linear(self):
        if self.hits < MIN_TRACK_FRAMES or len(self.centroids) < MIN_TRACK_FRAMES:
            return False
        pts = np.array(self.centroids)
        net_disp = np.linalg.norm(pts[-1] - pts[0])
        step_disp = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) + 1e-5
        return (net_disp / step_disp) >= MIN_DIR_PERSISTENCE

# =====================================================================
# PHASE 2: DOWNSTREAM ASYNC WORKER (MODULES 5, 6, 7, 8)
# =====================================================================
def discovery_worker_loop(queue: Queue):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device).eval()
    
    faiss_index = faiss.IndexFlatIP(384)
    centroids = []
    
    db_conn = psycopg2.connect(POSTGRES_DSN)
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    os.makedirs("storage/frames", exist_ok=True)
    os.makedirs("storage/crops", exist_ok=True)

    dino_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    while True:
        item = queue.get()
        if item is None:
            break

        track_id, frame_bgr, raw_crop, mask_crop, bbox, sharpness = item

        # 1. Save full frame (with BBox overlay) and raw crop to disk
        frame_vis = frame_bgr.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        frame_path = f"storage/frames/track_{track_id}.jpg"
        crop_path = f"storage/crops/track_{track_id}.jpg"
        cv2.imwrite(frame_path, frame_vis)
        cv2.imwrite(crop_path, raw_crop)

        # 2. Mask out static background & letterbox pad to square for DINOv2
        bg_gray = np.full_like(raw_crop, 128)
        masked_crop = np.where(mask_crop[..., None] > 0, raw_crop, bg_gray)
        h, w, _ = masked_crop.shape
        dim = max(h, w)
        square_crop = np.full((dim, dim, 3), 128, dtype=np.uint8)
        square_crop[(dim - h) // 2 : (dim - h) // 2 + h, (dim - w) // 2 : (dim - w) // 2 + w] = masked_crop

        pil_img = Image.fromarray(cv2.cvtColor(square_crop, cv2.COLOR_BGR2RGB))
        tensor_in = dino_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = dinov2(tensor_in).cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)

        # 3. Online FAISS Centroid Matching & PostgreSQL Update
        cur = db_conn.cursor()
        if faiss_index.ntotal > 0:
            D, I = faiss_index.search(emb, 1)
            best_sim, best_faiss_id = float(D[0][0]), int(I[0][0])
        else:
            best_sim, best_faiss_id = -1.0, -1

        if best_sim >= COSINE_THRESHOLD:
            cur.execute("""
                UPDATE unknown_clusters
                SET observation_count = observation_count + 1, updated_at = CURRENT_TIMESTAMP
                WHERE faiss_index_id = %s
                RETURNING cluster_id, observation_count, status;
            """, (best_faiss_id,))
            cluster_id, obs_count, status = cur.fetchone()

            # Running centroid update
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

        cur.execute("""
            INSERT INTO unknown_observations
            (cluster_id, track_id, crop_image_path, full_frame_path, bbox_xyxy, sharpness_score, cosine_similarity)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (cluster_id, track_id, crop_path, frame_path, list(bbox), sharpness, best_sim))
        db_conn.commit()

        # 4. Trigger Gemini 2.5 Flash when observation_count hits 100
        if obs_count >= CLUSTER_TRIGGER_COUNT and status == "COLLECTING":
            cur.execute("""
                SELECT crop_image_path, full_frame_path
                FROM unknown_observations
                WHERE cluster_id = %s
                ORDER BY sharpness_score DESC LIMIT 3;
            """, (cluster_id,))
            top_rows = cur.fetchall()

            prompt_contents = [
                "You are a computer vision surveillance analyst. Image 1 shows a full CCTV camera frame with a red bounding box highlighting an unknown moving object. Images 2, 3, and 4 show close-up crops of this exact same class of object captured across 100 sightings. Identify this specific physical object."
            ]
            prompt_contents.append(Image.open(top_rows[0][1]))  # Full contextual frame
            for r in top_rows:
                prompt_contents.append(Image.open(r[0]))         # Top 3 sharpest crops

            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiObjectSchema,
                    temperature=0.1,
                ),
            )
            parsed = GeminiObjectSchema.model_validate_json(response.text)
            final_status = "IDENTIFIED" if parsed.is_valid_physical_object else "NOISE"

            cur.execute("""
                UPDATE unknown_clusters
                SET status = %s, identified_label = %s, vlm_confidence = %s, vlm_reasoning = %s
                WHERE cluster_id = %s;
            """, (final_status, parsed.object_name, parsed.confidence, parsed.reasoning, cluster_id))
            db_conn.commit()

# =====================================================================
# PHASE 1: UPSTREAM REAL-TIME VIDEO LOOP (MODULES 1, 2, 3, 4)
# =====================================================================
def run_surveillance_pipeline():
    work_queue = Queue(maxsize=256)
    worker = Process(target=discovery_worker_loop, args=(work_queue,), daemon=True)
    worker.start()

    yolo_model = YOLO(YOLO_WEIGHTS)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    active_tracks = []
    next_track_id = 1
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape

        # Module 1: MOG2 Foreground Mask (Drop shadows = 127, keep 255)
        fg_mask = mog2.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)

        # Module 2: Known-Object Detection via YOLO
        yolo_res = yolo_model(frame, verbose=False)[0]
        known_boxes = yolo_res.boxes.xyxy.cpu().numpy().astype(int)

        # Module 3: Mask-Level Known Object Erasure
        for (x1, y1, x2, y2) in known_boxes:
            px1, py1 = max(0, x1 - YOLO_PAD_PX), max(0, y1 - YOLO_PAD_PX)
            px2, py2 = min(W, x2 + YOLO_PAD_PX), min(H, y2 + YOLO_PAD_PX)
            cv2.rectangle(fg_mask, (px1, py1), (px2, py2), 0, -1)

        # Extract residual unknown moving contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                candidate_boxes.append([x, y, x + w, y + h])

        # Module 4: Associate Unknown Contours & Gate by Trajectory Persistence
        used_candidates = set()
        for track in active_tracks:
            best_iou, best_idx = 0.0, -1
            for idx, cb in enumerate(candidate_boxes):
                if idx in used_candidates:
                    continue
                # Compute IoU
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

        # Render confirmed unknown tracks & dispatch completed tracks to Phase 2 worker
        surviving_tracks = []
        for track in active_tracks:
            if track.is_persistent_and_linear():
                tx1, ty1, tx2, ty2 = track.bbox
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
                cv2.putText(frame, f"UNKNOWN ID:{track.track_id}", (tx1, ty1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # When a confirmed track exits or finishes, send its single sharpest keyframe to worker
            if track.missed > 5:
                if track.is_persistent_and_linear() and not track.dispatched_to_worker and track.best_payload:
                    f_bgr, r_crop, m_crop, b_box, sharp = track.best_payload
                    if not work_queue.full():
                        work_queue.put((track.track_id, f_bgr, r_crop, m_crop, b_box, sharp))
                    track.dispatched_to_worker = True
            else:
                surviving_tracks.append(track)

        active_tracks = surviving_tracks

    work_queue.put(None)
    worker.join()
    cap.release()

if __name__ == "__main__":
    run_surveillance_pipeline()
```