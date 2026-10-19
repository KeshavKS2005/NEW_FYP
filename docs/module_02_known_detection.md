# Module 2: Known-Object Detection (Stream B)

## 1. Executive Summary & Purpose

Module 2 acts as the **closed-world anchor** in **Phase 1** of the surveillance pipeline. It continuously scans each surveillance frame using the currently deployed **Champion YOLO model** (`champion_latest.pt`, bootstrapped from `yolo11n.pt`).

Its critical role is to identify every entity that the system **already knows how to recognize**. When combined with Module 3, these known bounding boxes are used to erase recognized objects from the motion mask, leaving behind only the unknown residual movers for discovery.

Crucially, Module 2 supports **live in-memory hot-swapping**: when Phase 3 promotes a retrained model, Module 2 swaps its active weights on the fly without restarting the video stream or dropping frames.

---

## 2. Submodule Breakdown

Module 2 is organized into four distinct submodules located in [stream_b/](file:///Users/mayur/Documents/NEW_FYP/stream_b/):

```mermaid
graph TD
    A[Raw BGR Frame] --> B[Submodule 2.1: YOLOv11 Champion Detector]
    B --> C[Submodule 2.2: Holdout Class Filter]
    C --> D[Active Known Detections [K, 4]]
    D --> E[Submodule 2.4: Stream B Visualizer]
    F[Module 12 Promotion Signal] --> G[Submodule 2.3: Zero-Downtime Dynamic Weight Reloader]
    G --> B
```

### Submodule 2.1: YOLOv11 Champion Inference Engine
- **Source File:** [detector.py](file:///Users/mayur/Documents/NEW_FYP/stream_b/detector.py#L30-L75)
- **Model Architecture:** Ultralytics YOLOv11 (defaults to `yolo11n.pt` for real-time 30+ FPS edge execution).
- **Execution Mechanism:**
  - Runs in half-precision (`fp16`) on CUDA or optimized float32 on Apple Silicon / CPU.
  - Applies Non-Maximum Suppression (NMS) with `conf_threshold = 0.35` and `iou_threshold = 0.45`.
  - Extracts standard 4-coordinate bounding boxes $[x_1, y_1, x_2, y_2]$, confidence scores $c \in [0, 1]$, and class IDs $k \in \{0, 1, \dots, C-1\}$.

### Submodule 2.2: Holdout Class Gating (Open-World Evaluation Protocol)
- **Source File:** [detector.py](file:///Users/mayur/Documents/NEW_FYP/stream_b/detector.py#L76-L95)
- **Purpose:** In benchmark evaluation, standard pretrained models already know COCO classes like `bicycle` (ID 1) or `motorcycle` (ID 3). To test the discovery pipeline on standard datasets, Module 2 can artificially "forget" specified classes by filtering them out.
- **Mechanism:**
  - Configured via environment variable: `TEST_HOLDOUT_CLASS_IDS="1,3"`.
  - When the detector predicts an object with class ID in `holdout_class_ids`, that detection is **dropped from the known output list**:
    $$\mathbf{D}_{\text{known}} = \{ (b, c, k) \in \mathbf{D}_{\text{all}} \mid k \notin \mathcal{H}_{\text{holdout}} \}$$
  - As a result, Module 3 will NOT erase the holdout object from the motion mask, allowing the downstream pipeline to discover, cluster, query Gemini, and retrain on it as if it were a truly novel real-world object.

### Submodule 2.3: Zero-Downtime Dynamic Weight Reloader
- **Source File:** [detector.py](file:///Users/mayur/Documents/NEW_FYP/stream_b/detector.py#L96-L125)
- **Purpose:** Enables live hot-swapping of weights promoted by Module 12 without stream interruption.
- **Mechanism:**
  - The Phase 1 tracker loop monitors `model_reload_event.is_set()` on every video frame.
  - When set, it invokes:
    ```python
    detector.reload_weights("models/champion_latest.pt")
    ```
  - The detector re-initializes the PyTorch model instance in place, updates internal class mappings, resets CUDA cache, and clears the event flag.
  - Latency is under 80 ms, completed during an inter-frame interval.

### Submodule 2.4: Stream B Visualizer & Standalone Runner
- **Source Files:** [visualizer.py](file:///Users/mayur/Documents/NEW_FYP/stream_b/visualizer.py), [run_stream_b.py](file:///Users/mayur/Documents/NEW_FYP/stream_b/run_stream_b.py)
- **Features:**
  - Renders bounding boxes color-coded by class with confidence percentages.
  - Displays class label tags and active model version.
  - Standalone debugging mode with custom video inputs and webcam feeds.

---

## 3. Configuration & Parameter Reference

Configuration is managed via dataclass in [stream_b/config.py](file:///Users/mayur/Documents/NEW_FYP/stream_b/config.py):

| Parameter | Type | Default | Description | Impact of Tuning |
|---|---|---|---|---|
| `model_path` | `str` | `"models/champion_latest.pt"` | Filepath to active Champion YOLO weights. | Points to production model; updated during hot-swap. |
| `conf_threshold` | `float` | `0.35` | Minimum confidence score to register a known detection. | Lower: Erases more known objects (fewer false unknown alarms); Higher: Leaves borderline knowns for discovery. |
| `iou_threshold` | `float` | `0.45` | NMS IoU overlap threshold. | Lower: Reduces duplicate boxes; Higher: Preserves dense crowds. |
| `holdout_class_ids` | `Set[int]` | `set()` | Set of COCO class indices to treat as unknown. | Set to `{1}` to benchmark autonomous discovery of bicycles. |
| `device` | `str` | `"cuda"` / `"cpu"` | Compute device for inference. | Autodetects CUDA, MPS (Apple Silicon), or CPU. |

---

## 4. Input / Output Data Specifications

### Input
- **Type:** BGR Video Frame (`np.ndarray`)
- **Dimensions:** `(H, W, 3)`
- **Frame Index:** Sequential integer `frame_idx`

### Output: `StreamBFrameResult`
- **`boxes`**: `List[Tuple[int, int, int, int]]` in $[x_1, y_1, x_2, y_2]$ absolute coordinates.
- **`scores`**: `List[float]` confidence scores $c \in [0.0, 1.0]$.
- **`class_ids`**: `List[int]` integer class IDs corresponding to model class catalog.
- **`class_names`**: `List[str]` human-readable class names (e.g., `"car"`, `"person"`).
- **`latency_ms`**: Float inference time in milliseconds (typically 6–12 ms on GPU).

---

## 5. Failure Modes & Edge Case Mitigations

| Failure Mode | Physical Cause | Mitigation in Module 2 |
|---|---|---|
| **Partial Occlusion of Known Object** | A known car is 60% hidden behind a billboard; YOLO confidence drops to 0.28. | Downstream Module 3 uses 15px bounding box padding. Even with slightly loose boxes, spatial dilation covers residual edges. |
| **Model Weight File Corruption** | A retrained model checkpoint is corrupted during write. | Handled by Module 12's atomic staging swap (`os.replace`). The champion file is guaranteed to be a complete, fully flushed binary. |
| **Rapid Successive Hot-Swaps** | Multiple novel classes discovered in rapid sequence. | `reload_weights()` is thread-safe and protected by inter-process event flags, preventing concurrent reload collisions. |

---

## 6. Code Verification & Standalone Execution

To verify Module 2 independently:

```bash
python stream_b/run_stream_b.py \
    --video-path "datasets/test_videos/bike_traffic_montreal.webm" \
    --weights "models/champion_latest.pt" \
    --conf 0.35
```
