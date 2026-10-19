# Module 1: Class-Agnostic Motion Extraction (Stream A)

## 1. Executive Summary & Purpose

Module 1 serves as the sensory foundation of **Phase 1** in the surveillance pipeline. Its fundamental purpose is to perform **class-agnostic moving pixel segmentation** on high-frame-rate (30+ FPS) surveillance video. 

Traditional closed-world detectors only look for known patterns (e.g. car, person). In contrast, Module 1 makes no assumptions about semantic class: **if a physical entity undergoes coherent spatial displacement relative to the stationary surveillance background, it is detected**. Conversely, motionless known entities (e.g., parked cars, trash cans) are completely absorbed into the background model, ensuring compute is never wasted on static objects.

---

## 2. Submodule Breakdown

Module 1 is composed of four coordinated submodules located in [stream_a/](file:///Users/mayur/Documents/NEW_FYP/stream_a/):

```mermaid
graph LR
    A[Raw BGR Frame] --> B[Submodule 1.1: MOG2 Subtractor]
    B --> C[Submodule 1.2: Shadow Eliminator]
    C --> D[Submodule 1.3: Morphological Filter]
    D --> E[Binary Motion Mask {0, 255}]
    D --> F[Motion Bounding Boxes [M, 4]]
    E --> G[Submodule 1.4: Stream A Visualizer]
    F --> G
```

### Submodule 1.1: Gaussian Mixture Background Subtraction (`MOG2`)
- **Source File:** [motion_extractor.py](file:///Users/mayur/Documents/NEW_FYP/stream_a/motion_extractor.py#L35-L65)
- **Algorithm:** Adaptive Gaussian Mixture Model (`cv2.createBackgroundSubtractorMOG2`).
- **Equation & Modeling:**
  Each pixel $(x, y)$ over time history $T$ is modeled by a mixture of $K$ Gaussians ($K \in [3, 5]$):
  $$P(X_t) = \sum_{k=1}^K w_{k, t} \cdot \eta(X_t, \boldsymbol{\mu}_{k, t}, \boldsymbol{\Sigma}_{k, t})$$
  Where $w_{k, t}$ is the mixture weight, $\boldsymbol{\mu}_{k, t}$ is the mean color vector, and $\boldsymbol{\Sigma}_{k, t} = \sigma_{k, t}^2 \mathbf{I}$ is the variance.
- **Dynamic Adaptation:** If illumination changes (e.g., passing clouds), the model updates parameters with learning rate $\alpha = 1 / \text{history}$.
- **Default Hyperparameters:**
  - `history = 500`: Accommodates gradual daylight shifts while preserving background stability.
  - `varThreshold = 16.0`: Mahalanobis distance squared gating threshold. Lower values increase sensitivity; 16 provides an optimal balance between sensor noise and moving targets.
  - `detectShadows = True`: Distinguishes true physical motion from cast illumination drop.

### Submodule 1.2: Shadow Elimination & Binary Thresholding
- **Source File:** [motion_extractor.py](file:///Users/mayur/Documents/NEW_FYP/stream_a/motion_extractor.py#L75-L85)
- **Problem Solved:** Cast shadows of vehicles or pedestrians alter pixel RGB values without representing physical objects. If not removed, shadows artificially distort bounding box sizes and merge adjacent objects.
- **Mechanism:**
  - OpenCV MOG2 outputs three values in its raw mask:
    - `0`: Definite static background
    - `127`: Detected cast shadow (chromaticity preserved, luminance decreased)
    - `255`: True physical foreground motion
  - **Hard Thresholding:**
    $$\mathbf{M}_{\text{clean}}(x, y) = \begin{cases} 255 & \text{if } \mathbf{M}_{\text{raw}}(x, y) > 254 \\ 0 & \text{otherwise} \end{cases}$$
  - Pixels with intensity `127` are stripped immediately to zero.

### Submodule 1.3: Morphological Noise Suppression & Contour Filtering
- **Source File:** [motion_extractor.py](file:///Users/mayur/Documents/NEW_FYP/stream_a/motion_extractor.py#L86-L120)
- **Operations:**
  1. **Morphological Opening ($5 \times 5$ Elliptical Kernel):**
     $$\mathbf{M}_{\text{opened}} = (\mathbf{M}_{\text{clean}} \ominus \mathbf{K}) \oplus \mathbf{K}$$
     Removes high-frequency camera sensor noise, compression artifacts, and tiny swaying leaves.
  2. **Contour Extraction:**
     Executes `cv2.findContours(cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`.
  3. **Area Filtering:**
     Discards contours where $\text{Area}(c) < \text{min\_contour\_area}$ (default: $500\text{ px}^2$).
  4. **Output Formulation:**
     Extracts candidate bounding boxes $[x_1, y_1, x_2, y_2]$ and total moving pixel density.

### Submodule 1.4: Real-Time Stream A Visualizer & Standalone Runner
- **Source Files:** [visualizer.py](file:///Users/mayur/Documents/NEW_FYP/stream_a/visualizer.py), [run_stream_a.py](file:///Users/mayur/Documents/NEW_FYP/stream_a/run_stream_a.py)
- **Features:**
  - Overlays green motion bounding boxes with area metrics directly onto the video feed.
  - Generates side-by-side or picture-in-picture view of raw frame alongside the binary MOG2 mask.
  - Outputs real-time FPS telemetry.

---

## 3. Configuration & Parameter Reference

Configuration is managed via dataclass in [stream_a/config.py](file:///Users/mayur/Documents/NEW_FYP/stream_a/config.py):

| Parameter | Type | Default | Description | Impact of Tuning |
|---|---|---|---|---|
| `history` | `int` | `500` | Number of past frames used for background modeling. | Higher: Slower adaptation to lighting; Lower: Objects stopping temporarily fade into background. |
| `var_threshold` | `float` | `16.0` | Mahalanobis variance threshold for pixel classification. | Lower: More sensitive (more noise); Higher: Misses low-contrast movers. |
| `detect_shadows` | `bool` | `True` | Flag to tag cast shadows with gray value (127). | Must be `True` to allow downstream thresholding at 254. |
| `morph_kernel_size` | `Tuple[int, int]` | `(5, 5)` | Dimensions of the elliptical structuring element. | Larger: Merges disjoint fragments, but may erode small distant targets. |
| `min_contour_area` | `float` | `500.0` | Minimum pixel area for motion candidates. | Rejects raindrops, insects, and sensor noise. |

---

## 4. Input / Output Data Specifications

### Input
- **Type:** BGR Video Frame (`np.ndarray`)
- **Dimensions:** `(H, W, 3)`, typically `(1080, 1920, 3)` or `(720, 1280, 3)`
- **Data Type:** `uint8`
- **Frame Index:** Monotonically increasing `int`

### Output: `StreamAFrameResult`
- **`motion_mask`**: `np.ndarray` of shape `(H, W)`, `uint8` with values strictly in $\{0, 255\}$.
- **`boxes`**: `List[Tuple[int, int, int, int]]` in absolute coordinates $[x_1, y_1, x_2, y_2]$.
- **`areas`**: `List[float]` pixel area per detected contour.
- **`latency_ms`**: Float execution time in milliseconds (typically 3–7 ms on 1080p CPU).

---

## 5. Failure Modes & Edge Case Mitigations

| Failure Mode | Physical Cause | Mitigation in Module 1 |
|---|---|---|
| **High-Gain Sensor Noise** | Low-light / night CCTV cameras exhibit random salt-and-pepper grain. | Elliptical morphological opening ($5 \times 5$) and area floor ($500\text{ px}^2$) effectively eliminate isolated noise clusters. |
| **Sudden Global Lighting Shift** | Cloud clearing, auto-exposure adjustment, or hallway light switched on. | Triggers full-frame flash; Module 3 and Module 4 detect anomalously large area and zero directional persistence, discarding the frame. |
| **Cast Shadows on Pavement** | Low sun angle causes long dark silhouettes following vehicles. | MOG2 shadow detection assigns 127 to shadow pixels; hard threshold at $> 254$ isolates only true physical bodies. |
| **Camera Shake / Wind Oscillation** | Pole-mounted CCTV swayed by strong wind gusts. | Handled downstream by Module 4 (Kalman velocity consistency and directional persistence check). |

---

## 6. Code Verification & Standalone Execution

To verify Module 1 independently without running the entire 12-module pipeline:

```bash
python stream_a/run_stream_a.py \
    --video-path "datasets/test_videos/bike_traffic_montreal.webm" \
    --show-mask \
    --min-area 500
```
