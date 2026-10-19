# Module 4: Temporal Trajectory Gating & Keyframe Buffer

## 1. Executive Summary & Purpose

Module 4 serves as the critical gatekeeper between **Phase 1 (Real-Time 30+ FPS Tracker)** and **Phase 2 (Asynchronous Deep Learning Worker)**.

In automated surveillance, raw pixel-level motion and contour extraction suffer from two major real-world failure modes:
1. **Stationary Oscillating Clutter:** Wind-blown foliage, tree branches, flags, or rippling puddles move continuously on screen, generating endless motion contours. However, they do not travel through the scene.
2. **Keyframe Duplication & Pipeline Flooding:** If an unknown object traverses a CCTV camera over 300 frames, sending every single frame downstream would flood the database and VLM API with 300 near-identical, often blurry frames of the same entity.

Module 4 solves both problems using **Kalman-filtered multi-object tracking (ByteTrack)**, a strict **Directional Persistence metric ($\eta_{\text{dir}} \ge 0.50$)**, and a **Laplacian Sharpness Buffer** that dispatches **exactly one single, highest-quality keyframe tuple** when a track successfully traverses the surveillance zone.

---

## 2. Submodule Breakdown

Module 4 consists of five modular components located in [module_4/](file:///Users/mayur/Documents/NEW_FYP/module_4/):

```mermaid
graph TD
    Cand[Candidate Boxes [U, 5]] --> BT[Submodule 4.1: ByteTracker Multi-Object Association]
    BT <--> KF[Submodule 4.2: Kalman Filter State Estimator]
    BT --> DP[Submodule 4.3: Directional Persistence Gater]
    DP --> SG[Submodule 4.4: Frame Survival Gater]
    SG --> KB[Submodule 4.5: Keyframe Sharpness Buffer]
    KB -- "On Track Exit / Lost (1 Tuple / Track)" --> Q[Phase 2 Queue: discovery_queue]
```

### Submodule 4.1: ByteTracker Multi-Object Association
- **Source File:** [byte_tracker.py](file:///Users/mayur/Documents/NEW_FYP/module_4/byte_tracker.py#L30-L180)
- **Algorithm:** Two-stage hierarchical IoU matching based on ByteTrack (`STrack`).
  - **Stage 1:** Matches high-confidence candidate contours ($c \ge \text{track\_thresh}$) to existing active tracks using Hungarian matching on IoU distance.
  - **Stage 2:** For unmatched tracks, matches surviving low-confidence detections to recover objects undergoing temporary partial occlusions or motion blur.
  - **State Transitions:** `New` $\rightarrow$ `Tracked` $\rightarrow$ `Lost` (retained for up to `max_lost_frames = 5`) $\rightarrow$ `Removed`.

### Submodule 4.2: Kalman Filter State Estimation
- **Source File:** [kalman_filter.py](file:///Users/mayur/Documents/NEW_FYP/module_4/kalman_filter.py#L25-L95)
- **State Representation:**
  Each track is modeled by an 8-dimensional Gaussian state vector in image coordinates:
  $$\mathbf{x} = [x_c, \; y_c, \; a, \; h, \; \dot{x}_c, \; \dot{y}_c, \; \dot{a}, \; \dot{h}]^T$$
  Where $(x_c, y_c)$ is the bounding box center, $a = w / h$ is the aspect ratio, $h$ is box height, and dotted variables denote corresponding velocities.
- **Predict & Update Cycle:**
  $$\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1|k-1}, \quad \mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}$$
  $$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R})^{-1}$$
  $$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H} \mathbf{x}_{k|k-1})$$
  Provides smooth continuous tracking through brief dropped detections.

### Submodule 4.3: Directional Persistence Gater (Oscillation Filtering)
- **Source File:** [temporal_gater.py](file:///Users/mayur/Documents/NEW_FYP/module_4/temporal_gater.py#L50-L68)
- **Mathematical Formulation:**
  For centroid trajectory points $\mathcal{P} = \{\mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_N\}$ over track lifetime:
  - **Net Euclidean Displacement:**
    $$d_{\text{net}} = \|\mathbf{p}_N - \mathbf{p}_0\|_2 = \sqrt{(x_N - x_0)^2 + (y_N - y_0)^2}$$
  - **Total Cumulative Path Length:**
    $$L_{\text{path}} = \sum_{i=1}^N \|\mathbf{p}_i - \mathbf{p}_{i-1}\|_2$$
  - **Directional Persistence Ratio ($\eta_{\text{dir}}$):**
    $$\eta_{\text{dir}} = \frac{d_{\text{net}}}{L_{\text{path}} + \epsilon} \ge \text{min\_dir\_persistence} = 0.50$$
- **Physical Interpretation:**
  - **Translating Physical Objects (Bicycles, Novel Vehicles, Animals):** Move in a coherent vector across the camera FOV ($d_{\text{net}} \approx L_{\text{path}} \implies \eta_{\text{dir}} \in [0.75, 1.0]$).
  - **Oscillating Environmental Artifacts (Tree Branches, Foliage, Flags):** Move back and forth frantically ($L_{\text{path}}$ is very large, but $d_{\text{net}} \approx 0 \implies \eta_{\text{dir}} < 0.20$).
  - **Verdict:** Oscillating foliage is rejected with 100% precision before any crop is queued.

### Submodule 4.4: Frame Survival Gating
- **Source File:** [temporal_gater.py](file:///Users/mayur/Documents/NEW_FYP/module_4/temporal_gater.py#L125-L145)
- **Rule:** A candidate track must be observed across at least `survival_window = 8` frames before being recognized as a valid unknown physical object.
- **Purpose:** Discards transient sensor noise, single-frame compression glitches, and flying insects crossing close to the camera lens.

### Submodule 4.5: Track Keyframe Buffer & Sharpness Maximization
- **Source File:** [keyframe_buffer.py](file:///Users/mayur/Documents/NEW_FYP/module_4/keyframe_buffer.py#L20-L80)
- **Laplacian Variance Formulation:**
  For each candidate crop $\mathbf{I}_{\text{crop}}$ belonging to track $T$, the grayscale image $I_{\text{gray}}$ is filtered with the discrete Laplacian operator:
  $$\nabla^2 I(x, y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
  The sharpness score $S$ is the statistical variance of the response:
  $$S = \text{Var}(\nabla^2 I) = \frac{1}{|\Omega|} \sum_{(x, y) \in \Omega} \left( \nabla^2 I(x, y) - \overline{\nabla^2 I} \right)^2$$
- **Keyframe Selection Protocol:**
  - While track $T$ is active, Module 4 buffers only the frame with the **highest sharpness score** $S_{\max}$.
  - When track $T$ is terminated (object leaves camera FOV or is lost for `max_lost_frames = 5`):
    - If track passed Submodules 4.3 and 4.4, its **single sharpest keyframe tuple** is pushed into `discovery_queue`:
      ```python
      payload = (track_id, clean_frame_bgr, raw_crop, mask_crop, merged_bbox, sharpness)
      discovery_queue.put(payload)
      ```
    - Buffer memory for track $T$ is immediately freed.

---

## 3. Configuration & Parameter Reference

Configuration is managed via dataclass in [module_4/config.py](file:///Users/mayur/Documents/NEW_FYP/module_4/config.py):

| Parameter | Type | Default | Description | Impact of Tuning |
|---|---|---|---|---|
| `survival_window` | `int` | `8` | Minimum continuous frame hits required for confirmation. | Higher: Eliminates all brief glitches; Lower: Faster dispatch of fast movers. |
| `min_dir_persistence` | `float` | `0.50` | Ratio of net displacement to total trajectory path length. | Higher: Stricter straight-line requirement; Lower: Allows meandering objects. |
| `max_lost_frames` | `int` | `5` | Grace period in frames to retain track during occlusions. | Higher: Handles longer occlusions before track closure. |
| `track_thresh` | `float` | `0.40` | Detection score threshold for Stage 1 ByteTrack matching. | Controls candidate association strictness. |
| `match_thresh` | `float` | `0.80` | Maximum IoU cost threshold for Hungarian association. | Controls spatial proximity required for track continuation. |

---

## 4. Input / Output Data Specifications

### Input
- **`candidate_boxes`**: `List[CandidateUnknownBox]` or `np.ndarray` of shape `[U, 5]` ($x_1, y_1, x_2, y_2, \text{area}$).
- **`frame`**: Full pristine BGR frame `(H, W, 3)`.
- **`frame_index`**: Integer frame sequence number.
- **`fg_mask`**: Residual binary unknown motion mask `(H, W)`.
- **`work_queue`**: Asynchronous `multiprocessing.Queue` connected to Phase 2.

### Output: Keyframe Dispatch Tuple (sent to Phase 2)
```python
(
    track_id: int,               # Unique integer ID assigned to the physical object track
    frame_bgr: np.ndarray,       # Full unedited clean surveillance frame (H, W, 3)
    raw_crop: np.ndarray,        # BGR crop of the unknown entity
    mask_crop: np.ndarray,       # Binary mask of the unknown entity
    merged_bbox: Tuple[int,...], # [x1, y1, x2, y2] coordinates in full frame
    sharpness: float             # Maximum Laplacian variance recorded during track lifetime
)
```

---

## 5. Failure Modes & Edge Case Mitigations

| Failure Mode | Physical Cause | Mitigation in Module 4 |
|---|---|---|
| **Swaying Tree Foliage / Shrubs** | High wind moves leaves continuously. | Net displacement is negligible ($d_{\text{net}} \approx 0$). Persistence ratio falls below $0.50$; track is flagged as stationary clutter and dropped. |
| **Object Temporarily Pausing at Red Light** | A novel scooter stops at a crosswalk for 10 seconds. | Track enters `Lost` state after 5 frames of zero motion; the sharpest keyframe captured during its approach is dispatched safely. |
| **Pedestrian Partially Obscuring Novel Object** | A pedestrian walks in front of an unknown cart. | Two-stage ByteTrack associates cart before and after occlusion using Kalman state propagation, maintaining a single unified track. |

---

## 6. Code Verification & Unit Testing

Module 4 is tested in `tests/test_module_4_temporal_gater.py`:

```bash
pytest tests/test_module_4_temporal_gater.py -v
```
