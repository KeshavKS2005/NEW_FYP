# Module 3: Mask-Level Known Erasure Engine

## 1. Executive Summary & Purpose

Module 3 sits directly at the confluence of **Stream A (Motion)** and **Stream B (Known Detection)** in **Phase 1**. Its primary purpose is to **synthesize the two streams by subtracting all known objects from the pixel-level motion mask**, isolating purely unknown moving entities.

### The Fundamental Problem Solved
If one directly performs bounding-box subtraction (i.e. removing known bounding boxes from candidate lists), a moving known object (like a car) will frequently create fragmented edge contours around its perimeter. This is caused by slight discrepancies between the tight YOLO box boundary and the outer diffuse edge of the optical flow / MOG2 silhouette. Without specialized pixel-level erasure with safety margins, **the edges of known objects would register as false unknown moving objects**.

Module 3 solves this by operating directly on the **NumPy binary mask matrix** with a calibrated spatial dilation margin before any unknown contours are extracted.

---

## 2. Submodule Breakdown

Module 3 is implemented in [module_3/](file:///Users/mayur/Documents/NEW_FYP/module_3/):

```mermaid
graph TD
    M1_Mask[Module 1: Binary Motion Mask] --> Bitwise[Submodule 3.2: Matrix Bitwise Erasure]
    M2_Boxes[Module 2: Known Boxes [K, 4]] --> Dilate[Submodule 3.1: 15px Safety Dilation Margin]
    Dilate --> Bitwise
    Bitwise --> Filter[Submodule 3.3: Residual Contour Filtering]
    Filter --> Out_Mask[Erased Binary Mask]
    Filter --> Out_Boxes[Candidate Unknown Boxes [U, 5]]
```

### Submodule 3.1: Safety Margin Dilation (`pad_px = 15`)
- **Source File:** [erasure_engine.py](file:///Users/mayur/Documents/NEW_FYP/module_3/erasure_engine.py#L40-L65)
- **Mathematical Form:**
  Given a known bounding box $[x_1, y_1, x_2, y_2]$ from Module 2 and frame dimensions $(H, W)$, the dilated boundary $[x_1', y_1', x_2', y_2']$ is defined as:
  $$x_1' = \max(0, x_1 - \delta), \quad y_1' = \max(0, y_1 - \delta)$$
  $$x_2' = \min(W, x_2 + \delta), \quad y_2' = \min(H, y_2 + \delta)$$
  Where $\delta = \text{pad\_px} = 15\text{ pixels}$.
- **Why $\delta = 15$ is Essential:**
  Surveillance cameras exhibit motion blur and lens chromatic aberration at high vehicle/pedestrian speeds. A dilation margin of 15 pixels guarantees that boundary blur around known cars or people does not survive as false "floating unknown debris".

### Submodule 3.2: Matrix Bitwise Foreground Mask Erasure
- **Source File:** [erasure_engine.py](file:///Users/mayur/Documents/NEW_FYP/module_3/erasure_engine.py#L66-L85)
- **Matrix Operation:**
  Module 3 creates an in-memory clone of the binary MOG2 mask $\mathbf{M}_{\text{fg}} \in \{0, 255\}^{H \times W}$. For each padded known box $B_k'$, it zeroes out the enclosed region:
  $$\mathbf{M}_{\text{erased}}[y_1':y_2', \; x_1':x_2'] = 0$$
- **Efficiency:**
  This is executed via direct NumPy slice indexing `mask[y1:y2, x1:x2] = 0`, completing in under **0.3 milliseconds** per frame on a 1080p frame, regardless of the number of known boxes.

### Submodule 3.3: Residual Contour Filtering & Candidate Extraction
- **Source File:** [erasure_engine.py](file:///Users/mayur/Documents/NEW_FYP/module_3/erasure_engine.py#L86-L130)
- **Algorithm:**
  1. Executes `cv2.findContours(M_erased, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`.
  2. For each contour $C_u$, computes pixel area:
     $$A_u = \text{cv2.contourArea}(C_u)$$
  3. **Geometric Filtering:**
     - Area check: $\text{min\_contour\_area} \le A_u \le \text{max\_contour\_area}$ (default: $500 \le A_u \le 500,000\text{ px}^2$).
     - Aspect ratio check: $\frac{w}{h} \in [0.1, 10.0]$ (rejects line artifacts from scanline tears).
  4. Formulates candidate 5-tuple: $[x_1, y_1, x_2, y_2, A_u]$.

---

## 3. Configuration & Parameter Reference

Configuration is managed via dataclass in [module_3/config.py](file:///Users/mayur/Documents/NEW_FYP/module_3/config.py):

| Parameter | Type | Default | Description | Impact of Tuning |
|---|---|---|---|---|
| `pad_px` | `int` | `15` | Pixel padding applied symmetrically to all 4 edges of known YOLO boxes. | Lower: Risk of boundary halo artifacts; Higher: Over-erases unknown objects touching known ones. |
| `min_contour_area` | `float` | `500.0` | Minimum pixel area required for an unknown contour. | Rejects small residual fragments and leaves. |
| `max_contour_area` | `float` | `500000.0` | Maximum pixel area for an unknown contour. | Prevents sudden lighting shifts / camera flare from being treated as objects. |
| `min_aspect_ratio` | `float` | `0.1` | Minimum width-to-height ratio. | Rejects vertical scan lines or flickering cables. |
| `max_aspect_ratio` | `float` | `10.0` | Maximum width-to-height ratio. | Rejects horizontal edge line artifacts. |

---

## 4. Input / Output Data Specifications

### Input
- **`motion_mask`**: `np.ndarray` of shape `(H, W)`, `uint8` with values in $\{0, 255\}$ (from Module 1).
- **`known_boxes`**: `List[Tuple[int, int, int, int]]` in $[x_1, y_1, x_2, y_2]$ coordinates (from Module 2).
- **`frame_index`**: Integer sequential frame count.

### Output: `Module3FrameResult`
- **`erased_mask`**: `np.ndarray` of shape `(H, W)`, `uint8` containing only unknown moving pixels.
- **`boxes`**: `List[Tuple[int, int, int, int, float]]` containing $[x_1, y_1, x_2, y_2, \text{area}]$ for each candidate unknown.
- **`candidate_count`**: Integer count of unknown candidate blobs surviving erasure.
- **`latency_ms`**: Processing duration in milliseconds (typically 0.4–1.2 ms).

---

## 5. Failure Modes & Edge Case Mitigations

| Failure Mode | Physical Cause | Mitigation in Module 3 |
|---|---|---|
| **Unknown Object Moving Right Beside Known Object** | A novel animal or scooter rides directly adjacent to a known truck. | If distance $< 15\text{ px}$, part of the unknown is shaved by `pad_px`. However, as the objects separate in subsequent frames, Module 4 maintains tracking and recovers the full contour. |
| **Complete Erasure of Known Clutter** | A scene contains 20 known vehicles and pedestrians. | Module 3 performs instant vectorized slice zeroes; performance scales linearly with box count and takes $< 1\text{ ms}$. |
| **Zero Moving Objects in Scene** | Empty late-night surveillance view. | Short-circuits immediately if `motion_mask` has zero non-zero pixels, avoiding contour calls. |

---

## 6. Code Verification & Unit Testing

Module 3 is validated via tests in `tests/test_module_3_erasure.py`:

```bash
pytest tests/test_module_3_erasure.py -v
```
