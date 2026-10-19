# Module 5: Compound Blob Decomposition (Method 2: FastSAM + DINOv2)

## 1. Executive Summary & Purpose

Module 5 is the mathematical core of **Phase 2 (Asynchronous Discovery Worker)**. It resolves one of the most stubborn failure modes in computer vision and motion detection: **The Compound Blob Merging Problem**.

### The Fundamental Problem Solved
When classical background subtraction (MOG2) detects unknown moving objects, **any two objects that physically touch and move in synchrony merge into a single connected silhouette**. 
- *Classic Example:* An unknown Person riding an unknown Bicycle, or an unknown Worker pushing a Handcart.
- *Consequence without Module 5:* DINOv2 embeds the entire merged blob as a single vector. The system learns a nonsensical hybrid class ("person-bicycle-amalgam") that corrupts both the vector space and subsequent YOLO retraining.

Module 5 implements **Method 2 (`FastSAM` Boundary Proposals + `DINOv2` Mask-Guided Spatial Patch Pooling & Iterative Cosine Merging)**. It autonomously decomposes touching compound silhouettes into separate, isolated sub-objects (e.g. `Bicycle` and `Rider`), producing tight individual bounding boxes and clean binary masks.

---

## 2. Submodule Breakdown

Module 5 is implemented in [module_5/](file:///Users/mayur/Documents/NEW_FYP/module_5/):

```mermaid
graph TD
    Keyframe[Confirmed Keyframe + MOG2 Mask] --> FastSAM[Submodule 5.1: FastSAM Part Proposals]
    FastSAM --> Intersect[Submodule 5.2: MOG2 Intersection & Static Part Pruning]
    Intersect --> DINO_Pool[Submodule 5.3: DINOv2 Spatial Patch Pooling]
    DINO_Pool --> Merge[Submodule 5.4: Iterative Agglomerative Cosine Merging]
    Merge -- "If Single Mask" --> MaskCut[Submodule 5.5: Normalized Cut / MaskCut Fallback]
    Merge -- "Split Confirmed" --> SubObjs[List of Isolated Sub-Objects [1..S]]
    MaskCut --> SubObjs
    SubObjs --> Dedup[Submodule 5.6: Keyframe Deduplicator]
```

### Submodule 5.1: FastSAM High-Resolution Boundary Proposals
- **Source File:** [compound_decomposer.py](file:///Users/mayur/Documents/NEW_FYP/module_5/compound_decomposer.py#L35-L65)
- **Model:** `FastSAM-s.pt` (Fast Segment Anything Model, CNN-based real-time instance segmenter).
- **Execution:**
  Runs on the high-resolution crop of the candidate blob:
  ```python
  sam_res = fastsam(raw_crop, retina_masks=True, conf=0.35, iou=0.7, verbose=False)[0]
  ```
  FastSAM acts as a class-agnostic edge proposal engine, proposing high-fidelity boundary masks $\{\hat{\mathbf{M}}_1, \hat{\mathbf{M}}_2, \dots, \hat{\mathbf{M}}_P\}$ for every visual texture discontinuity.

### Submodule 5.2: MOG2 Mask Intersection & Static Part Pruning
- **Source File:** [compound_decomposer.py](file:///Users/mayur/Documents/NEW_FYP/module_5/compound_decomposer.py#L90-L115)
- **Problem Solved:** FastSAM often proposes masks for static background textures visible inside the bounding box crop (e.g., pavement markings, road curbs).
- **Pruning Formula:**
  Each proposed part mask $\hat{\mathbf{M}}_p$ is intersected with the MOG2 residual motion mask $\mathbf{M}_{\text{fg}}$:
  $$\mathbf{M}_p^{\text{inter}} = \hat{\mathbf{M}}_p \odot (\mathbf{M}_{\text{fg}} > 0)$$
  Static background masks are discarded using the foreground coverage threshold:
  $$\frac{\sum \mathbf{M}_p^{\text{inter}}}{\sum \mathbf{M}_{\text{fg}}} \ge \text{PART\_MIN\_FG\_RATIO} = 0.06 \; (6\%)$$

### Submodule 5.3: DINOv2 Mask-Guided Spatial Patch Feature Pooling
- **Source File:** [compound_decomposer.py](file:///Users/mayur/Documents/NEW_FYP/module_5/compound_decomposer.py#L116-L140)
- **Grid Architecture:**
  The crop is passed through `dinov2_vits14_reg`. From `feats["x_norm_patchtokens"]`, Module 5 extracts the dense $16 \times 16 = 256$ spatial patch tokens:
  $$\mathbf{T} \in \mathbb{R}^{256 \times 384}$$
- **Mask-Weighted Average Pooling:**
  For each valid sub-mask $\mathbf{M}_p^{\text{inter}}$, it is resized to $16 \times 16$ via area interpolation to obtain spatial weights $\mathbf{w}_p \in \mathbb{R}^{256}$:
  $$\mathbf{v}_p = \frac{\sum_{i=1}^{256} w_{p, i} \cdot \mathbf{T}_i}{\sum_{i=1}^{256} w_{p, i} + \epsilon}, \quad \hat{\mathbf{v}}_p = \frac{\mathbf{v}_p}{\|\mathbf{v}_p\|_2}$$
  Yields a 384-D semantically rich embedding vector for each proposed part.

### Submodule 5.4: Iterative Agglomerative Cosine Merging (UnSAM Conquer Stage)
- **Source File:** [compound_decomposer.py](file:///Users/mayur/Documents/NEW_FYP/module_5/compound_decomposer.py#L141-L190)
- **The Challenge:** FastSAM over-segments (e.g., a person's shirt, pants, and shoes are proposed as 3 separate masks; a bicycle's wheels, handlebar, and pedals are proposed as 4 separate masks).
- **Merging Algorithm:**
  Module 5 initializes each part as an independent cluster $\mathcal{G}_i = \{i\}$ with vector $\hat{\mathbf{v}}_i$.
  1. Computes all pairwise cosine similarities:
     $$S_{ij} = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{v}}_j = \cos(\theta_{ij})$$
  2. Finds maximum similarity pair $(i^*, j^*) = \arg\max_{i < j} S_{ij}$.
  3. If $S_{i^*j^*} \ge \text{PART\_MERGE\_COSINE\_THRESH} = 0.62$:
     - Merge group: $\mathcal{G}_{i^*} \leftarrow \mathcal{G}_{i^*} \cup \mathcal{G}_{j^*}$
     - Update centroid: $\mathbf{v}_{i^*} \leftarrow \frac{\mathbf{v}_{i^*} + \mathbf{v}_{j^*}}{2}, \; \hat{\mathbf{v}}_{i^*} = \frac{\mathbf{v}_{i^*}}{\|\mathbf{v}_{i^*}\|_2}$
     - Pop cluster $j^*$ and repeat.
  4. Merges corresponding masks using morphological closing with a $17 \times 17$ elliptical kernel:
     $$\mathbf{M}_{\text{obj}} = \left( \bigcup_{p \in \mathcal{G}} \mathbf{M}_p \right) \bullet \mathbf{K}_{17\times 17}$$
- **Outcome:** Shirts and pants fuse into `Person`; wheels, pedals, and handlebars fuse into `Bicycle`. Because person clothing vectors and bicycle metallic frame vectors have cosine similarity $\approx 0.38 < 0.62$, they remain **strictly separated**.

### Submodule 5.5: Fallback: Normalized Cut / MaskCut Fiedler Eigenvector Splitting
- **Source File:** [compound_decomposer.py](file:///Users/mayur/Documents/NEW_FYP/module_5/compound_decomposer.py#L191-L235)
- **Purpose:** If FastSAM fails to detect multiple parts (e.g., low contrast lighting causes FastSAM to output only 1 mask for Rider + Bike), Module 5 activates a mathematical spectral graph cut fallback (MaskCut / CutLER).
- **Spectral Graph Formulation:**
  1. Takes the active foreground patch features $\mathbf{F}_{\text{fg}} \in \mathbb{R}^{K \times 384}$ ($K \ge 16$).
  2. Forms affinity matrix $\mathbf{W} \in \mathbb{R}^{K \times K}$:
     $$W_{ij} = \begin{cases} \mathbf{F}_i \cdot \mathbf{F}_j & \text{if } \mathbf{F}_i \cdot \mathbf{F}_j \ge 0.20 \\ 10^{-5} & \text{otherwise} \end{cases}$$
  3. Degree matrix $D_{ii} = \sum_j W_{ij}$. Normalized symmetric Laplacian:
     $$\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}$$
  4. Extracts the **Fiedler Eigenvector** (second smallest eigenvector $\mathbf{v}_1$):
     $$\mathbf{f} = \mathbf{D}^{-1/2} \mathbf{v}_1$$
  5. Bi-partitions foreground patches by thresholding at $\mu_f = \text{mean}(\mathbf{f})$:
     $$\Omega_A = \{i \mid f_i > \mu_f\}, \quad \Omega_B = \{i \mid f_i \le \mu_f\}$$
  6. Evaluates inter-cluster versus intra-cluster separation score:
     $$\Delta_{\text{sep}} = \frac{S_{AA} + S_{BB}}{2} - S_{AB} \ge \text{MASKCUT\_SEP\_THRESH} = 0.22$$
  7. If confirmed, splits the single blob into two crisp sub-objects.

### Submodule 5.6: Track Keyframe Deduplicator
- **Source File:** [keyframe_deduplicator.py](file:///Users/mayur/Documents/NEW_FYP/module_5/keyframe_deduplicator.py)
- **Role:** Ensures that if consecutive tracks generate near-identical decomposed crops of the same physical object within short time windows, only the distinct exemplar is advanced.

---

## 3. Configuration & Parameter Reference

Configuration is managed via dataclass in [module_5/config.py](file:///Users/mayur/Documents/NEW_FYP/module_5/config.py):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fastsam_weights` | `str` | `"FastSAM-s.pt"` | Pretrained weights for FastSAM. |
| `part_min_fg_ratio` | `float` | `0.06` | Minimum fraction of MOG2 moving pixels a FastSAM part must cover. |
| `part_merge_cosine_thresh` | `float` | `0.62` | Cosine similarity threshold for fusing sub-parts into a single sub-object. |
| `maskcut_sep_thresh` | `float` | `0.22` | Eigenvector separation threshold to trigger Normalized Cut fallback. |
| `min_sub_object_area` | `int` | `350` | Minimum bounding box pixel area for decomposed sub-objects. |
| `device` | `str` | `"cuda"` / `"cpu"` | Compute device. |

---

## 4. Input / Output Data Specifications

### Input
- **`frame_bgr`**: Full clean CCTV frame `(H, W, 3)`.
- **`raw_crop`**: BGR image crop of the merged blob `(crop_h, crop_w, 3)`.
- **`mask_crop`**: Binary MOG2 mask of the merged blob `(crop_h, crop_w)`.
- **`merged_bbox`**: Absolute coordinates $[x_1, y_1, x_2, y_2]$ in full frame.

### Output: `List[DecomposedSubObject]`
Each decomposed item contains:
- **`bbox`**: Tight absolute bounding box $[x_1, y_1, x_2, y_2]$ in full frame coordinates.
- **`raw_crop`**: Clean BGR crop of the isolated sub-object.
- **`sub_mask`**: Isolated binary mask `(h, w)` with touching neighbor pixels zeroed.
- **`method`**: String tag (`"METHOD2_FASTSAM_DINOV2"`, `"FALLBACK_MASKCUT"`, or `"SINGLE"`).

---

## 5. Failure Modes & Edge Case Mitigations

| Failure Mode | Physical Cause | Mitigation in Module 5 |
|---|---|---|
| **Single Isolated Mover** | A novel autonomous robot travels alone without touching any other entity. | FastSAM and merging logic recognize high intra-similarity; outputs single sub-object (`SINGLE`) with zero over-fragmentation. |
| **FastSAM Under-Segmentation** | Extreme backlighting blinds FastSAM boundary edges. | Submodule 5.5 automatically activates Normalized Cut Fiedler eigenvector spectral bi-partitioning. |
| **Tiny Sub-Pixel Fragments** | Pedestrian shoe tip splits as a separate 15-pixel blob. | Enforced area floor (`min_sub_object_area = 350 px`) discards negligible fragments. |

---

## 6. Code Verification & Unit Testing

Module 5 is verified via tests in `tests/test_module_5_decomposition.py`:

```bash
pytest tests/test_module_5_decomposition.py -v
```
