"""
Module 5: Compound Blob Decomposer (Method 2: FastSAM + DINOv2)
Decomposes merged moving blobs (such as Rider + Bicycle) into isolated sub-objects
with tight bounding boxes [x1, y1, x2, y2] and sub-object binary masks.
Adheres strictly to Module 5 in fyp_architecture_final_object.md.
"""

from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import FastSAM

from module_5.config import Module5Config, DecomposedSubObject


class CompoundBlobDecomposer:
    """
    Method 2 Compound Blob Decomposer:
    - FastSAM boundary proposals on raw crop.
    - DINOv2 spatial patch token grid extraction.
    - Mask-guided feature pooling.
    - Iterative Agglomerative Cosine Merging (UnSAM conquer stage).
    - Normalized Cut (MaskCut) Fiedler eigenvector fallback.
    """

    def __init__(self, config: Optional[Module5Config] = None, dinov2_model: Optional[Any] = None):
        self.config = config or Module5Config()
        self.device = self.config.device
        if self.device != "cpu" and not torch.cuda.is_available():
            self.device = "cpu"

        # Load FastSAM
        try:
            self.fastsam = FastSAM(self.config.fastsam_weights)
        except Exception:
            # Fallback if weights not downloaded yet
            self.fastsam = FastSAM("FastSAM-s.pt")

        # Load or use shared DINOv2 model
        if dinov2_model is not None:
            self.dinov2 = dinov2_model
        else:
            try:
                self.dinov2 = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14_reg"
                ).to(self.device).eval()
            except Exception:
                self.dinov2 = None

        self.bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def decompose(
        self,
        frame_bgr: np.ndarray,
        raw_crop: np.ndarray,
        mask_crop: np.ndarray,
        merged_bbox: Tuple[int, int, int, int]
    ) -> List[DecomposedSubObject]:
        """
        Decomposes a merged candidate blob into 1..S isolated sub-objects.

        Args:
            frame_bgr: Clean full CCTV frame (H, W, 3).
            raw_crop: BGR crop of the merged candidate blob.
            mask_crop: Binary residual motion mask of the blob (crop_h, crop_w).
            merged_bbox: [x1, y1, x2, y2] bounding box of the crop in full-frame coordinates.

        Returns:
            List of DecomposedSubObject instances.
        """
        x1, y1, x2, y2 = merged_bbox
        crop_h, crop_w = mask_crop.shape[:2]

        if crop_h < 32 or crop_w < 32 or self.dinov2 is None:
            return [DecomposedSubObject(
                bbox=merged_bbox,
                raw_crop=raw_crop.copy(),
                sub_mask=(mask_crop > 0).astype(np.uint8) * 255,
                method="SINGLE"
            )]

        # Step 5.3: Extract DINOv2 dense spatial patch tokens (16x16 = 256 patches, 384-D)
        bg_gray = np.full_like(raw_crop, 128)
        masked_rgb = np.where(mask_crop[..., None] > 0, raw_crop, bg_gray)
        pil_img = Image.fromarray(cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2RGB))
        tensor_in = self.dino_transform(pil_img).unsqueeze(0).to(self.device)

        try:
            feats = self.dinov2.forward_features(tensor_in)
            patch_tokens = feats["x_norm_patchtokens"][0].cpu().numpy()
            norms = np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-8
            patch_tokens = patch_tokens / norms
        except Exception:
            patch_tokens = None

        valid_submasks: List[np.ndarray] = []
        submask_vectors: List[np.ndarray] = []
        total_fg_pixels = max(1, cv2.countNonZero(mask_crop))

        # Step 5.1: Run FastSAM on the high-resolution crop
        try:
            sam_res = self.fastsam(
                raw_crop, device=self.device, retina_masks=True, conf=0.35, iou=0.7, verbose=False
            )[0]
        except Exception:
            sam_res = None

        if sam_res is not None and sam_res.masks is not None and len(sam_res.masks.data) > 1 and patch_tokens is not None:
            for m in sam_res.masks.data.cpu().numpy():
                m_bin = (cv2.resize(m, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
                fg_inter = cv2.bitwise_and(m_bin, (mask_crop > 0).astype(np.uint8))

                # Step 5.2: Drop parts covering < 6% of moving blob
                if cv2.countNonZero(fg_inter) < self.config.part_min_fg_ratio * total_fg_pixels:
                    continue

                w_16 = cv2.resize(fg_inter, (16, 16), interpolation=cv2.INTER_AREA).flatten()
                if np.sum(w_16) < 1e-3:
                    continue

                # Step 5.4: Mask-guided average pooling of patch tokens
                pooled_vec = np.average(patch_tokens, axis=0, weights=w_16)
                pooled_vec = pooled_vec / (np.linalg.norm(pooled_vec) + 1e-8)

                valid_submasks.append(fg_inter * 255)
                submask_vectors.append(pooled_vec)

        # Step 5.5: Iterative Agglomerative Cosine Merging
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

                if best_sim >= self.config.part_merge_cosine_thresh and best_pair is not None:
                    i, j = best_pair
                    groups[i].extend(groups[j])
                    merged_v = (g_vecs[i] + g_vecs[j]) / 2.0
                    g_vecs[i] = merged_v / (np.linalg.norm(merged_v) + 1e-8)
                    groups.pop(j)
                    g_vecs.pop(j)
                else:
                    break

            if len(groups) > 1:
                decomposed_list: List[DecomposedSubObject] = []
                for g in groups:
                    comb_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                    for idx in g:
                        comb_mask = cv2.bitwise_or(comb_mask, valid_submasks[idx])
                    comb_mask = cv2.morphologyEx(comb_mask, cv2.MORPH_CLOSE, self.bridge_kernel)

                    pts = cv2.findNonZero(comb_mask)
                    if pts is not None:
                        sx, sy, sw, sh = cv2.boundingRect(pts)
                        if sw * sh >= self.config.min_sub_object_area:
                            abs_box = (x1 + sx, y1 + sy, x1 + sx + sw, y1 + sy + sh)
                            s_raw = raw_crop[sy : sy + sh, sx : sx + sw].copy()
                            s_mask = comb_mask[sy : sy + sh, sx : sx + sw].copy()
                            decomposed_list.append(DecomposedSubObject(
                                bbox=abs_box,
                                raw_crop=s_raw,
                                sub_mask=s_mask,
                                method="METHOD2_FASTSAM_DINOV2"
                            ))
                if len(decomposed_list) > 1:
                    return decomposed_list

        # Step 5.6: Fallback Normalized Cuts (MaskCut) Fiedler eigenvector
        if patch_tokens is not None:
            mask_16 = cv2.resize(mask_crop, (16, 16), interpolation=cv2.INTER_AREA)
            fg_idx = np.where(mask_16.flatten() > 64)[0]
            if len(fg_idx) >= 16:
                fg_feats = patch_tokens[fg_idx]
                sim_mat = np.dot(fg_feats, fg_feats.T)
                W = np.where(sim_mat >= 0.20, sim_mat, 1e-5)
                D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-8))
                L_sym = np.eye(len(fg_idx)) - D_inv_sqrt @ W @ D_inv_sqrt

                try:
                    _, eigvecs = np.linalg.eigh(L_sym)
                    fiedler = D_inv_sqrt @ eigvecs[:, 1]
                    split_val = float(np.mean(fiedler))
                    idx_a = np.where(fiedler > split_val)[0]
                    idx_b = np.where(fiedler <= split_val)[0]

                    if len(idx_a) >= 4 and len(idx_b) >= 4:
                        sim_aa = float(np.mean(sim_mat[np.ix_(idx_a, idx_a)]))
                        sim_bb = float(np.mean(sim_mat[np.ix_(idx_b, idx_b)]))
                        sim_ab = float(np.mean(sim_mat[np.ix_(idx_a, idx_b)]))
                        if (((sim_aa + sim_bb) / 2.0) - sim_ab) >= self.config.maskcut_sep_thresh:
                            ncut_list: List[DecomposedSubObject] = []
                            for subset in (fg_idx[idx_a], fg_idx[idx_b]):
                                g16 = np.zeros(256, dtype=np.uint8)
                                g16[subset] = 255
                                up_m = cv2.resize(g16.reshape(16, 16), (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                                sub_m = cv2.bitwise_and(mask_crop, (up_m > 110).astype(np.uint8) * 255)
                                sub_m = cv2.morphologyEx(sub_m, cv2.MORPH_CLOSE, self.bridge_kernel)
                                pts = cv2.findNonZero(sub_m)
                                if pts is not None:
                                    sx, sy, sw, sh = cv2.boundingRect(pts)
                                    if sw * sh >= self.config.min_sub_object_area:
                                        abs_box = (x1 + sx, y1 + sy, x1 + sx + sw, y1 + sy + sh)
                                        s_raw = raw_crop[sy : sy + sh, sx : sx + sw].copy()
                                        s_mask = sub_m[sy : sy + sh, sx : sx + sw].copy()
                                        ncut_list.append(DecomposedSubObject(
                                            bbox=abs_box,
                                            raw_crop=s_raw,
                                            sub_mask=s_mask,
                                            method="FALLBACK_MASKCUT"
                                        ))
                            if len(ncut_list) > 1:
                                return ncut_list
                except Exception:
                    pass

        return [DecomposedSubObject(
            bbox=merged_bbox,
            raw_crop=raw_crop.copy(),
            sub_mask=(mask_crop > 0).astype(np.uint8) * 255,
            method="SINGLE"
        )]
