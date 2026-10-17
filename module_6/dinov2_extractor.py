"""
Module 6: Neighbor-Suppressed Preprocessor & Global Embedder (DINOv2 with Registers)
Replaces both background and touching neighbor pixels with neutral gray (128),
pads to square letterbox (224x224), and extracts L2-normalized 384-D embedding.
Adheres strictly to Module 6 in fyp_architecture_final_object.md.
"""

from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import faiss

from module_6.config import Module6Config


class DINOv2FeatureExtractor:
    """
    Self-supervised feature extraction engine using DINOv2 with register tokens.
    Supports both direct image embeddings and full sub-object neighbor suppression.
    """

    def __init__(self, config: Optional[Module6Config] = None):
        self.config = config or Module6Config()
        self.device = torch.device(self.config.device)
        self.model = self._load_model()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.config.image_size, self.config.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    def _load_model(self) -> torch.nn.Module:
        """
        Loads dinov2_vits14_reg model from torch.hub, with HuggingFace fallback if needed.
        """
        try:
            model = torch.hub.load(
                self.config.repo_name,
                self.config.model_name
            )
            model.to(self.device)
            self._mode = "torch_hub"
            return model
        except Exception as e:
            try:
                from transformers import AutoModel
                hf_model = AutoModel.from_pretrained(self.config.hf_fallback_repo)
                hf_model.to(self.device)
                self._mode = "huggingface"
                return hf_model
            except Exception as hf_err:
                raise RuntimeError(
                    f"Failed to load DINOv2 via both torch.hub ({e}) and Hugging Face ({hf_err})"
                )

    def extract(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extracts L2-normalized embedding vector for a single PIL or NumPy RGB image.

        Args:
            image: PIL Image or NumPy RGB array.

        Returns:
            L2-normalized float32 numpy array of shape (1, 384).
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        tensor_in = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self._mode == "torch_hub":
                emb = self.model(tensor_in)
            else:
                outputs = self.model(pixel_values=tensor_in)
                emb = outputs.last_hidden_state[:, 0, :]

            emb_np = emb.cpu().numpy().astype("float32")

        faiss.normalize_L2(emb_np)
        return emb_np

    def extract_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Extracts L2-normalized embeddings for a batch of images.
        """
        if not images:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)
        vectors = [self.extract(img) for img in images]
        return np.vstack(vectors)

    def process_sub_object(
        self,
        sub_raw: np.ndarray,
        sub_mask: np.ndarray,
        abs_bbox: Tuple[int, int, int, int],
        frame_hw: Tuple[int, int],
        method_label: str = "METHOD2_FASTSAM_DINOV2"
    ) -> Dict[str, Any]:
        """
        Suppresses background and touching neighbor pixels to neutral gray (128),
        square letterboxes, and computes L2-normalized 384-D vector.

        Args:
            sub_raw: BGR crop of the sub-object.
            sub_mask: Binary mask of the isolated sub-object.
            abs_bbox: (x1, y1, x2, y2) in full-frame coordinates.
            frame_hw: (frame_h, frame_w).
            method_label: Decomposition method used.

        Returns:
            Dictionary with processed assets and metadata (strict bounding box only).
        """
        bg_gray = np.full_like(sub_raw, 128)
        isolated_bgr = np.where(sub_mask[..., None] > 0, sub_raw, bg_gray)

        h, w = isolated_bgr.shape[:2]
        dim = max(h, w)
        square_bgr = np.full((dim, dim, 3), 128, dtype=np.uint8)
        y_off, x_off = (dim - h) // 2, (dim - w) // 2
        square_bgr[y_off : y_off + h, x_off : x_off + w] = isolated_bgr

        pil_sub = Image.fromarray(cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB))
        emb = self.extract(pil_sub)

        gray = cv2.cvtColor(sub_raw, cv2.COLOR_BGR2GRAY) if sub_raw.ndim == 3 else sub_raw
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size > 0 else 0.0

        return {
            "bbox": abs_bbox,
            "frame_hw": list(frame_hw),
            "raw_crop": sub_raw,
            "masked_crop": square_bgr,
            "embedding": emb,
            "sharpness": sharpness,
            "method": method_label
        }
