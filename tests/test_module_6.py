"""
Tests for Module 6: Self-Supervised Feature Extraction (DINOv2 with Registers)
"""

import pytest
import numpy as np
from PIL import Image

from module_6.config import Module6Config
from module_6.dinov2_extractor import DINOv2FeatureExtractor


class TestModule6:
    """Test suite for DINOv2 feature extractor."""

    @pytest.fixture(scope="class")
    def extractor(self):
        config = Module6Config(image_size=224, embedding_dim=384)
        return DINOv2FeatureExtractor(config)

    def test_embedding_output_shape_and_norm(self, extractor):
        """Feature extractor must output L2-normalized float32 array of shape (1, 384)."""
        # Create synthetic test image
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        emb = extractor.extract(img)

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (1, 384)
        assert emb.dtype == np.float32

        # Verify unit L2 norm
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm 1.0, got {norm}"

    def test_numpy_input_support(self, extractor):
        """Extractor must seamlessly accept both PIL Image and NumPy array inputs."""
        arr = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)
        emb = extractor.extract(arr)

        assert emb.shape == (1, 384)
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

    def test_batch_extraction(self, extractor):
        """Batch extraction must process list of images and return (N, 384) matrix."""
        imgs = [
            Image.new("RGB", (224, 224), color=(i * 40, 100, 150))
            for i in range(3)
        ]
        batch_emb = extractor.extract_batch(imgs)

        assert batch_emb.shape == (3, 384)
        for i in range(3):
            norm = float(np.linalg.norm(batch_emb[i]))
            assert abs(norm - 1.0) < 1e-5
