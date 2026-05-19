"""
DINOv3 ViT image feature extractor for the MLX pipeline.

Implementation strategy: keep this on PyTorch + MPS for now (path A in the
design doc) — the HuggingFace `transformers` DINOv3 model is a stock ViT and
already runs well on Apple Silicon via MPS. We just expose a small adapter
that returns an mx.array.

If `torch` / `transformers` are not installed, calling the adapter raises a
helpful error so the rest of the MLX pipeline can still be imported and tested
in isolation.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import mlx.core as mx


class DinoV3FeatureExtractorMLX:
    """
    Image -> patch token features.

    Returns an mx.array of shape [B, N, D] matching the reference DINOv3 extractor's
    output (post-LayerNorm), which is what SLatFlowModel cross-attention consumes.
    """

    def __init__(self, model_name: str, image_size: int = 512, device: Optional[str] = None):
        self.model_name = model_name
        self.image_size = image_size
        self._device = device
        self._model = None
        self._transform = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            import torch
            from torchvision import transforms
            from transformers import DINOv3ViTModel
        except ImportError as e:
            raise RuntimeError(
                "DinoV3FeatureExtractorMLX requires torch + transformers; the MLX path "
                "currently delegates image conditioning to the PyTorch MPS backend."
            ) from e
        device = self._device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._device = device
        self._model = DINOv3ViTModel.from_pretrained(self.model_name).eval().to(device)
        self._transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._torch = torch

    def to(self, device: str):
        self._device = device
        if self._model is not None:
            self._model.to(device)

    def cuda(self):
        # MLX path does not use CUDA; CPU is a safe fallback.
        self.to("cpu")

    def cpu(self):
        self.to("cpu")

    def __call__(self, image: Union[List, mx.array]) -> mx.array:
        from PIL import Image as PILImage
        self._lazy_init()
        torch = self._torch
        if isinstance(image, mx.array):
            t = torch.from_numpy(np.asarray(image)).float()
        elif isinstance(image, list) and all(isinstance(i, PILImage.Image) for i in image):
            resized = [i.resize((self.image_size, self.image_size), PILImage.LANCZOS) for i in image]
            arrs = [np.array(i.convert("RGB"), dtype=np.float32) / 255 for i in resized]
            t = torch.from_numpy(np.stack([np.transpose(a, (2, 0, 1)) for a in arrs])).float()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        t = t.to(self._device)
        t = self._transform(t)
        # Match reference DinoV3FeatureExtractor.extract_features() in the codebase:
        # use the encoder layer stack with RoPE embeddings, then a final layer-norm.
        m = self._model
        t = t.to(next(m.parameters()).dtype)
        hidden = m.embeddings(t, bool_masked_pos=None)
        pos = m.rope_embeddings(t)
        for layer in m.layer:
            hidden = layer(hidden, position_embeddings=pos)
        # final fp32 layer norm
        hidden = torch.nn.functional.layer_norm(hidden, hidden.shape[-1:])
        return mx.array(hidden.detach().to("cpu").float().numpy())
