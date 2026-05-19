"""
MLX high-level model classes. Names mirror trellis2/models/ so config files map 1:1.
"""
from .sparse_structure_flow import SparseStructureFlowModel, TimestepEmbedder
from .sparse_structure_vae import SparseStructureDecoder, SparseStructureEncoder
from .structured_latent_flow import SLatFlowModel
from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
from .sc_vaes.fdg_vae import FlexiDualGridVaeDecoder

__all__ = [
    "SparseStructureFlowModel",
    "SparseStructureDecoder",
    "SparseStructureEncoder",
    "SLatFlowModel",
    "SparseUnetVaeEncoder",
    "SparseUnetVaeDecoder",
    "FlexiDualGridVaeDecoder",
    "TimestepEmbedder",
]


_REGISTRY = {
    "SparseStructureFlowModel": SparseStructureFlowModel,
    "SparseStructureDecoder": SparseStructureDecoder,
    "SparseStructureEncoder": SparseStructureEncoder,
    "SLatFlowModel": SLatFlowModel,
    "ElasticSLatFlowModel": SLatFlowModel,  # elastic is a training-time mixin; same arch for inference
    "SparseUnetVaeEncoder": SparseUnetVaeEncoder,
    "SparseUnetVaeDecoder": SparseUnetVaeDecoder,
    "FlexiDualGridVaeDecoder": FlexiDualGridVaeDecoder,
}


def from_config(name: str, **kwargs):
    cls = _REGISTRY[name]
    return cls(**kwargs)
