"""
Trellis2ImageTo3DPipelineMLX: MLX equivalent of Trellis2ImageTo3DPipeline.

Currently supports `pipeline_type='512'` (single-resolution flow). The cascade
modes are present in the class but require additional plumbing in
`upsample()` paths; you can experiment by setting pipeline_type='512' first.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ..extractors.dinov3 import DinoV3FeatureExtractorMLX
from ..io import weight_convert
from ..models import from_config as build_model
from ..ops.sparse_tensor import SparseTensor
from . import samplers as _samplers


# Mesh-with-voxel container produced by the pipeline.
class MeshWithVoxel:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, voxel_size: float,
                 coords: np.ndarray, attrs: np.ndarray, voxel_shape, layout: Dict[str, slice],
                 origin=(-0.5, -0.5, -0.5)):
        self.vertices = vertices
        self.faces = faces
        self.voxel_size = float(voxel_size)
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout
        self.origin_np = np.array(origin, dtype=np.float32)


def _load_model_from_local_or_hub(prefix: str) -> nn.Module:
    cfg, state = weight_convert.load_config_and_state(prefix)
    model = build_model(cfg["name"], **cfg["args"])
    mlx_state = weight_convert.convert_state_dict(state)
    items = list(mlx_state.items())

    # MLX's load_weights raises a single uninformative error when there is any
    # mismatch. Pre-compute the diff against the model's expected keys so a
    # human can see exactly what's wrong (and we degrade gracefully via
    # strict=False so the user can still tinker with partial loads).
    try:
        from mlx.utils import tree_flatten
        expected = {k for k, _ in tree_flatten(model.parameters())}
        provided = {k for k, _ in items}
        if expected != provided:
            _, _, msg = weight_convert.diff_keys(provided, expected)
            print(f"[trellis2_mlx] checkpoint <-> model key mismatch for {cfg.get('name')!r}:\n{msg}",
                  file=sys.stderr)
    except Exception as e:
        # Don't let diagnostics block the load.
        print(f"[trellis2_mlx] (diag failed: {e})", file=sys.stderr)

    try:
        model.load_weights(items, strict=True)
    except TypeError:
        # Older MLX versions don't accept strict=.
        model.load_weights(items)
    return model


def _resolve_and_load(path: str, prefix: str) -> nn.Module:
    """
    `prefix` from pipeline.json can be either:
      (a) a sub-path inside the pipeline repo (e.g. 'ckpts/my_model'), or
      (b) a cross-repo HF id (e.g. 'microsoft/TRELLIS-image-large/ckpts/foo')
          when the pipeline reuses a checkpoint from another repo.

    Try (a) first; if its config is missing on the hub fall through to (b)
    without printing the 404 noise.
    """
    if not os.path.exists(f"{path}/{prefix}.json"):
        # Probe locally only when something resembling the file exists locally.
        local_ok = os.path.exists(f"{path}/{prefix}.safetensors")
    else:
        local_ok = True
    candidate_in_repo = f"{path}/{prefix}"
    if local_ok:
        return _load_model_from_local_or_hub(candidate_in_repo)
    # Two paths to try, in order. Catch RemoteEntryNotFoundError quietly.
    try:
        return _load_model_from_local_or_hub(candidate_in_repo)
    except Exception as e_inner:
        msg = str(e_inner)
        if "404" in msg or "Not Found" in msg or "RemoteEntry" in msg:
            return _load_model_from_local_or_hub(prefix)
        raise


class Trellis2ImageTo3DPipelineMLX:
    """
    Image -> 3D MeshWithVoxel using the MLX inference backend.

    The constructor signature intentionally mirrors the reference pipeline so a
    config from `microsoft/TRELLIS.2-4B/pipeline.json` can be consumed.
    """

    model_names_to_load = [
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_decoder",
        "tex_slat_flow_model_512",
        "tex_slat_decoder",
    ]

    def __init__(
        self,
        models: Optional[Dict[str, nn.Module]] = None,
        sparse_structure_sampler=None,
        shape_slat_sampler=None,
        tex_slat_sampler=None,
        sparse_structure_sampler_params: Optional[dict] = None,
        shape_slat_sampler_params: Optional[dict] = None,
        tex_slat_sampler_params: Optional[dict] = None,
        shape_slat_normalization: Optional[dict] = None,
        tex_slat_normalization: Optional[dict] = None,
        image_cond_model: Optional[DinoV3FeatureExtractorMLX] = None,
        rembg_model=None,
        default_pipeline_type: str = "512",
    ):
        self.models = models or {}
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params or {}
        self.shape_slat_sampler_params = shape_slat_sampler_params or {}
        self.tex_slat_sampler_params = tex_slat_sampler_params or {}
        self.shape_slat_normalization = shape_slat_normalization or {}
        self.tex_slat_normalization = tex_slat_normalization or {}
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }

    # ---- Construction --------------------------------------------------------

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipelineMLX":
        if os.path.exists(f"{path}/{config_file}"):
            cfg_path = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(path, config_file)
        with open(cfg_path) as f:
            args = json.load(f)["args"]

        # Load every checkpoint declared in args['models'] that we support.
        models: Dict[str, nn.Module] = {}
        for k, prefix in args["models"].items():
            if k not in cls.model_names_to_load:
                continue
            print(f"[trellis2_mlx] loading '{k}' from '{prefix}'...")
            models[k] = _resolve_and_load(path, prefix)

        ss_sampler = _samplers.from_config(args["sparse_structure_sampler"]["name"],
                                           **args["sparse_structure_sampler"]["args"])
        shape_sampler = _samplers.from_config(args["shape_slat_sampler"]["name"],
                                              **args["shape_slat_sampler"]["args"])
        tex_sampler = _samplers.from_config(args["tex_slat_sampler"]["name"],
                                            **args["tex_slat_sampler"]["args"])

        image_cond_args = args["image_cond_model"]["args"]
        image_cond = DinoV3FeatureExtractorMLX(**image_cond_args)

        return cls(
            models=models,
            sparse_structure_sampler=ss_sampler,
            shape_slat_sampler=shape_sampler,
            tex_slat_sampler=tex_sampler,
            sparse_structure_sampler_params=args["sparse_structure_sampler"]["params"],
            shape_slat_sampler_params=args["shape_slat_sampler"]["params"],
            tex_slat_sampler_params=args["tex_slat_sampler"]["params"],
            shape_slat_normalization=args["shape_slat_normalization"],
            tex_slat_normalization=args["tex_slat_normalization"],
            image_cond_model=image_cond,
            rembg_model=None,  # MLX path requires a pre-segmented image for now
            default_pipeline_type=args.get("default_pipeline_type", "512"),
        )

    # ---- Inference ----------------------------------------------------------

    def get_cond(self, image, resolution: int, include_neg_cond: bool = True) -> dict:
        self.image_cond_model.image_size = resolution
        cond = self.image_cond_model(image)
        if not include_neg_cond:
            return {"cond": cond}
        return {"cond": cond, "neg_cond": mx.zeros_like(cond)}

    def sample_sparse_structure(self, cond: dict, resolution: int, num_samples: int = 1,
                                sampler_params: Optional[dict] = None) -> mx.array:
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        in_ch = flow_model.in_channels
        noise = mx.random.normal((num_samples, in_ch, reso, reso, reso))
        params = {**self.sparse_structure_sampler_params, **(sampler_params or {})}
        ret = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **params,
            tqdm_desc="Sampling sparse structure",
        )
        z_s = ret.samples
        decoder = self.models["sparse_structure_decoder"]
        decoded = decoder(z_s) > 0  # [B, 1, R, R, R] bool
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            # Channel-first max-pool downsample
            dec_np = np.asarray(decoded).astype(np.uint8)
            B, C, D, H, W = dec_np.shape
            dec_np = dec_np.reshape(B, C, D // ratio, ratio, H // ratio, ratio, W // ratio, ratio).max(axis=(3, 5, 7))
            decoded = mx.array(dec_np > 0)
        # active voxel coords (b, x, y, z)
        decoded_np = np.asarray(decoded).reshape(decoded.shape[0], decoded.shape[2], decoded.shape[3], decoded.shape[4])
        coords_np = np.argwhere(decoded_np).astype(np.int32)  # (n,4) -> (b,x,y,z)
        return mx.array(coords_np)

    def sample_shape_slat(self, cond: dict, flow_model, coords: mx.array,
                          sampler_params: Optional[dict] = None) -> SparseTensor:
        in_ch = flow_model.in_channels
        N = coords.shape[0]
        feats = mx.random.normal((N, in_ch))
        noise = SparseTensor(feats=feats, coords=coords)
        params = {**self.shape_slat_sampler_params, **(sampler_params or {})}
        ret = self.shape_slat_sampler.sample(
            flow_model, noise, **cond, **params, tqdm_desc="Sampling shape SLat",
        )
        slat = ret.samples
        std = mx.array(self.shape_slat_normalization["std"])[None]
        mean = mx.array(self.shape_slat_normalization["mean"])[None]
        return slat.replace(slat.feats * std + mean)

    def decode_shape_slat(self, slat: SparseTensor, resolution: int):
        dec = self.models["shape_slat_decoder"]
        dec.set_resolution(resolution)
        return dec(slat, return_subs=True)

    def sample_tex_slat(self, cond: dict, flow_model, shape_slat: SparseTensor,
                        sampler_params: Optional[dict] = None) -> SparseTensor:
        std = mx.array(self.shape_slat_normalization["std"])[None]
        mean = mx.array(self.shape_slat_normalization["mean"])[None]
        shape_slat = shape_slat.replace((shape_slat.feats - mean) / std)
        in_ch = flow_model.in_channels
        noise_ch = in_ch - shape_slat.feats.shape[1]
        noise = shape_slat.replace(mx.random.normal((shape_slat.coords.shape[0], noise_ch)))
        params = {**self.tex_slat_sampler_params, **(sampler_params or {})}
        ret = self.tex_slat_sampler.sample(
            flow_model, noise, concat_cond=shape_slat, **cond, **params,
            tqdm_desc="Sampling texture SLat",
        )
        slat = ret.samples
        std = mx.array(self.tex_slat_normalization["std"])[None]
        mean = mx.array(self.tex_slat_normalization["mean"])[None]
        return slat.replace(slat.feats * std + mean)

    def decode_tex_slat(self, slat: SparseTensor, subs):
        out = self.models["tex_slat_decoder"](slat, guide_subs=subs)
        return out.replace(out.feats * 0.5 + 0.5)

    def decode_latent(self, shape_slat: SparseTensor, tex_slat: SparseTensor, resolution: int):
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        results = []
        layout = tex_voxels.layout
        coords_all = np.asarray(tex_voxels.coords, dtype=np.int32)
        feats_all = np.asarray(tex_voxels.feats)
        for m, sl in zip(meshes, layout):
            c = coords_all[sl, 1:]
            a = feats_all[sl]
            results.append(MeshWithVoxel(
                vertices=m.vertices,
                faces=m.faces,
                voxel_size=1.0 / resolution,
                coords=c,
                attrs=a,
                voxel_shape=(1, a.shape[1], resolution, resolution, resolution),
                layout=self.pbr_attr_layout,
            ))
        return results

    def run(
        self,
        image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: Optional[dict] = None,
        shape_slat_sampler_params: Optional[dict] = None,
        tex_slat_sampler_params: Optional[dict] = None,
        pipeline_type: Optional[str] = None,
    ) -> List[MeshWithVoxel]:
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type != "512":
            raise NotImplementedError(
                "The MLX path currently supports pipeline_type='512' only. "
                "Cascade and 1024 modes will follow in a subsequent milestone."
            )
        mx.random.seed(seed)
        cond_512 = self.get_cond([image], 512)

        coords = self.sample_sparse_structure(
            cond_512, resolution=32, num_samples=num_samples,
            sampler_params=sparse_structure_sampler_params,
        )
        shape_slat = self.sample_shape_slat(
            cond_512, self.models["shape_slat_flow_model_512"], coords,
            sampler_params=shape_slat_sampler_params,
        )
        tex_slat = self.sample_tex_slat(
            cond_512, self.models["tex_slat_flow_model_512"], shape_slat,
            sampler_params=tex_slat_sampler_params,
        )
        return self.decode_latent(shape_slat, tex_slat, resolution=512)
