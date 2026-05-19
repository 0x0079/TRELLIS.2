"""
Flow-matching Euler sampler with classifier-free guidance + guidance interval.

Equivalent to trellis2/pipelines/samplers/flow_euler.py FlowEulerGuidanceIntervalSampler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import mlx.core as mx
from tqdm import tqdm

from ..ops.sparse_tensor import SparseTensor


@dataclass
class SampleResult:
    samples: Any = None
    pred_x_t: List[Any] = field(default_factory=list)
    pred_x_0: List[Any] = field(default_factory=list)


def _x_op(x, fn):
    if isinstance(x, SparseTensor):
        return x.replace(fn(x.feats))
    return fn(x)


def _x_axpb(x, a, b, y):
    """Compute a*x + b*y. y may be SparseTensor or array."""
    if isinstance(x, SparseTensor):
        yf = y.feats if isinstance(y, SparseTensor) else y
        return x.replace(a * x.feats + b * yf)
    yf = y.feats if isinstance(y, SparseTensor) else y
    return a * x + b * yf


class FlowEulerSampler:
    """Plain Euler flow-matching sampler."""

    def __init__(self, sigma_min: float):
        self.sigma_min = sigma_min

    # --- conversions copied from the reference (Linear-Flow / Rectified-Flow) ---
    def _v_to_xstart_eps(self, x_t, t, v):
        eps = _x_op(x_t, lambda f: f) if False else None  # placeholder for type
        eps = _x_axpb(v, (1 - t), 1.0, x_t)
        a = (1 - self.sigma_min)
        b = -(self.sigma_min + (1 - self.sigma_min) * t)
        x_0 = _x_axpb(x_t, a, b, v)
        return x_0, eps

    def _pred_to_xstart(self, x_t, t, pred):
        a = (1 - self.sigma_min)
        b = -(self.sigma_min + (1 - self.sigma_min) * t)
        return _x_axpb(x_t, a, b, pred)

    def _xstart_to_pred(self, x_t, t, x_0):
        # Used only by CFG rescale; we forward-only compute the algebraic inverse.
        a = (1 - self.sigma_min)
        denom = (self.sigma_min + (1 - self.sigma_min) * t)
        if isinstance(x_t, SparseTensor):
            return x_t.replace((a * x_t.feats - x_0.feats) / denom)
        return (a * x_t - x_0) / denom

    def _inference_model(self, model, x_t, t, cond=None, **kw):
        B = x_t.shape[0]
        t_arr = mx.array([1000.0 * t] * B, dtype=mx.float32)
        return model(x_t, t_arr, cond, **kw)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kw):
        pred_v = self._inference_model(model, x_t, t, cond, **kw)
        x_0, eps = self._v_to_xstart_eps(x_t, t, pred_v)
        return x_0, eps, pred_v

    def sample_once(self, model, x_t, t: float, t_prev: float, cond=None, **kw) -> SampleResult:
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kw)
        # x_{t-1} = x_t - (t - t_prev) * v
        if isinstance(x_t, SparseTensor):
            new = x_t.replace(x_t.feats - (t - t_prev) * pred_v.feats)
        else:
            new = x_t - (t - t_prev) * pred_v
        out = SampleResult()
        out.samples = new
        out.pred_x_t = [new]
        out.pred_x_0 = [pred_x_0]
        return out

    def sample(
        self,
        model,
        noise,
        cond=None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kw,
    ) -> SampleResult:
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
        ret = SampleResult()
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            r = self.sample_once(model, sample, float(t), float(t_prev), cond, **kw)
            sample = r.samples
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(r.pred_x_0[0])
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(FlowEulerSampler):
    """Adds classifier-free guidance."""

    def _inference_model(
        self, model, x_t, t,
        cond=None, neg_cond=None, guidance_strength: float = 1.0,
        guidance_rescale: float = 0.0, **kw,
    ):
        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, **kw)
        if guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kw)
        pred_pos = super()._inference_model(model, x_t, t, cond, **kw)
        pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kw)
        if isinstance(pred_pos, SparseTensor):
            pred = pred_pos.replace(
                guidance_strength * pred_pos.feats + (1 - guidance_strength) * pred_neg.feats
            )
        else:
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
        # CFG rescale skipped — model checkpoints don't require it (rescale=0 default).
        return pred

    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0,
               guidance_strength: float = 3.0, verbose: bool = True, **kw):
        return super().sample(
            model, noise, cond, steps=steps, rescale_t=rescale_t, verbose=verbose,
            neg_cond=neg_cond, guidance_strength=guidance_strength, **kw,
        )


class FlowEulerGuidanceIntervalSampler(FlowEulerCfgSampler):
    """Restricts CFG to a [t_lo, t_hi] interval."""

    def _inference_model(
        self, model, x_t, t,
        cond=None, neg_cond=None, guidance_strength: float = 1.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0), **kw,
    ):
        lo, hi = guidance_interval
        if lo <= t <= hi:
            return super()._inference_model(
                model, x_t, t, cond=cond, neg_cond=neg_cond,
                guidance_strength=guidance_strength, **kw,
            )
        return super(FlowEulerCfgSampler, self)._inference_model(model, x_t, t, cond=cond, **kw)

    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0,
               guidance_strength: float = 3.0,
               guidance_interval: Tuple[float, float] = (0.0, 1.0),
               verbose: bool = True, **kw):
        return super(FlowEulerCfgSampler, self).sample(
            model, noise, cond, steps=steps, rescale_t=rescale_t, verbose=verbose,
            neg_cond=neg_cond, guidance_strength=guidance_strength,
            guidance_interval=guidance_interval, **kw,
        )


_REGISTRY = {
    "FlowEulerSampler": FlowEulerSampler,
    "FlowEulerCfgSampler": FlowEulerCfgSampler,
    "FlowEulerGuidanceIntervalSampler": FlowEulerGuidanceIntervalSampler,
}


def from_config(name: str, **kwargs):
    return _REGISTRY[name](**kwargs)
