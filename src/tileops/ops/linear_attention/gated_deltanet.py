import math
from typing import Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op

__all__ = ["GatedDeltaNetFwdOp"]


class GatedDeltaNetFwdOp(Op):
    """Dense inference forward for the gated delta rule.

    ``q`` and ``k`` use ``[B, T, H, K]``. ``v``, ``g``, ``beta`` and the
    output use ``HV`` recurrent heads, where ``HV`` is a multiple of ``H``.
    The recurrent state is always FP32 and value-major: ``[B, HV, V, K]``.

    One interface covers prefill and decode. The target callable selects the
    execution path from the current sequence length; in particular, ``T == 1``
    is decode rather than a separate public Op.
    """

    def __init__(
        self,
        scale: Optional[float] = None,
        use_qk_l2norm: bool = False,
        use_gate_activation: bool = False,
        use_beta_sigmoid: bool = False,
        allow_neg_eigval: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        target: Target = None,
    ) -> None:
        """Fix recurrence semantics; tensor metadata comes from each call.

        Args:
            scale: Query scale, or ``None`` for ``K**-0.5``.
            use_qk_l2norm: Normalize Q and K inside the implementation.
            use_gate_activation: Convert raw gates using ``A_log`` and
                ``dt_bias`` inside the implementation.
            use_beta_sigmoid: Apply sigmoid to raw ``beta`` inside the
                implementation.
            allow_neg_eigval: Use the signed beta transform. Valid only when
                ``use_beta_sigmoid`` is enabled.
            kernel_map: Optional in-tree kernel overrides.
            target: Backend target, or ``None`` to resolve from the input
                device.
        """
        if scale is not None and not math.isfinite(scale):
            raise ValueError(f"scale must be finite, got {scale}")
        if allow_neg_eigval and not use_beta_sigmoid:
            raise ValueError("allow_neg_eigval requires use_beta_sigmoid=True")

        self.scale = scale
        self.use_qk_l2norm = use_qk_l2norm
        self.use_gate_activation = use_gate_activation
        self.use_beta_sigmoid = use_beta_sigmoid
        self.allow_neg_eigval = allow_neg_eigval
        self.target = target
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
        initial_state_shape: Optional[tuple[int, ...]] = None,
        A_log_shape: Optional[tuple[int, ...]] = None,
        dt_bias_shape: Optional[tuple[int, ...]] = None,
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, g_shape, beta_shape, initial_state_shape, A_log_shape, dt_bias_shape
        batch, seq_len, _heads, dim_k = q_shape
        _batch, _seq_len, value_heads, dim_v = v_shape
        return {
            "o": (batch, seq_len, value_heads, dim_v),
            "final_state": (batch, value_heads, dim_v, dim_k),
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> None:
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("q must have float16 or bfloat16 dtype")
        for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta)):
            if tensor.dtype != q.dtype:
                raise ValueError(f"{name} must have the same dtype as q")
        for name, tensor in (
            ("initial_state", initial_state),
            ("A_log", A_log),
            ("dt_bias", dt_bias),
        ):
            if tensor is not None and tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have float32 dtype")

    def eval_roofline(self) -> tuple[int, int]:
        raise NotImplementedError("GatedDeltaNetFwdOp has no in-tree implementation yet")

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        A_log: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
    ) -> None:
        if q.ndim != 4 or k.shape != q.shape:
            raise ValueError("q and k must have the same [B, T, H, K] shape")
        if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
            raise ValueError("v must have shape [B, T, HV, V]")

        batch, seq_len, heads, dim_k = q.shape
        value_heads, dim_v = v.shape[2:]
        if heads == 0 or value_heads % heads != 0:
            raise ValueError("HV must be divisible by H")
        if g.shape != (batch, seq_len, value_heads):
            raise ValueError("g must have shape [B, T, HV]")
        if beta.shape != g.shape:
            raise ValueError("beta must have the same shape as g")

        if initial_state is not None and initial_state.shape != (
            batch,
            value_heads,
            dim_v,
            dim_k,
        ):
            raise ValueError("initial_state must have shape [B, HV, V, K]")

        if self.use_gate_activation:
            if A_log is None or dt_bias is None:
                raise ValueError("use_gate_activation=True requires A_log and dt_bias")
            if A_log.shape != (value_heads,) or dt_bias.shape != (value_heads,):
                raise ValueError("A_log and dt_bias must have shape [HV]")
        elif A_log is not None or dt_bias is not None:
            raise ValueError("A_log and dt_bias require use_gate_activation=True")

        self._validate_dtypes(q, k, v, g, beta, initial_state, A_log, dt_bias)
        for name, tensor in (
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("initial_state", initial_state),
            ("A_log", A_log),
            ("dt_bias", dt_bias),
        ):
            if tensor is not None and tensor.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")

    @staticmethod
    def _canonicalize_inputs(
        *inputs: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        return tuple(tensor.contiguous() if tensor is not None else None for tensor in inputs)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run dense prefill or decode and return ``(o, final_state)``."""
        self._validate_forward_inputs(q, k, v, g, beta, initial_state, A_log, dt_bias)
        inputs = self._canonicalize_inputs(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            A_log,
            dt_bias,
        )
        kernel = self.get_or_build_kernel("gated_deltanet", inputs)
        return kernel(*inputs)
