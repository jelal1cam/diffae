import torch
import torch.nn.functional as F
from typing import Optional, Union

TensorOrScalar = Union[torch.Tensor, float]

def get_opt_fn(
    classifier_fn,
    cls_id: int,
    latent_shape,
    x0_flat: torch.Tensor,
    classifier_weight: float = 1.0,
    reg_norm_weight: float = 0.1,
    reg_norm_type: str = "L2",
    target_logit: Optional[TensorOrScalar] = None,
):
    """
    Returns opt_fn(x_flat, idx=None) -> Tensor of shape (B,).

    Args:
      classifier_fn      – fn: Tensor(B,D) -> Tensor(B,num_classes)
      cls_id             – which logit to use
      latent_shape       – unused here
      x0_flat            – Tensor(B,D), original codes
      classifier_weight  – weight on the classification term
      reg_norm_weight    – weight on the ||x-x0|| term
      reg_norm_type      – "L2" or "L1"
      target_logit       – None (maximize raw logit), float, or Tensor.
                           If Tensor.ndim == 0 we treat it as scalar.
                           If Tensor.ndim == 1 must be length B.

    Behavior:
      If target_logit is None:
        cls_loss = - logit_cls_id(x)
      Else:
        cls_loss = (logit_cls_id(x) - tlt)^2
      reg_loss = ∥x_flat - x0∥_{L1 or L2} per sample
      return classifier_weight * cls_loss + reg_norm_weight * reg_loss
    """
    def opt_fn(x_flat: torch.Tensor, idx=None) -> torch.Tensor:
        # pick the matching x0's
        x0 = x0_flat[idx] if idx is not None else x0_flat

        logits = classifier_fn(x_flat)          # (B, num_classes)
        pred   = logits[:, cls_id]              # (B,)

        # build tlt = (B,) tensor on the same device/dtype
        if target_logit is None:
            cls_loss = -pred
        else:
            if torch.is_tensor(target_logit):
                tlt = target_logit.to(pred.device).type_as(pred)
                # if it's a scalar tensor, expand to match batch
                if tlt.ndim == 0:
                    tlt = tlt.expand_as(pred)
                elif tlt.ndim == 1 and tlt.shape[0] == pred.shape[0]:
                    pass
                else:
                    raise ValueError(f"target_logit tensor must be scalar or shape {(pred.shape[0],)}, got {tuple(tlt.shape)}")
            else:
                # Python float → full‐batch tensor
                tlt = pred.new_full(pred.shape, float(target_logit))

            cls_loss = (pred - tlt).pow(2)

        # regularization
        if reg_norm_weight > 0:
            if reg_norm_type.upper() == "L2":
                reg_loss = F.mse_loss(x_flat, x0, reduction="none").sum(dim=1)
            elif reg_norm_type.upper() == "L1":
                reg_loss = F.l1_loss(x_flat, x0, reduction="none").sum(dim=1)
            else:
                raise ValueError(f"Unknown reg_norm_type {reg_norm_type!r}")
        else:
            reg_loss = x_flat.new_zeros(x_flat.size(0))

        return classifier_weight * cls_loss + reg_norm_weight * reg_loss

    return opt_fn


def get_opt_fn_debug(
    classifier_fn,
    cls_id: int,
    latent_shape,
    x0_flat: torch.Tensor,
    classifier_weight: float = 1.0,
    reg_norm_weight: float = 0.1,
    reg_norm_type: str = "L2",
    target_logit: Optional[TensorOrScalar] = None,
):
    """
    Returns debug_fn(x_flat) -> (total_loss, cls_loss, reg_loss),
    exposing each component under the same logic as get_opt_fn.
    """
    def debug_fn(x_flat: torch.Tensor):
        logits = classifier_fn(x_flat)
        pred   = logits[:, cls_id]

        # classification term
        if target_logit is None:
            cls_loss = -pred
        else:
            if torch.is_tensor(target_logit):
                tlt = target_logit.to(pred.device).type_as(pred)
                if tlt.ndim == 0:
                    tlt = tlt.expand_as(pred)
                elif tlt.ndim == 1 and tlt.shape[0] == pred.shape[0]:
                    pass
                else:
                    raise ValueError(f"target_logit tensor must be scalar or shape {(pred.shape[0],)}, got {tuple(tlt.shape)}")
            else:
                tlt = pred.new_full(pred.shape, float(target_logit))

            cls_loss = (pred - tlt).pow(2)

        # regularization term
        if reg_norm_weight > 0:
            if reg_norm_type.upper() == "L2":
                reg_loss = F.mse_loss(x_flat, x0_flat, reduction="none").sum(dim=1)
            elif reg_norm_type.upper() == "L1":
                reg_loss = F.l1_loss(x_flat, x0_flat, reduction="none").sum(dim=1)
            else:
                raise ValueError(f"Unknown reg_norm_type {reg_norm_type!r}")
        else:
            reg_loss = x_flat.new_zeros(x_flat.size(0))

        total_loss = classifier_weight * cls_loss + reg_norm_weight * reg_loss
        return total_loss, cls_loss, reg_loss

    return debug_fn
