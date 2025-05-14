import torch
from ..utils import flatten_tensor
from ...objectives import get_opt_fn_debug

def linear_manipulation(
    ae,
    cls_lin,
    batch,
    median_logit: float,
    cfg: dict,
    cid: int,
    debug: bool = False
):
    """
    Baseline linear attribute manipulation, with optional debug printing.

    Args:
      ae          : autoencoder (LitModel)
      cls_lin     : linear classifier (ClsModel)
      batch       : Tensor, shape (B,C,H,W)
      median_logit: float, the target logit to hit
      cid         : int, the class index for the attribute
      cfg         : dict, config (for reg weights/types)
      debug       : bool, if True print classification & L2 losses
    Returns:
      z_lin       : Tensor (B, latent_dim), the manipulated flattened code
    """
    # --- encode & normalize ---
    cond = ae.encode(batch)            # (B, latent_dims...)
    latent_shape = cond.shape[1:]
    z0   = flatten_tensor(cond)        # (B, D)
    z0n  = cls_lin.normalize(z0)       # (B, D)

    # --- compute shift along classifier normal ---
    w      = cls_lin.classifier.weight[cid]    # (D,)
    b      = cls_lin.classifier.bias[cid]      # ()
    w_norm = (w.pow(2)).sum()                  # ()
    proj   = z0n @ w                            # (B,)
    s      = (median_logit - b - proj) / w_norm # (B,)
    if debug:
        print(f"[Linear] shift s: {s}")

    # --- apply shift and denormalize ---
    z_lin = cls_lin.denormalize(z0n + s.unsqueeze(1) * w.unsqueeze(0))

    debug_outputs = None
    if debug:
        z_lin_norm = cls_lin.normalize(z_lin)
        cls_fn_debug = lambda z: cls_lin.classifier(z)
        debug_fn = get_opt_fn_debug(
            cls_fn_debug,
            cls_id=cid,
            latent_shape=latent_shape,
            x0_flat=z0n,
            classifier_weight=cfg.get("classifier_weight", 1.0),
            reg_norm_weight=cfg.get("reg_norm_weight", 0.5),
            reg_norm_type=cfg.get("reg_norm_type", "L2"),
            target_logit=median_logit,
        )
        with torch.no_grad():
            total_vals, cls_vals, reg_vals = debug_fn(z_lin_norm)
        debug_outputs = {
            "total": total_vals,
            "cls": cls_vals,
            "reg": reg_vals,
        }

    return z_lin, debug_outputs