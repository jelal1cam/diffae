import torch
import torch.nn.functional as F

def get_opt_fn(classifier_fn, cls_id, latent_shape, x0_flat,
               classifier_weight=1., reg_norm_weight=0.1, reg_norm_type="L2"):
    """
    Returns opt_fn(x_flat, idx=None) -> tensor (B,)
    Combines classifier loss with per-sample L1/L2 distance to each sample's original x0_flat.
    """

    def opt_fn(x_flat, idx=None):
        # Select matching reference points for this sub-batch
        if idx is not None:
            x0 = x0_flat[idx]
        else:
            x0 = x0_flat

        logits = classifier_fn(x_flat)
        cls_loss = -logits[:, cls_id]

        # Compute per-sample regularization
        if reg_norm_weight > 0:
            if reg_norm_type.upper() == "L2":
                reg_loss = F.mse_loss(x_flat, x0, reduction='none').sum(dim=1)
            elif reg_norm_type.upper() == "L1":
                reg_loss = F.l1_loss(x_flat, x0, reduction='none').sum(dim=1)
            else:
                raise ValueError(f"Unknown norm type: {reg_norm_type}. Use 'L2' or 'L1'.")
        else:
            reg_loss = x_flat.new_zeros(x_flat.size(0))

        return classifier_weight * cls_loss + reg_norm_weight * reg_loss

    return opt_fn
