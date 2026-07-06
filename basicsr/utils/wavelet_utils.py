import torch
import torch.nn.functional as F


def dwt2(x):
    """
    Differentiable Haar DWT.

    Args:
        x: [B, C, H, W]

    Returns:
        ll, lh, hl, hh: each [B, C, ceil(H/2), ceil(W/2)]
    """
    B, C, H, W = x.shape

    pad_h = H % 2
    pad_w = W % 2

    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5

    return ll, lh, hl, hh


def idwt2(ll, lh, hl, hh, out_size=None):
    """
    Differentiable inverse Haar DWT.

    Args:
        ll/lh/hl/hh: [B, C, H, W]
        out_size: optional (H_out, W_out), used to crop padding.

    Returns:
        x: [B, C, 2H, 2W] or cropped to out_size
    """
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5

    B, C, H, W = ll.shape

    x = torch.zeros(
        B, C, H * 2, W * 2,
        device=ll.device,
        dtype=ll.dtype
    )

    x[:, :, 0::2, 0::2] = x00
    x[:, :, 0::2, 1::2] = x01
    x[:, :, 1::2, 0::2] = x10
    x[:, :, 1::2, 1::2] = x11

    if out_size is not None:
        H_out, W_out = out_size
        x = x[:, :, :H_out, :W_out]

    return x