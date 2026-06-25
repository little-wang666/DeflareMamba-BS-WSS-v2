import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_even(x):
    h, w = x.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect'), (h, w)


def crop_to_size(x, size):
    h, w = size
    return x[..., :h, :w]


class HaarDWT(nn.Module):
    def forward(self, x):
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 + x01 - x10 - x11) * 0.5
        hl = (x00 - x01 + x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


class HaarIWT(nn.Module):
    def forward(self, ll, lh, hl, hh):
        b, c, h, w = ll.shape
        out = ll.new_empty(b, c, h * 2, w * 2)
        out[..., 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        out[..., 0::2, 1::2] = (ll + lh - hl - hh) * 0.5
        out[..., 1::2, 0::2] = (ll - lh + hl - hh) * 0.5
        out[..., 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
        return out


class DirectionalEdgeBranch(nn.Module):
    def __init__(self, dim, kernel=7):
        super().__init__()
        padding = kernel // 2
        self.lh_conv = nn.Sequential(
            nn.Conv2d(dim, dim, (1, kernel), padding=(0, padding), groups=dim),
            nn.Conv2d(dim, dim, (kernel, 1), padding=(padding, 0), groups=dim),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.hl_conv = nn.Sequential(
            nn.Conv2d(dim, dim, (kernel, 1), padding=(padding, 0), groups=dim),
            nn.Conv2d(dim, dim, (1, kernel), padding=(0, padding), groups=dim),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.edge_gate = nn.Conv2d(dim * 3, dim * 2, 1)

    def forward(self, lh, hl, ll_context):
        gate = torch.sigmoid(self.edge_gate(torch.cat([ll_context, lh, hl], dim=1)))
        gate_lh, gate_hl = gate.chunk(2, dim=1)
        lh_out = lh + gate_lh * self.lh_conv(lh)
        hl_out = hl + gate_hl * self.hl_conv(hl)
        return lh_out, hl_out


class ConfidenceDetailBranch(nn.Module):
    def __init__(self, dim, use_confidence=True):
        super().__init__()
        self.use_confidence = use_confidence
        self.expand = nn.Conv2d(dim, dim * 2, 1)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
        self.project = nn.Conv2d(dim, dim, 1)
        self.confidence = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, hh, ll_context, lh_context, hl_context):
        detail = self.dwconv(self.expand(hh))
        a, b = detail.chunk(2, dim=1)
        detail = self.project(a * b)
        if self.use_confidence:
            confidence = torch.sigmoid(self.confidence(torch.cat([ll_context, lh_context, hl_context, hh], dim=1)))
            detail = confidence * detail
        return hh + detail


class CrossBandInteraction(nn.Module):
    def __init__(self, dim, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.message = nn.Conv2d(dim * 4, dim * 4, 1)
        self.gate = nn.Conv2d(dim * 4, dim * 4, 1)

    def forward(self, ll, lh, hl, hh):
        if not self.enabled:
            return ll, lh, hl, hh
        cat = torch.cat([ll, lh, hl, hh], dim=1)
        fused = cat + torch.sigmoid(self.gate(cat)) * self.message(cat)
        return fused.chunk(4, dim=1)
