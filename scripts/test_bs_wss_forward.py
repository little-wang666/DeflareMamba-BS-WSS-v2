import argparse
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from basicsr.archs.DeflareMamba_arch import DeflareMamba


def parse_depths(value):
    return tuple(int(item.strip()) for item in value.split(',') if item.strip())


def build_model(use_wavelet_vssm, img_size, embed_dim, d_state, depths, stages):
    return DeflareMamba(
        upscale=1,
        in_chans=3,
        img_size=img_size,
        img_range=1.,
        d_state=d_state,
        depths=depths,
        embed_dim=embed_dim,
        mlp_ratio=1.,
        use_wavelet_vssm=use_wavelet_vssm,
        wavelet_vssm_stages=stages,
        wavelet_vssm_last_only=True,
        wavelet_edge_kernel=7,
        wavelet_use_hh_confidence=True,
        wavelet_use_cross_band_interaction=True,
    )


def run_case(name, use_wavelet_vssm, args, device):
    model = build_model(
        use_wavelet_vssm=use_wavelet_vssm,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        d_state=args.d_state,
        depths=parse_depths(args.depths),
        stages=args.wavelet_stages,
    ).to(device)
    model.eval()
    x = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        y = model(x)
    expected = (1, 6, args.img_size, args.img_size)
    print(f'{name}: input={tuple(x.shape)} output={tuple(y.shape)}')
    if tuple(y.shape) != expected:
        raise AssertionError(f'{name} output shape {tuple(y.shape)} != {expected}')


def main():
    parser = argparse.ArgumentParser(description='Forward shape test for DeflareMamba BS-WSS.')
    parser.add_argument('--case', choices=['baseline', 'bs_wss', 'both'], default='both')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--embed-dim', type=int, default=40)
    parser.add_argument('--d-state', type=int, default=10)
    parser.add_argument('--depths', default='1,2,4,4,4,2,1')
    parser.add_argument('--wavelet-stages', nargs='+', default=['encoder_l', 'decoder_h'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.case in ('baseline', 'both'):
        run_case('baseline', False, args, device)
    if args.case in ('bs_wss', 'both'):
        run_case('bs_wss', True, args, device)


if __name__ == '__main__':
    main()
