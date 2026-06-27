# DeflareMamba-BS-WSS v2.6

Learning in progress.

## What Changed In v2.6

v2.6 rebuilds the repository health checks around the problem found in v2.2-v2.5: key Python, YAML, and Markdown files must stay as real multi-line text files after commit and push.

Algorithm logic is unchanged from the BS-WSS implementation:

- Keep the original DeflareMamba U-shaped backbone and training flow.
- Add Haar DWT/IWT inside the optional wavelet VSSM path.
- Keep LL on the original VSSM scan path.
- Process LH/HL with the lightweight directional edge branch.
- Process HH with the confidence-guided detail branch.
- Reconstruct the normal feature map through inverse Haar wavelet transform.

## Required Verification

Run the full v2.6 repository check:

```bash
python scripts/verify_v2_6.py
```

If the current machine does not have the full PyTorch, BasicSR, and Mamba runtime, first run the static checks:

```bash
python scripts/verify_v2_6.py --skip-forward
```

Then run the forward checks inside the actual training environment:

```bash
python scripts/test_bs_wss_forward.py --case baseline
python scripts/test_bs_wss_forward.py --case bs_wss
```

Expected output for both cases:

```text
(1, 6, 512, 512)
```

## Line Count Guard

Before publishing v2.6, these files must be normal multi-line files:

```text
basicsr/archs/wavelet_vssm_modules.py      > 100 lines
basicsr/archs/DeflareMamba_arch.py         > 500 lines
scripts/test_bs_wss_forward.py             > 50 lines
scripts/check_text_format.py               > 40 lines
scripts/verify_v2_6.py                     > 50 lines
options/DeflareMamba_flare7kpp_bs_wss_option.yml > 100 lines
```

The verification script enforces these limits and also runs `py_compile` on the Python files.

## Config Files

Baseline DeflareMamba:

```text
options/DeflareMamba_flare7kpp_baseline_option.yml
```

Important setting:

```yaml
use_wavelet_vssm: false
```

BS-WSS:

```text
options/DeflareMamba_flare7kpp_bs_wss_option.yml
```

Recommended setting:

```yaml
use_wavelet_vssm: true
wavelet_vssm_stages: [encoder_l, decoder_h]
wavelet_vssm_last_only: true
wavelet_edge_kernel: 7
wavelet_use_hh_confidence: true
wavelet_use_cross_band_interaction: true
```

For a lighter first ablation:

```yaml
wavelet_vssm_stages: [decoder_h]
```

For a more aggressive experiment:

```yaml
wavelet_vssm_stages: [all]
```

## Notes

The baseline config remains a true baseline with `use_wavelet_vssm: false`. The BS-WSS config is separate so later training logs and reports do not confuse baseline results with the wavelet experiment.
