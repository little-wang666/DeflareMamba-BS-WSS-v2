# DeflareMamba-BS-WSS v2.5

正在努力学习中

## What Changed In v2.5

This branch follows `DeflareMamba_BS_WSS_Modification_Guide.pdf`.

The BS-WSS algorithm logic is unchanged from v2.1-v2.4. v2.5 focuses on repository correctness and reproducibility:

- Keeps the true baseline config:
  `options/DeflareMamba_flare7kpp_baseline_option.yml`
- Keeps the separate BS-WSS config:
  `options/DeflareMamba_flare7kpp_bs_wss_option.yml`
- Keeps normal multi-line Python/YAML/Markdown files.
- Adds `scripts/verify_v2_5.py` for one-command repository verification.
- Extends text-format checking to include the v2.5 verification script.

Recommended v2.5 verification:

```bash
python scripts/verify_v2_5.py
```

If the current machine does not have the required runtime dependencies for forward tests, run:

```bash
python scripts/verify_v2_5.py --skip-forward
```

Then run the forward tests in the actual training environment:

```bash
python scripts/test_bs_wss_forward.py --case baseline
python scripts/test_bs_wss_forward.py --case bs_wss
```

## What Changed In v2.4

This branch keeps the BS-WSS algorithm logic unchanged and fixes the text-format issue more strictly.

Changes:

- Expanded `.gitattributes` to cover normal source/config/document files.
- Added `.editorconfig` for UTF-8 and LF editor defaults.
- Strengthened `scripts/check_text_format.py` so key files must be LF-only and normal multi-line text.
- Re-saved key model/config/test files with LF line endings.

Run the strict format check:

```bash
python scripts/check_text_format.py
```

## What Changed In v2.3

This branch keeps the BS-WSS algorithm logic unchanged and fixes repository text-format hygiene.

Changes:

- Added `.gitattributes` to force LF line endings for Python, YAML, Markdown, shell, and text files.
- Added `scripts/check_text_format.py` to verify that key files are normal multi-line text files.
- Kept the v2.2 baseline/BS-WSS config split and forward test script.

Run the format check:

```bash
python scripts/check_text_format.py
```

## What Changed In v2.2

This branch keeps the BS-WSS model logic from v2.1 and focuses on reproducibility and configuration cleanup.

Changes:

- Restored a true baseline config:
  `options/DeflareMamba_flare7kpp_baseline_option.yml`
- Added a separate BS-WSS config:
  `options/DeflareMamba_flare7kpp_bs_wss_option.yml`
- Added a minimal forward shape test:
  `scripts/test_bs_wss_forward.py`
- Kept the v2.1 HH dual-gate detail branch:

```text
HH' = HH - suppress_gate * artifact_feature + restore_gate * detail_feature
```

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

## Forward Test

Run both baseline and BS-WSS shape checks:

```bash
python scripts/test_bs_wss_forward.py
```

Run only BS-WSS:

```bash
python scripts/test_bs_wss_forward.py --case bs_wss
```

Run only baseline:

```bash
python scripts/test_bs_wss_forward.py --case baseline
```

Expected output shape:

```text
(1, 6, 512, 512)
```

## Suggested Order

1. Run `python -m py_compile basicsr/archs/DeflareMamba_arch.py`.
2. Run `python -m py_compile basicsr/archs/wavelet_vssm_modules.py`.
3. Run `python scripts/test_bs_wss_forward.py`.
4. Start training with the BS-WSS config after the shape test passes.

## Notes

The original DeflareMamba input/output format, U-shaped backbone, training pipeline, and 6-channel output head are kept. The main algorithmic change is inside the `SS2D` feature path, where the feature map is split by Haar DWT into LL/LH/HL/HH bands and restored through band-specific branches before Haar IWT reconstruction.
