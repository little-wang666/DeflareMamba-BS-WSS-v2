# DeflareMamba-BS-WSS v2.1

正在努力学习中

## v2.1 修改说明

本仓库基于 DeflareMamba 添加 Band-specific Wavelet Selective Scan (BS-WSS)。

核心路径为：

```text
feature -> Haar DWT -> LL/LH/HL/HH -> band-specific branches -> cross-band interaction -> Haar IWT
```

主要改动：

- 默认训练配置已启用 `use_wavelet_vssm: true`。
- 默认频带模块位置为 `wavelet_vssm_stages: [encoder_l, decoder_h]`。
- LL 分支继续使用原 DeflareMamba 的 selective scan。
- LH/HL 分支使用轻量方向卷积分支，不是完整 Strip Attention。
- HH 分支升级为双门控：

```text
HH' = HH - suppress_gate * artifact_feature + restore_gate * detail_feature
```

## 推荐训练配置

默认配置文件位于：

```text
options/DeflareMamba_flare7kpp_baseline_option.yml
```

当前推荐配置：

```yaml
use_wavelet_vssm: true
wavelet_vssm_stages: [encoder_l, decoder_h]
wavelet_vssm_last_only: true
wavelet_edge_kernel: 7
wavelet_use_hh_confidence: true
wavelet_use_cross_band_interaction: true
```

如果只想做第一版轻量消融，可以改为：

```yaml
wavelet_vssm_stages: [decoder_h]
```

如果想做更激进实验，可以改为：

```yaml
wavelet_vssm_stages: [all]
```

## 消融实验建议

建议依次比较：

```text
Baseline DeflareMamba
BS-WSS decoder_h only
BS-WSS encoder_l + decoder_h
BS-WSS all stages
BS-WSS without HH dual gate
BS-WSS without cross-band interaction
```

## 注意

本版本保持 DeflareMamba 原始输入输出、U-shaped 主干和训练流程基本不变，主要修改位于 `SS2D` 内部的小波频带路径。
