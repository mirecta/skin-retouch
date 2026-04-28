<div align="center">

# FaceRetouchNet

**Deep learning face retouching via spectral restoration**

Based on *"Face Retouching with Diffusion Data Generation and Spectral Restoration"* (ICCV 2025)

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Export-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai)
[![License](https://img.shields.io/badge/License-Research-yellow?style=for-the-badge)]()

<sub>10.1M params &bull; 256/512 progressive training &bull; tiled inference for 20+ MP portraits &bull; ONNX for C++/Rust plugins</sub>

</div>

---

## Architecture

```mermaid
%%{init: {'theme': 'dark'}}%%
graph TD
    INPUT(["Input Image - B x 3 x H x W"]):::input

    subgraph PYRAMID [" Laplacian Pyramid "]
        L0["Level 0 - Full res"]:::pyramid
        L1["Level 1 - H/2"]:::pyramid
        L2["Level 2 - H/4"]:::pyramid
    end

    subgraph SMGB_BOX [" SMGB - Soft Mask Generation "]
        ENC["Encoder - 4x Conv+Pool"]:::smgb
        BOTTLE["Bottleneck"]:::smgb
        DEC["Decoder - 2x Up+Conv"]:::smgb
        MASK(["Mask - B x 3 x H/4 x W/4"]):::mask
        ENC --> BOTTLE --> DEC --> MASK
    end

    subgraph FSR_BOX [" FSR - Frequency Selection and Restoration "]
        FSR0["FSR @ L0 : Conv > FADA > SDP > SFFN > Conv"]:::fsr
        FSR1["FSR @ L1"]:::fsr
        FSR2["FSR @ L2"]:::fsr
    end

    subgraph MRF_BOX [" MRF - Multi-Resolution Fusion "]
        CLC1["CLC - Fuse L2 + L1"]:::mrf
        CLC0["CLC - Fuse L1 + L0"]:::mrf
        OUT(["Output - B x 3 x H x W"]):::output
        FSR2 --> CLC1
        FSR1 --> CLC1
        CLC1 --> CLC0
        FSR0 --> CLC0
        CLC0 --> OUT
    end

    INPUT --> PYRAMID
    INPUT --> SMGB_BOX
    L0 --> FSR0
    L1 --> FSR1
    L2 --> FSR2
    MASK -.->|guide| FSR0
    MASK -.->|guide| FSR1
    MASK -.->|guide| FSR2

    classDef input fill:#7c3aed,stroke:#a78bfa,color:#fff,stroke-width:2px
    classDef output fill:#059669,stroke:#34d399,color:#fff,stroke-width:2px
    classDef pyramid fill:#1e3a5f,stroke:#60a5fa,color:#e0e0e0
    classDef smgb fill:#4a1942,stroke:#c084fc,color:#e0e0e0
    classDef mask fill:#7c2d12,stroke:#fb923c,color:#fff
    classDef fsr fill:#1a1a2e,stroke:#7c3aed,color:#e0e0e0
    classDef mrf fill:#064e3b,stroke:#34d399,color:#e0e0e0
```

<table>
<tr><td><b>1.</b></td><td><b>Pyramid</b></td><td>Decompose input into 3 Laplacian frequency scales</td></tr>
<tr><td><b>2.</b></td><td><b>SMGB</b></td><td>U-Net predicts soft blemish mask at &frac14; resolution</td></tr>
<tr><td><b>3.</b></td><td><b>FSR</b></td><td>Frequency-domain restoration at each scale &mdash; FADA + SDP attention + SFFN</td></tr>
<tr><td><b>4.</b></td><td><b>MRF</b></td><td>Bottom-up fusion of restored scales back to full resolution</td></tr>
</table>

---

## Setup

```bash
cd /root/retouch/skin_retouch_dl
uv venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install pyyaml tqdm tensorboard einops pillow scikit-image
uv pip install onnx onnxruntime  # for export
uv pip install lpips             # optional, for evaluation
```

---

## Dataset

Expects **FFHQR** &mdash; 70k paired 1024&times;1024 face images:

| Split | ID Range | Count |
|:------|:---------|------:|
| Train | `00000` &ndash; `55999` | 56,000 |
| Val   | `56000` &ndash; `62999` | 7,000 |
| Test  | `63000` &ndash; `69999` | 7,000 |

```
dataset/
  images1024x1024/    # originals (flat dir: 00000.png .. 69999.png)
  ffhqr/              # retouched  (subdirs of 10k each)
```

---

## Training

```bash
# Train from scratch
python train.py --config configs/default.yaml

# Resume from best checkpoint
python train.py --config configs/default.yaml --resume checkpoints/best.pth

# Resume from specific epoch
python train.py --config configs/default.yaml --resume checkpoints/epoch_030.pth
```

> **Progressive training** is automatic: `256x256` patches (batch 32) &rarr; `512x512` (batch 8) at epoch 30.
> Configure in `configs/default.yaml` under `progressive:`.

```bash
# Monitor with TensorBoard
tensorboard --logdir ./logs --bind_all
```

---

## Evaluation

```bash
# Test split metrics: PSNR, SSIM, LPIPS
python evaluate.py --checkpoint checkpoints/best.pth

# Custom data directories
python evaluate.py --checkpoint checkpoints/best.pth \
    --original-dir /path/to/originals \
    --retouched-dir /path/to/retouched
```

| Target | PSNR | SSIM | LPIPS |
|:-------|-----:|-----:|------:|
| Paper  | >47 dB | >0.990 | <0.013 |

---

## Inference

```bash
# Single image
python inference.py photo.jpg --checkpoint checkpoints/best.pth -o ./output

# Large image (20+ MP) with tiled processing
python inference.py photo.jpg --checkpoint checkpoints/best.pth --tile 1024

# Partial retouching (50% strength blend)
python inference.py photo.jpg --checkpoint checkpoints/best.pth -s 0.5

# Export blemish mask only
python inference.py photo.jpg --checkpoint checkpoints/best.pth --mask-only
```

---

## ONNX Export

> For deployment in **C++/Rust** plugins (GIMP, Darktable) &mdash; no Python dependency needed.

```bash
# Fixed 512x512 tile export
python export_onnx.py --checkpoint checkpoints/best.pth \
    --output face_retouch.onnx

# Dynamic input size (any H/W multiple of 16)
python export_onnx.py --checkpoint checkpoints/best.pth \
    --dynamic --output face_retouch.onnx

# Image-only output (no mask — simpler for plugins)
python export_onnx.py --checkpoint checkpoints/best.pth \
    --image-only --dynamic --output face_retouch.onnx

# With onnx-simplifier
python export_onnx.py --checkpoint checkpoints/best.pth \
    --dynamic --simplify --output face_retouch.onnx
```

---

## Config

Edit `configs/default.yaml`:

| Section | Key | Default | Description |
|:--------|:----|--------:|:------------|
| `model` | `levels` | `2` | Laplacian pyramid levels |
| `model` | `channels` | `64` | FSR feature channels |
| `model` | `fsr_blocks` | `2` | FSR blocks per scale |
| `model` | `window_size` | `8` | SDP attention window size |
| `model` | `patch_size` | `8` | SFFN frequency patch size |
| `train` | `lr` | `5e-4` | Learning rate |
| `train` | `epochs` | `100` | Total training epochs |
| `loss` | `lambda_mask` | `0.1` | Mask BCE loss weight |
| `loss` | `lambda_perceptual` | `0.01` | VGG perceptual loss weight |

---

<div align="center">
<sub><b>~10.1M parameters</b> &bull; Retouched image + blemish mask at &frac14; resolution</sub>
</div>
