# Image Quality Metrics Testing

This folder contains scripts to evaluate underwater image enhancement quality using both Full-Reference (FR) and No-Reference (NR) metrics.

## Structure
- `fr/`: Full-Reference metrics (requires Ground Truth).
- `nr/`: No-Reference metrics (does not require Ground Truth, except for PS/PSNR if provided).
- `measure_ps.py`: Perceptual Score (PS) script (Unified FR/NR).

## Installation

1. **Prerequisites**: Ensure you have Python installed (Python 3.9+ recommended).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `numpy` should be <2.0 to ensure compatibility with `skimage` and other libraries.*

## Usage

### 1. Full-Reference Metrics
Calculates **MSE**, **SSIM**, **MAE**, and **PSNR**.
**Input**: Enhanced images folder and Ground Truth (GT) folder.

```bash
python fr/measure_metrics.py --enhanced_dir /path/to/enhanced_images --gt_dir /path/to/gt_images
```
*   Images are resized to 256x256 for calculation.
*   MAE and MSE are calculated on raw pixel values (0-255).

### 2. No-Reference Metrics
Calculates **UCIQE**, **UIQM**, **NIQE**, **MUSIQ**, and **PS** (PSNR).
**Input**: Enhanced images folder. (Optional: GT folder for PSNR).

```bash
python nr/measure_metrics_nr.py --enhanced_dir /path/to/enhanced_images [--gt_dir /path/to/gt_images]
```
*   **UCIQE**: Underwater Color Image Quality Evaluation.
*   **UIQM**: Underwater Image Quality Measure.
*   **NIQE**: Natural Image Quality Evaluator.
*   **MUSIQ**: Multi-scale Image Quality Transformer.
*   **PS**: Peak Signal-to-Noise Ratio (calculated only if `--gt_dir` is provided).

### 3. Perceptual Score (PS) Metric
Calculates **LPIPS** (if GT provided) and **MUSIQ** (Perceptual Quality).
**Input**: Enhanced images folder. (Optional: GT folder).
**Input Resize**: Images are resized to **256x256** for this calculation.

```bash
python measure_ps.py --enhanced_dir /path/to/enhanced_images [--gt_dir /path/to/gt_images]
```

## Notes
- The NR script downloads pretrained models for NIQE and MUSIQ upon first run (via `pyiqa`).
- Ensure your images are in common formats (png, jpg, jpeg, bmp).
If you want to pre-download the model weights instead of downloading them at runtime, manually download and place them in the cache directory:

**MUSIQ weight:**
```bash
# Download
wget https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_koniq_ckpt-e95806b9.pth

# Place in cache (Linux)
mkdir -p ~/.cache/torch/hub/checkpoints/
mv musiq_koniq_ckpt-e95806b9.pth ~/.cache/torch/hub/checkpoints/
```

**NIQE weight:**
```bash
wget https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/niqe_modelparameters.mat
mkdir -p ~/.cache/torch/hub/pyiqa/
mv niqe_modelparameters.mat ~/.cache/torch/hub/pyiqa/
```

**LPIPS weight (for measure_ps.py):**
```bash
wget https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/LPIPS_v0.1_alex-df73285e.pth
mkdir -p ~/.cache/torch/hub/pyiqa/
mv LPIPS_v0.1_alex-df73285e.pth ~/.cache/torch/hub/pyiqa/

# Also need AlexNet backbone
wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
mv alexnet-owt-7be5be79.pth ~/.cache/torch/hub/checkpoints/
```

