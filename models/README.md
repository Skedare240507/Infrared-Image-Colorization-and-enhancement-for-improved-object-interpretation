# Pix2Pix Thermal Colorization

Complete PyTorch Pix2Pix implementation for translating **100m thermal GeoTIFF** inputs to **RGB** outputs.

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| Generator | `models/architectures/unet_generator.py` | 8-layer U-Net with skip connections, tanh output |
| Discriminator | `models/architectures/patchgan_discriminator.py` | 70×70 PatchGAN on `[thermal, rgb]` |
| Combined model | `models/architectures/pix2pix.py` | Wraps generator + discriminator |
| Losses | `training/losses/gan_loss.py` | LSGAN + λ·L1 (default λ=100) |

## Dataset layout

```
data/splits/
  train/
    thermal/
      scene_001.tif    # 100m thermal GeoTIFF
    rgb/
      scene_001.png    # aligned RGB reference
  val/
    thermal/
    rgb/
  test/
    thermal/
    rgb/
```

Paired files must share the same filename stem. Thermal TIFFs are read with **Rasterio**; RGB images with **OpenCV/Rasterio**.

## Training

```bash
python -m training.train --config configs/experiments/pix2pix.yaml --data-root data/splits
```

Checkpoints saved to `models/weights/`:

| File | Contents |
|------|----------|
| `best.pt` | Full checkpoint (G + D + optimizers) |
| `colorization.pt` | Generator weights for inference/API |
| `last.pt` | Final epoch checkpoint |

## Inference

```bash
python scripts/run_inference.py \
  --input data/splits/test/thermal/scene_001.tif \
  --output outputs/scene_001_colorized.png \
  --checkpoint models/weights/colorization.pt
```

Python API:

```python
from inference.pix2pix_inference import Pix2PixInference

engine = Pix2PixInference("models/weights/colorization.pt", device="auto")
rgb = engine.colorize_file("thermal.tif", "colorized.png")
```

## Evaluation

```bash
python -m training.evaluate \
  --checkpoint models/weights/best.pt \
  --data-root data/splits \
  --split val \
  --output-dir outputs/eval
```

## Hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| Input channels | 1 (thermal) |
| Output channels | 3 (RGB) |
| Image size | 256×256 |
| ngf / ndf | 64 |
| Learning rate | 2×10⁻⁴ |
| β₁ | 0.5 |
| λ_L1 | 100 |
| Epochs | 200 |

Minimum training/inference spatial size is **256×256** due to the 8-layer U-Net downsampling path.
