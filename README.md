# Froth DeepLab Family – Baseline Semantic Segmentation with DeepLabV3 and DeepLabV3+

## Authors
- Sina Lotfi
- Reza Dadbin


## Project Overview
This repository provides conventional convolutional neural network (CNN) baselines for froth segmentation in flotation cell imagery. It implements two DeepLab-based model families—DeepLabV3 with a ResNet-101 backbone from torchvision and DeepLabV3+ with a ResNet-152 backbone (output stride 8) from segmentation-models-pytorch. Both families train and evaluate on the same LabelMe-style dataset used by the complementary froth-SAM projects, enabling fair comparisons between promptable SAM variants and fully supervised baselines. Use these scripts to benchmark strong, reproducible CNN models against prompt-based approaches on identical data.

## Environment and Dependencies
- Recommended Python version: **3.10+** (the code has been exercised on Python 3.10).
- Install Python dependencies from the repository root:
  ```bash
  pip install -r requirements.txt
  ```
- DeepLabV3+ relies on [`segmentation-models-pytorch`](https://github.com/qubvel/segmentation_models.pytorch) and [`timm`](https://github.com/huggingface/pytorch-image-models); both packages are included in `requirements.txt`.

## Dataset Format and Folder Structure
The training scripts expect the following directory layout relative to the project root:
```
data/
├── train/
└── test/
```

### `data/train`
- Contains the training split.
- Each sample consists of:
  - A TIFF image (`.tif` or `.tiff`).
  - A matching LabelMe JSON file with the same filename stem.
- Example:
  ```
  data/train/0001.tif
  data/train/0001.json
  data/train/0002.tiff
  data/train/0002.json
  ```

### `data/test`
- Serves as the validation split within this repository.
- Mirrors the `data/train` structure with paired image/annotation files.

### LabelMe Annotation Expectations
- JSON files follow the standard LabelMe schema.
- Polygons are listed in the top-level `"shapes"` array; each entry includes a `"label"` and polygon `"points"`.
- The froth foreground must be labeled as `"froth"`.
- Non-`"froth"` labels are ignored. Optionally, labels such as `"ignore"`, `"_ignore"`, or other variants can be mapped to the ignore index.

### Class Mapping
- The configuration assumes exactly two semantic classes:
  - `0`: background
  - `1`: froth
- If your dataset names the froth class differently, update the dataset parsing logic accordingly.

## Configuration Summary (`config.py`)
`config.py` centralizes dataset paths, model hyperparameters, and runtime options:

- **Paths**
  - `data_root`: defaults to `./data`
  - `train_dir`: `data/train`
  - `val_dir`: `data/test`
  - `outputs_root`: `./outputs`
  - `deeplab_out`: `./outputs/deeplabv3_resnet101_finetune_out`
- **Classes and Masks**
  - `class_names = ["background", "froth"]`
  - `num_classes = 2`
  - `ignore_index = 255`
- **Data Pipeline Hyperparameters**
  - `resize_short_train`, `resize_short_val`
  - `train_crop_size`, `val_crop_size`
  - `batch_size_train`, `batch_size_val`
- **Optimization Settings**
  - `epochs`, `accum_steps`, `use_amp`
  - `backbone_lr`, `head_lr`
  - `weight_decay`, `grad_clip_norm`, `power_poly`
- **Runtime Controls**
  - `device` (auto-detects CUDA when available)
  - `seed`
  - `resume_ckpt` (path to resume training from a checkpoint)

`Config.setup()` creates the data and output directories if they do not already exist, ensuring training runs without manual folder preparation.

## DeepLabV3 (ResNet-101) Pipeline

### Dataset Loader (`deeplab_froth/data/froth_dataset.py`)
- `FrothLabelMeDataset` reads image/annotation pairs from a directory.
- Rasterizes LabelMe polygons into masks with values `{0: background, 1: froth, 255: ignore}`.
- Training transforms: resize shorter side, random horizontal flip, random crop.
- Validation transforms: resize shorter side, center crop with padding when necessary.
- Outputs tensors:
  - Images normalized with ImageNet mean/std (shape `C×H×W`).
  - Masks as `long` tensors (shape `H×W`).

### Model Builder (`deeplab_froth/models/deeplabv3.py`)
- `build_deeplab_model()` loads `torchvision.models.segmentation.deeplabv3_resnet101` with ImageNet weights.
- Replaces the classifier head with `DeepLabHead(2048, num_classes)` for two-class segmentation.

### Training Script (`scripts/train_v3.py`)
- Uses `FrothLabelMeDataset` for both training and validation splits.
- Optimizer: `AdamW` with differential learning rates for backbone and classifier head.
- Loss: `CrossEntropyLoss(ignore_index=255)`.
- Scheduler: polynomial LR decay per epoch.
- Mixed precision supported via AMP on CUDA.
- Checkpoints saved in `outputs/deeplabv3_resnet101_finetune_out/`:
  - `last.pth`: latest state after each epoch.
  - `best_mIoU.pth`: best validation IoU so far.

### Evaluation Script (`scripts/eval_v3.py`)
- Loads `best_mIoU.pth` if available, otherwise `last.pth`, else falls back to the ImageNet-pretrained weights.
- Computes average loss and mean IoU over the selected split.

### Prediction Script (`scripts/predict_v3.py`)
- Loads the best (or latest) checkpoint automatically.
- Runs inference on `train` or `val` split.
- Writes binary froth masks to `outputs/pred_masks/deeplabv3_resnet101/<split>/mask_XXXX.png` (0 = background, 255 = froth).

## Running DeepLabV3 Experiments

### Train
```bash
python -m scripts.train_v3 --epochs 50
```
- Overrides the default epoch count in `config.py` when provided.
- Automatically initializes the output directory and starts from ImageNet weights for the backbone plus a randomly initialized head.
- Logs progress per epoch and saves `last.pth` / `best_mIoU.pth` as described above.
- To resume, set `Config.resume_ckpt` to a checkpoint path and rerun the command.

### Evaluate
```bash
python -m scripts.eval_v3 --split val
```
- `--split` accepts `val` (default; uses `data/test`) or `train`.
- Automatically loads the finetuned checkpoint hierarchy and prints mean loss and mean IoU.
- Falls back to evaluating the ImageNet-pretrained model if no fine-tuned weights exist.

### Predict
```bash
python -m scripts.predict_v3 --split val --thr 0.5
```
- `--split`: choose `val` (default) or `train`.
- `--thr`: probability threshold for class 1 (`froth`); values above the threshold are written as 255 in the output mask.
- `--argmax` (optional): use argmax across logits instead of thresholding the froth probability.
- Outputs masks to `outputs/pred_masks/deeplabv3_resnet101/<split>/` as single-channel 8-bit PNGs.

## DeepLabV3+ (ResNet-152, OS=8) Pipeline

- Built on [`segmentation-models-pytorch`](https://github.com/qubvel/segmentation_models.pytorch) with configuration:
  - `smp.DeepLabV3Plus(
        encoder_name="resnet152",
        encoder_weights="imagenet",
        encoder_output_stride=8,
        in_channels=3,
        classes=2,
    )`
- Shares the same `FrothLabelMeDataset` preprocessing and LabelMe inputs.
- Dedicated scripts:
  - Training: `scripts/train_v3plus.py`
  - Evaluation: `scripts/eval_v3plus.py`
  - Prediction: `scripts/predict_v3plus.py`
- Checkpoints stored separately, e.g. `outputs/deeplabv3plus_resnet152_finetune_out/`.

## Running DeepLabV3+ Experiments

### Train
```bash
python -m scripts.train_v3plus --epochs 50
```
- Uses the same dataset splits as DeepLabV3.
- Writes checkpoints to the DeepLabV3+ output directory (`last.pth`, `best_mIoU.pth`).
- Hyperparameters mirror those defined in `config.py` and script-level defaults.

### Evaluate
```bash
python -m scripts.eval_v3plus --split val
```
- `--split`: choose between `val` (default, `data/test`) and `train`.
- Loads `best_mIoU.pth` or `last.pth` automatically.
- Reports loss and mean IoU.

### Predict
```bash
python -m scripts.predict_v3plus --split val --thr 0.5
```
- `--split`: `val` or `train`.
- `--thr`: froth probability threshold.
- May provide an `--argmax` flag analogous to the V3 script (if present in the code) for argmax decoding.
- Saves masks to `outputs/pred_masks/deeplabv3plus_resnet152/<split>/mask_XXXX.png` with 0 = background and 255 = froth.

## Notes and Tips
- **GPU Memory**: Both DeepLabV3 and DeepLabV3+ are memory intensive. Reduce `batch_size_train`/`batch_size_val` in `config.py` if you encounter out-of-memory errors. For DeepLabV3+, switching `encoder_output_stride` to 16 can further lower memory usage (requires code edit).
- **Mixed Precision**: Enable or disable AMP via `Config.use_amp` (or `USE_AMP` flags). Disable AMP if you observe numerical instability.
- **Reproducibility**: A fixed seed (`Config.seed`) is applied to NumPy and PyTorch. Exact reproducibility across different hardware or framework versions is not guaranteed.
- **Comparisons with SAM**: This repository complements the `froth-sam-family` project. DeepLab models here act as strong, non-promptable baselines that can be compared against SAM/HQ-SAM/MedSAM models on identical froth datasets, facilitating comprehensive benchmarking.