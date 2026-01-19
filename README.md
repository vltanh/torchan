# torchan: PyTorch Training and Evaluation Template

A flexible, configuration-driven PyTorch framework designed to streamline deep learning projects. It supports **Classification** and **Segmentation** tasks out-of-the-box, featuring modular components for models, datasets, losses, and metrics.

## Features

* **Config-Driven Training**: Define experiments entirely via YAML configuration files (models, optimizers, schedulers, etc.).
* **Modular Architecture**:
  * **Models**: Easy integration of backbones like ResNet and EfficientNet.
  * **Losses**: Support for `CrossEntropy`, `FocalLoss`, `DiceLoss`, and `MixedLoss` (combining multiple objectives).
  * **Metrics**: Track `Accuracy`, `F1`, `AUC`, `MeanIoU`, `DiceScore`, and `ConfusionMatrix`.
* **Training Loop**: Includes a `SupervisedTrainer` with automatic validation, checkpointing (best loss/metric), and TensorBoard logging.
* **Data Handling**: Custom `ImageFolderDataset` and `MNISTDataset` examples included.

## Project Structure

```text
.
├── configs/              # YAML configuration files for training/validation
├── torchan/
│   ├── datasets/         # Dataset implementations (MNIST, ImageFolder)
│   ├── losses/           # Loss functions (Classification & Segmentation)
│   ├── metrics/          # Evaluation metrics
│   ├── models/           # Model architectures and extractors
│   ├── loggers/          # Tensorboard logging wrapper
│   ├── trainers/         # Training loops (SupervisedTrainer)
│   └── utils/            # Utility scripts (getter, random_seed, etc.)
├── scripts/              # Helper scripts (e.g., train/val splitting)
├── train.py              # Main training entry point
└── test.py               # Inference/Testing entry point
```

## Getting Started

### 1. Prerequisites

Install the required dependencies. Based on the imports found in the codebase, you will need:

```bash
pip install torch torchvision numpy pandas pyyaml scikit-learn tqdm efficientnet_pytorch seaborn matplotlib
```

### 2. Configuration (`.yaml`)

Experiments are defined in YAML files located in `configs/`. See `configs/train/sample.yaml` for a reference.

**Key Config Sections:**

* **`model`**: Selects the architecture (e.g., `BaseClassifier` with `ResNetExtractor`).
* **`optimizer`** & **`scheduler`**: Standard PyTorch optimizers (Adam, SGD) and schedulers.
* **`dataset`**: definitions for `train` and `val` sets, including loader parameters like `batch_size`.
* **`loss`**: The loss function class name (e.g., `CrossEntropyLoss`).
* **`metric`**: A list of metrics to track during evaluation.

### 3. Training

To start a training session, run `train.py` with your config file.

```bash
# Basic usage
python train.py --config configs/train/sample.yaml

# Specify GPU device (e.g., cuda:0)
python train.py --config configs/train/sample.yaml --gpus 0
```

**What happens during training?**

* Logs are saved to `runs/<config_id>-<timestamp>`.
* Checkpoints are saved automatically when validation loss or metrics improve.
* TensorBoard logs are generated for real-time visualization.

### 4. Inference / Testing

Use `test.py` to generate predictions for a folder of images.

```bash
python test.py -d <path_to_images> -w <path_to_best_model.pth> -b 64 -o results.csv
```

**Arguments:**

* `-d`: Path to the folder containing query images.
* `-w`: Path to the trained weight file (`.pth`).
* `-g`: GPU index (optional).
* `-b`: Batch size (default: 64).
* `-o`: Output CSV filename (default: `test.csv`).

## Customization

### Adding a New Module (Model, Metric, Loss)

The framework uses a dynamic `getter` utility to instantiate classes defined in the config.

1. **Create your class** in the appropriate directory (e.g., `torchan/models/my_model.py`).
2. **Register it**: Ensure your class is imported in the `__init__.py` of its parent package.
* *Example*: If you create `MyNewLoss`, add `from .my_new_loss import MyNewLoss` to `torchan/losses/__init__.py`.

3. **Use it in Config**:
```yaml
loss:
  name: MyNewLoss
  args:
    weight: 0.5
```

### Adding a Dataset

Implement a class inheriting from `DatasetTemplate` or standard PyTorch `Dataset` and add it to `torchan/datasets/`. Ensure it returns `(image, label)` for training.

## Monitoring

To view training progress:

```bash
tensorboard --logdir runs
```
