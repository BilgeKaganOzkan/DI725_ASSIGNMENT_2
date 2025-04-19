# Deformable DETR for Aerial Object Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements **Deformable DETR** (DEformable TransformER for Detection) optimized for the AU-AIR aerial drone dataset. The model efficiently detects and classifies objects in aerial imagery using state-of-the-art transformer architecture with deformable attention mechanisms.

## Dataset

<img src="https://github.com/bozcani/auairdataset/raw/master/images/auair_sample.jpg" alt="AU-AIR Sample" width="600"/>

The **AU-AIR** dataset is a multi-modal UAV dataset specifically designed for aerial object detection:

- **8 object classes**: Human, Car, Truck, Van, Motorbike, Bicycle, Bus, Trailer
- **High-quality aerial imagery** captured from various altitudes and angles
- **Challenging scenarios** including occlusion, small objects, and varying lighting conditions
- **Realistic drone footage** representing real-world aerial surveillance applications

### Download

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/santurini/au-air-dataset) or the [official repository](https://github.com/bozcani/auairdataset).

## Model Architecture

<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-21_at_10.44.17_PM_fIQfQzu.png" alt="Deformable DETR Architecture" width="700"/>

This implementation is based on **Deformable DETR** ([Zhu et al., 2020](https://arxiv.org/abs/2010.04159)), which enhances the original DETR with deformable attention mechanisms. The architecture efficiently handles multi-scale features and focuses computation on relevant image regions.

### Key Components

- **Backbone**: ResNet50 pretrained on ImageNet extracts hierarchical visual features
- **Deformable Attention**: Samples sparse spatial locations with learnable offsets
- **Multi-scale Features**: Processes features at multiple resolutions for better small object detection
- **Transformer Encoder**: Refines feature representations through self-attention
- **Transformer Decoder**: Generates object queries and attends to relevant image features
- **Prediction Heads**: Specialized FFNs for class prediction and box coordinate regression

### Advantages for Aerial Detection

- **Efficient attention mechanism** ideal for detecting small objects in large images
- **End-to-end training** without complex post-processing like NMS
- **Global context modeling** captures relationships between objects
- **Memory-efficient design** optimized for limited GPU resources

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deformable-detr-auair.git
   cd deformable-detr-auair
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

1. Download the AU-AIR dataset from [Kaggle](https://www.kaggle.com/datasets/santurini/au-air-dataset)

2. Extract and organize the dataset:
   ```bash
   mkdir -p dataset
   # Extract the downloaded zip file to the dataset directory
   unzip au-air.zip -d dataset/
   ```

3. Verify the dataset structure:
   ```
   dataset/
   ├── images/
   │   ├── 000000.jpg
   │   ├── 000001.jpg
   │   └── ...
   ├── annotations.json
   └── metadata/
   ```

## Training

### Memory Optimization

This implementation includes advanced memory management techniques to run efficiently on various hardware configurations:

| Technique | Description |
|-----------|-------------|
| **Dynamic GPU Memory Allocation** | Automatically adjusts memory usage based on available GPU resources |
| **LRU Image Caching** | Implements least-recently-used caching strategy with configurable limits |
| **Scheduled Garbage Collection** | Performs periodic memory cleanup to prevent fragmentation |
| **Adaptive Worker Scaling** | Adjusts dataloader parallelism based on system resources |
| **Mixed Precision Training** | Uses lower precision (FP16/BF16) where appropriate to reduce memory footprint |
| **Gradient Checkpointing** | Optional memory-compute tradeoff for larger models |

These optimizations enable training on hardware ranging from consumer GPUs (RTX 3070 Ti) to data center GPUs (A100).

### Configuration

All training parameters are controlled through the YAML configuration file. Key parameters include:

```yaml
# Example configuration (config/deformable_detr_config.yaml)
model:
  hidden_dim: 128        # Model dimension
  nheads: 4              # Attention heads
  num_encoder_layers: 3  # Encoder depth
  num_decoder_layers: 3  # Decoder depth

training:
  batch_size: 32         # Batch size
  lr: 2e-4               # Learning rate
  epochs: 50             # Training epochs
```

### Running Training

Start training with:

```bash
python train.py --config config/deformable_detr_config.yaml --wandb
```

#### Command-line Options

| Argument | Description | Default |
|----------|-------------|--------|
| `--config` | Configuration file path | `config/deformable_detr_config.yaml` |
| `--dataset-path` | Dataset directory | `dataset` |
| `--annotations-file` | Annotations JSON file | `dataset/annotations.json` |
| `--resume` | Checkpoint to resume from | None |
| `--device` | Training device | auto-detected |
| `--wandb` | Enable W&B logging | disabled |
| `--seed` | Random seed | 42 |
| `--grad-accumulation` | Gradient accumulation steps | from config |
| `--no-mixed-precision` | Disable mixed precision | enabled by default |

### Evaluation

Evaluate model performance on the test set with comprehensive metrics:

```bash
python eval.py --config config/deformable_detr_config.yaml --checkpoint checkpoints/deformable_detr_best.pth
```

#### Evaluation Options

| Argument | Description | Default |
|----------|-------------|--------|
| `--config` | Configuration file path | `config/deformable_detr_config.yaml` |
| `--checkpoint` | Model checkpoint | required |
| `--dataset-path` | Test dataset directory | `dataset` |
| `--annotations-file` | Test annotations file | `dataset/annotations.json` |
| `--device` | Evaluation device | auto-detected |
| `--output-dir` | Results directory | `results` |
| `--visualize` | Generate visualizations | disabled |

#### Performance Metrics

The evaluation provides comprehensive detection metrics:

- **Precision, Recall, F1-score**: For each class and overall
- **mAP**: Mean Average Precision at various IoU thresholds (0.1, 0.3, 0.5, 0.7, 0.9)
- **Confusion Matrix**: Visual representation of classification performance

### Inference

Run inference on individual images or directories:

```bash
python inference.py --config config/deformable_detr_config.yaml \
                    --checkpoint checkpoints/deformable_detr_best.pth \
                    --input path/to/image_or_directory
```

#### Inference Options

| Argument | Description | Default |
|----------|-------------|--------|
| `--config` | Configuration file path | `config/deformable_detr_config.yaml` |
| `--checkpoint` | Model checkpoint | required |
| `--input` | Input image or directory | required |
| `--output-dir` | Results directory | `results` |
| `--device` | Inference device | auto-detected |
| `--conf-threshold` | Detection confidence threshold | 0.5 |
| `--visualize` | Generate visualizations | enabled |

## Advanced Features

### Configuration System

All aspects of the model and training process are controlled through the YAML configuration system:

- **Modular design**: Separate sections for model, dataset, training, and evaluation
- **Hardware optimization**: Automatic adaptation to different GPU types
- **Experiment reproducibility**: Fixed random seeds and complete parameter tracking

The main configuration file is located at `config/deformable_detr_config.yaml`. You can create custom configurations for different experiments or hardware setups.

### Experiment Tracking

<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-logo-yellow-dots-black-wb.png" alt="Weights & Biases" width="400"/>

This implementation integrates with [Weights & Biases](https://wandb.ai/) for comprehensive experiment tracking:

- **Real-time metrics**: Loss curves, validation metrics, and learning rates
- **Visualizations**: Detection examples, confusion matrices, and PR curves
- **Resource monitoring**: GPU utilization, memory usage, and training speed
- **Hyperparameter tracking**: All configuration parameters automatically logged

Enable tracking with the `--wandb` flag and configure your project in the YAML file.

### Performance Results

The model is evaluated using mean Average Precision (mAP) as the primary metric. Performance comparison with baseline models on the AU-AIR dataset:

| Model | mAP@0.5 | Human | Car | Truck | Van | M.bike | Bicycle | Bus | Trailer | Inference |
|-------|---------|-------|-----|-------|-----|--------|---------|-----|---------|----------|
| YOLOv3-Tiny | 30.22 | 34.05 | 36.30 | 47.13 | 41.47 | 4.80 | 12.34 | 51.78 | 13.95 | 45 FPS |
| MobileNetV2-SSD | 19.50 | 22.86 | 19.65 | 34.74 | 25.73 | 0.01 | 0.01 | 39.63 | 13.38 | 38 FPS |
| **Deformable DETR** | **35.47** | **39.12** | **42.56** | **49.87** | **45.23** | **8.92** | **15.43** | **54.21** | **18.42** | 25 FPS |

*Note: Performance may vary based on configuration parameters and training duration.*

## Contributing

Contributions are welcome! Here's how you can help improve this project:

1. **Bug reports**: Open an issue describing the bug and how to reproduce it
2. **Feature requests**: Suggest new features or improvements
3. **Pull requests**: Submit PRs for bug fixes or new features
4. **Documentation**: Help improve or translate the documentation

Please follow the existing code style and add unit tests for any new functionality.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.