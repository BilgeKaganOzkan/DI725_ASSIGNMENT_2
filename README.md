# Deformable DETR Implementation for AU-AIR Dataset

This repository contains an implementation of Deformable DETR (DEformable transformer for object DEtection) for the AU-AIR dataset. The model is designed to detect objects in aerial images from UAVs.

## Dataset

The AU-AIR dataset is a multi-modal UAV dataset for object detection. It contains images captured from a UAV along with corresponding annotations for objects such as humans, cars, trucks, vans, motorbikes, bicycles, buses, and trailers.

The dataset can be downloaded from [here](https://www.kaggle.com/datasets/santurini/au-air-dataset).

## Model Architecture

The implementation is based on the Deformable DETR architecture, which enhances the original DETR with deformable attention. The model uses a backbone (ResNet50) followed by a deformable transformer encoder-decoder architecture. This allows the model to focus on sparse spatial locations and efficiently handle multi-scale features.

The key components of the model are:

- **Backbone**: ResNet50 pretrained on ImageNet
- **Deformable Attention**: Multi-scale deformable attention mechanism
- **Transformer Encoder & Decoder**: Processes features from the backbone and generates object queries
- **Prediction Heads**: Linear layers to predict class labels and bounding box coordinates

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/deformable-detr-auair.git
cd deformable-detr-auair
```

2. Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Preprocessing

Before training, you need to organize the AU-AIR dataset properly. The dataset should be placed in the `dataset` folder.

### Training

To train the model, run:

```bash
python train.py --config config/deformable_detr_config.yaml --wandb
```

Command-line arguments:
- `--config`: Path to the configuration file
- `--dataset-path`: Path to the dataset directory (default: 'dataset')
- `--annotations-file`: Path to the annotations file (default: 'dataset/annotations.json')
- `--resume`: Path to checkpoint for resuming training (optional)
- `--device`: Device to use for training ('cuda' or 'cpu')
- `--wandb`: Enable wandb logging
- `--seed`: Random seed for reproducibility (default: 42)

### Evaluation

To evaluate the model on the test set, run:

```bash
python eval.py --checkpoint checkpoints/deformable_detr_best.pth
```

## Configuration

The model configuration is defined in `config/deformable_detr_config.yaml`. You can modify this file to change the model architecture, training parameters, and more.

## Experiment Tracking

The training script integrates with Weights & Biases (wandb) for experiment tracking. To enable it, use the `--wandb` flag and set your wandb credentials in the configuration file.

## Results

The model is evaluated using mean Average Precision (mAP) as the primary metric. The baseline models (YOLOv3-Tiny and MobileNetV2-SSDLite) achieve the following performance:

| Model | Human | Car | Truck | Van | M.bike | Bicycle | Bus | Trailer | mAP |
|-------|-------|-----|-------|-----|--------|---------|-----|---------|-----|
| YOLOV3-Tiny | 34.05 | 36.30 | 47.13 | 41.47 | 4.80 | 12.34 | 51.78 | 13.95 | 30.22 |
| MobileNetV2-SSDLite | 22.86 | 19.65 | 34.74 | 25.73 | 0.01 | 0.01 | 39.63 | 13.38 | 19.50 |

The Deformable DETR model aims to improve upon these baseline results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The AU-AIR dataset is provided by [AU-AIR](https://github.com/bozcani/auairdataset)
- The Deformable DETR implementation is inspired by the [original paper](https://arxiv.org/abs/2010.04159) by Zhu et al.
- Thanks to [Facebook Research](https://github.com/facebookresearch/detr) for the original DETR implementation 