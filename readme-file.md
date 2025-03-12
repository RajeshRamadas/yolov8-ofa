# YOLOv8 Training with Once-For-All (OFA) and MIT

This repository contains a script for training YOLOv8 models using the Once-For-All (OFA) methodology with Multiple Inheritance Training (MIT). This approach allows you to train a single model that can be deployed across different hardware platforms with varying computational constraints.

## Features

- **One Model, Multiple Deployments**: Train a single model that can be adapted to multiple devices
- **Multiple Inheritance Training**: Ensures all sub-networks are well-trained
- **Configurable Scaling**: Supports variable depth scales, width scales, and input resolutions
- **Comprehensive Metrics**: Evaluates all model configurations with detailed performance metrics
- **Visualization Tools**: Generates various plots to help analyze model trade-offs
- **Export Options**: Exports optimized models in various formats (ONNX, TFLite, TorchScript)

## Requirements

```
ultralytics>=8.0.0
torch>=1.8.0
pyyaml>=6.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/yolov8-ofa-mit.git
cd yolov8-ofa-mit

# Install requirements
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python yolov8_ofa_training.py --data path/to/data.yaml --model-size n --epochs 100 --batch-size 16
```

### Using a Pretrained Model

```bash
python yolov8_ofa_training.py --data path/to/data.yaml --model-size n --pretrained
```

### Training with Specific Scaling Options

```bash
python yolov8_ofa_training.py --data path/to/data.yaml --model-size n \
  --depth-scales 0.33 0.67 1.0 \
  --width-scales 0.25 0.5 0.75 1.0 \
  --input-sizes 320 416 640
```

### Training on Multiple GPUs

```bash
python yolov8_ofa_training.py --data path/to/data.yaml --model-size n --device 0,1,2,3
```

### Exporting Models After Training

```bash
python yolov8_ofa_training.py --data path/to/data.yaml --model-size n --export
```

## Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Path to data.yaml file | Required |
| `--model-size` | YOLOv8 model size (n, s, m, l, x) | `n` |
| `--epochs` | Number of training epochs | `100` |
| `--batch-size` | Batch size for training | `16` |
| `--device` | Device to train on (e.g., 0, 0,1,2,3, cpu) | auto-select |
| `--project` | Project name for organizing results | `yolov8_ofa` |
| `--name` | Experiment