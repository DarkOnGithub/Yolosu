# YOLO-based osu! Object Detection

This project implements object detection for osu! gameplay using YOLO (You Only Look Once). It can detect various game elements: hit circles, sliders, slider's ball, spinners, repeat slider.
Can run at over 100 fps on RTX 3070 mobile.

![](docs/output.gif)

## Features

- Dataset creation from beatmaps
- YOLO model training
- Video testing with trained models
- Support for TensorRT and ONNX model export
- Visualization capabilities

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project provides a command-line interface with three main commands:

### 1. Create Dataset

```bash
python main.py create-dataset --output-dir ./dataset [options]
```

Options:
- `--output-dir`: Output directory for the dataset (required)
- `--visualize`: Create visualization video
- `--num-beatmaps`: Number of beatmaps to process (default: 10)
- `--difficulties`: Number of difficulties per beatmap (default: 3)

### 2. Train Model

```bash
python main.py train --dataset-dir ./dataset [options]
```

Options:
- `--dataset-dir`: Directory containing the dataset (required)
- `--engine`: Export as TensorRT engine
- `--onnx`: Export as ONNX model
- `--fp16`: Use FP16 precision
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 64)
- `--imgsz`: Image size (default: 416)
- `--device`: Device to use (default: "0" for GPU)
- `--workers`: Number of worker threads (default: 4)

### 3. Test on Video

```bash
python main.py test --model-path ./models/best.pt --video-path ./test.mp4
```

Options:
- `--model-path`: Path to the trained model (required)
- `--video-path`: Path to the test video (required)
- `--imgsz`: Image size (default: 416)

## Configurable Parameters

### Object Counts
In `main.py`, you can modify the number of samples for each object type:
```python
object_counts = {
    'circle': 5000,
    'slider': 5000,
    'spinner': 2500,
    'approaching_circle': 5000,
    'ball': 5000,
    'repeat_point': 5000
}
```
A lower object count means faster training but might cause a worse accuracy.

### Model Configuration
- Base model: Currently using YOLOv8n (`yolo8n.pt`)
- Image size: Default 416x416
- Confidence threshold: 0.4
- IOU threshold: 0.45

### Dataset Configuration
- Default split ratio: 0.8 (80% training, 20% validation)
- Default FPS for visualization: 60

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Ultralytics YOLO
- Other dependencies listed in requirements.txt

## Notes

- The model is trained on YOLOv8n by default, but you can modify the base model in the code
- Dataset creation requires some beatmaps in the `./beatmaps` directory
- For best performance, use a GPU for training
- Visualization videos are created at 60 FPS by default
- This project uses [Danser](https://github.com/Wieku/danser-go) to create videos
