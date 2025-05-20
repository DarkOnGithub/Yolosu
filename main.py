import logging
import colorlog
import argparse
from dataset.dataset import Dataset
from emulator.config import DanserConfig
from ultralytics import YOLO

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(blue)s[%(asctime)s]%(reset)s %(log_color)s[%(levelname)s]%(reset)s %(purple)s[%(filename)s:%(lineno)d]%(reset)s: %(message)s',
    datefmt='%H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={
        'message': {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
    },
    style='%'
))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [handler]

    
def create_dataset(output_dir: str, config: DanserConfig, visualize: bool = False, num_beatmaps: int = 10, difficulties_per_beatmap: int = 3):
    dataset = Dataset.create_from_beatmaps(
        beatmaps_folder="./beatmaps", 
        output_folder=f"{output_dir}_dataset", 
        config=config, 
        num_beatmaps=num_beatmaps, 
        difficulties_per_beatmap=difficulties_per_beatmap, 
        visualize=False,
        object_counts={
            'circle': 5000,
            'slider': 5000,
            'spinner': 2500,
            'approaching_circle': 5000,
            'ball': 5000,
            'repeat_point': 5000
        }
    )
    dataset.export_yolo(output_folder=output_dir, split_ratio=0.8)


    if visualize:
        Dataset.create_visualization_video(
            yolo_dataset_path=output_dir,
            output_path="visualization.mp4",
            fps=60
        )


def train_model(dataset_dir: str, is_engine: bool = False, is_onnx: bool = False, fp16: bool = False, epochs: int = 100, batch: int = 64, imgsz: int = 416, device: str = "0", workers: int = 4):
    model = YOLO("yolo8n.pt")
    model.train(data=f"{dataset_dir}/dataset.yaml", epochs=epochs, batch=batch, imgsz=imgsz, device=device, workers=workers)
    if is_engine:
        model.export(format="engine", opset=12, dynamic=False, half=fp16)    
    if is_onnx:
        model.export(format="onnx", opset=12, dynamic=False, half=fp16)    

def test_on_video(model_path: str, video_path: str, imgsz: int = 416):
    model = YOLO(model_path)
    model.predict(video_path, show=False, conf=0.4, iou=0.45, imgsz=imgsz, save=True)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-based osu! object detection')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    dataset_parser = subparsers.add_parser('create-dataset', help='Create a dataset from beatmaps')
    dataset_parser.add_argument('--output-dir', required=True, help='Output directory for the dataset')
    dataset_parser.add_argument('--visualize', action='store_true', help='Create visualization video')
    dataset_parser.add_argument('--num-beatmaps', type=int, default=10, help='Number of beatmaps to process')
    dataset_parser.add_argument('--difficulties', type=int, default=3, help='Number of difficulties per beatmap')

    train_parser = subparsers.add_parser('train', help='Train the YOLO model')
    train_parser.add_argument('--dataset-dir', required=True, help='Directory containing the dataset')
    train_parser.add_argument('--engine', action='store_true', help='Export as TensorRT engine')
    train_parser.add_argument('--onnx', action='store_true', help='Export as ONNX model')
    train_parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--batch', type=int, default=64, help='Batch size')
    train_parser.add_argument('--imgsz', type=int, default=416, help='Image size')
    train_parser.add_argument('--device', default='0', help='Device to use (e.g., "0" for GPU)')
    train_parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')

    test_parser = subparsers.add_parser('test', help='Test the model on a video')
    test_parser.add_argument('--model-path', required=True, help='Path to the trained model')
    test_parser.add_argument('--video-path', required=True, help='Path to the test video')
    test_parser.add_argument('--imgsz', type=int, default=416, help='Image size')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.command == 'create-dataset':
        config = DanserConfig(
            width=1920,
            height=1080,
            fps=60,
            quickstart=True,
            mods="ATDT",
            dataset_dir="dataset_yolo",
            skin="Aristia(Edit)+trail"
        )  
        create_dataset(
            output_dir=args.output_dir,
            config=config,
            visualize=args.visualize,
            num_beatmaps=args.num_beatmaps,
            difficulties_per_beatmap=args.difficulties
        )
    elif args.command == 'train':
        train_model(
            dataset_dir=args.dataset_dir,
            is_engine=args.engine,
            is_onnx=args.onnx,
            fp16=args.fp16,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers
        )
    elif args.command == 'test':
        test_on_video(
            model_path=args.model_path,
            video_path=args.video_path,
            imgsz=args.imgsz
        )
    else:
        logger.error("Please specify a command. Use --help for more information.")