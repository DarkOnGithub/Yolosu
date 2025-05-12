from emulator.parser import beatmap_parser
import logging
import colorlog
from emulator.objects import HitCircle, Slider, Spinner, SliderBall
from emulator.objects.base import HitObjectType
from emulator.player import Player
from emulator.config import DanserConfig
import setup

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
config = DanserConfig(
    width=416,
    height=234,
    quickstart=True,
    mods="AT",
    dataset_dir="dataset_yolo_test",
    skin="Aristia(Edit)+trail"
)







































if __name__ == "__main__":
    from dataset.dataset import Dataset

    dataset = Dataset.create_from_beatmaps(
        beatmaps_folder="./beatmaps", 
        output_folder="./dataset_yolo_test", 
        config=config, 
        num_beatmaps=6, 
        difficulties_per_beatmap=2, 
        visualize=False,
        object_counts={
            'circle': 1000,
            'slider': 1000,
            'spinner': 500,
            'ball': 1000,
            'approaching_circle': 1000
        }
    )
    dataset.export_yolo(output_folder="./dataset_yolo_test_export", split_ratio=0.8)


    Dataset.create_visualization_video(
        yolo_dataset_path="./dataset_yolo_test_export",
        output_path="./dataset_yolo_test_export/visualization.mp4",
        fps=1
    )
