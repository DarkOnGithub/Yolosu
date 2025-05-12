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
    width=960,
    height=540,
    quickstart=True,
    mods="AT",
    dataset_dir="dataset_yolo_test"
)

# BEATMAP_NAME = "Time to say Goodbye"
# DIFFICULTY = "No Return"
# beatmap = beatmap_parser.extract_beatmap(BEATMAP_NAME, is_full_path=False)
# beatmap.parse_difficulties(None)
# difficulty = beatmap.get_difficulty(DIFFICULTY)
# player = Player(
#     beatmap=beatmap,
#     difficulty=difficulty,
#     config=config
# )

# player.play(visualize=False)
# from dataset.dataset_loader import DatasetLoader
# dataset_loader = DatasetLoader("dataset_yolo_test/Time to Say Goodbye (TV Size)_No Return_dataset.json")
# info = dataset_loader.get_dataset_info()
# print(info)
# dataset_loader.play_video(fps=60)
from dataset.dataset_balancer import DatasetBalancer

balancer = DatasetBalancer(
    source_datasets=['dataset_yolo_test'],
    output_dir='balanced_dataset',
    target_counts={
        'circle': 100,
        'slider': 50,
        'spinner': 25
    }
)

balancer.create_balanced_dataset()