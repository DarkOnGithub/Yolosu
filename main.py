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
    fps=60,
    quickstart=True,
    mods="AT",
    dataset_dir="dataset_yolo",
    skin="Aristia(Edit)+trail"
)

from model.reinforcement_learning.config import RL_Config
from model.reinforcement_learning.tracker import Tracker
from model.inference import Inference
WEIGHT_PATH = r"runs\detect\train6\weights\best.engine"


def main():
    inference = Inference(config=RL_Config(
        classes=["circle", "slider", "spinner", "approaching_circle", "ball", "repeat_point"],
    ))
    inference.run()


if __name__ == "__main__":
    main()
# beatmap = beatmap_parser.extract_beatmap("The Violation", False)
# beatmap.parse_difficulties(None)
# diff = beatmap.get_difficulty(-1)
# player = Player(beatmap=beatmap, difficulty=diff, config=config)
# player.play(visualize=True)

# loader = dataset_loader.DatasetLoader(f"dataset_yolo_test/{beatmap.title}_{diff.difficulty_name}_index.json")
# loader.play_video(fps=60)





































