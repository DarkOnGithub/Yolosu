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
    mods="AT"
)
beatmap = beatmap_parser.extract_beatmap("Time to say Goodbye", is_full_path=False)
beatmap.parse_difficulties(["No Return"])
difficulty = beatmap.get_difficulty("No Return")
player = Player(
    beatmap=beatmap,
    difficulty=difficulty,
    config=config
)

player.play()