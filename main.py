from emulator.parser import beatmap_parser
import logging
import colorlog
from emulator.objects import HitCircle, Slider, Spinner, SliderBall
from emulator.objects.base import HitObjectType
from emulator.player import Player
from emulator.config import DanserConfig
import setup
from dataset import dataset_loader, dataset_writer


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

def main():
    from dataset.dataset import Dataset

    dataset = Dataset.create_from_beatmaps(
        beatmaps_folder="./beatmaps", 
        output_folder="./dataset_yolo", 
        config=config, 
        num_beatmaps=1, 
        difficulties_per_beatmap=1, 
        visualize=True,
        object_counts={
            'circle': 5000,
            'slider': 2000,
            'spinner': 500,
            'approaching_circle': 5000,
            'ball': 5000,
            'repeat_point': 5000
        }
    )
    dataset.export_yolo(output_folder="./dataset_yolo_export", split_ratio=0.8)


    Dataset.create_visualization_video(
        yolo_dataset_path="./dataset_yolo_test_export",
        output_path="./dataset_yolo_test_export/visualization.mp4",
        fps=1
    )
    
    
import cProfile
import io
import pstats
from pstats import SortKey
def profile_entire_code():
    profiler = cProfile.Profile()
    try:
        # Run the main function under profiling
        profiler.enable()
        main()  # Replace with your main function or script logic
        profiler.disable()
    except KeyboardInterrupt:
        profiler.disable()  # Ensure profiling stops
        print("\nCaught KeyboardInterrupt, printing profiling stats...")
    # Process and print stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats()  # Show all results
    
    # Optionally save the profile data
    profiler.dump_stats('profile_data.prof')
    
    return stream.getvalue()
if __name__ == "__main__":
    main()
# beatmap = beatmap_parser.extract_beatmap("The Violation", False)
# beatmap.parse_difficulties(None)
# diff = beatmap.get_difficulty(-1)
# player = Player(beatmap=beatmap, difficulty=diff, config=config)
# player.play(visualize=True)

# loader = dataset_loader.DatasetLoader(f"dataset_yolo_test/{beatmap.title}_{diff.difficulty_name}_index.json")
# loader.play_video(fps=60)





































