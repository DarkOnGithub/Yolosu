from .difficulty_sections import GeneralSection, MetadataSection, DifficultySection, TimingPointsSection, ColoursSection, HitObjectsSection
from .parser.difficulty_parser import parse_difficulty
import re
import logging

def _extract_difficulty_name(beatmap_title):
    """Extracts the difficulty name from the beatmap title."""
    match = re.search(r'\[([^\[\]]*)\](?![^\[]*\])', beatmap_title)
    if match:
        return match.group(1)  
    return None

class Difficulty:
    difficulty_path: str
    general: GeneralSection
    metadata: MetadataSection
    difficulty: DifficultySection
    events: None
    timing_points: TimingPointsSection
    colours: ColoursSection
    hit_objects: HitObjectsSection
    
    def __init__(self, difficulty_path: str) -> None:
        self.difficulty_path = difficulty_path
        self.difficulty_name = _extract_difficulty_name(difficulty_path.split('/')[-1])
        self.general = GeneralSection()
        self.metadata = MetadataSection()
        self.difficulty = DifficultySection()
        self.events = None
        self.timing_points = TimingPointsSection()
        self.colours = ColoursSection()
        self.hit_objects = HitObjectsSection()

    def parse(self):
        """Parses the difficulty file."""
        logging.info(f"Parsing difficulty: {self.difficulty_name}")
        parse_difficulty(self.difficulty_path, self)



    def __repr__(self) -> str:
        return f"Difficulty(name: {self.difficulty_name})"