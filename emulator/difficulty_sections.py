from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from .objects import HitCircle, Slider, Spinner, base

class Section(ABC):
    @abstractmethod
    def parse(self, content: str) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

@dataclass
class GeneralSection(Section):
    audio_filename: str = ""
    audio_lead_in: int = 0
    preview_time: int = -1
    countdown: int = 1
    sample_set: str = "Normal"
    stack_leniency: float = 0.7
    mode: int = 0
    letterbox_in_breaks: bool = False
    use_skin_sprites: bool = False
    overlay_position: str = "NoChange"
    skin_preference: str = ""
    epilepsy_warning: bool = False
    countdown_offset: int = 0
    special_style: bool = False
    widescreen_storyboard: bool = False
    samples_match_playback_rate: bool = False

    def parse(self, content: str) -> None:
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if hasattr(self, key.lower()):
                    attr = getattr(self, key.lower())
                    if isinstance(attr, bool):
                        setattr(self, key.lower(), value == '1')
                    elif isinstance(attr, int):
                        setattr(self, key.lower(), int(value))
                    elif isinstance(attr, float):
                        setattr(self, key.lower(), float(value))
                    else:
                        setattr(self, key.lower(), value)

    def __repr__(self) -> str:
        return f"GeneralSection(audio_filename={self.audio_filename}, mode={self.mode})"

@dataclass
class MetadataSection(Section):
    title: str = ""
    title_unicode: str = ""
    artist: str = ""
    artist_unicode: str = ""
    creator: str = ""
    version: str = ""
    source: str = ""
    tags: List[str] = None
    beatmap_id: int = 0
    beatmap_set_id: int = 0

    def __init__(self):
        self.tags = []

    def parse(self, content: str) -> None:
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == "Tags":
                    self.tags = value.split()
                elif hasattr(self, key.lower()):
                    attr = getattr(self, key.lower())
                    if isinstance(attr, int):
                        setattr(self, key.lower(), int(value))
                    else:
                        setattr(self, key.lower(), value)

    def __repr__(self) -> str:
        return f"MetadataSection(title={self.title}, artist={self.artist}, version={self.version})"
def _camel_case_to_snake_case(string: str) -> str:
    return ''.join(['_' + c.lower() if c.isupper() else c for c in string]).lstrip('_')
@dataclass
class DifficultySection(Section):
    hp_drain_rate: float = 5.0
    circle_size: float = 5.0
    overall_difficulty: float = 5.0
    approach_rate: float = 5.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0

    def parse(self, content: str) -> None:
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = _camel_case_to_snake_case(key.strip())
                value = value.strip()

                if hasattr(self, key):
                    setattr(self, key, float(value))
    
    def get_radius(self) -> float:
        return 54.4 - 4.48 * self.circle_size

    def get_approach_time(self) -> float:
        if self.approach_rate < 5:
            return 1200 + 120 * (5 - self.approach_rate)
        return 1200 - 150 * (self.approach_rate - 5)
            

    def __repr__(self) -> str:
        return f"DifficultySection(hp={self.hp_drain_rate}, cs={self.circle_size}, od={self.overall_difficulty}, ar={self.approach_rate})"

@dataclass
class Event:
    event_type: str
    start_time: int
    params: List[str]

class EventsSection(Section):
    events: List[Event]

    def __init__(self):
        self.events = []

    def parse(self, content: str) -> None:
        for line in content.split('\n'):
            if ',' in line:
                parts = line.split(',')
                event_type = parts[0].strip()
                start_time = int(parts[1].strip())
                params = [p.strip() for p in parts[2:]]
                self.events.append(Event(event_type, start_time, params))

    def __repr__(self) -> str:
        return f"EventsSection(events={len(self.events)})"

@dataclass
class TimingPoint:
    time: int
    beat_length: float
    meter: int
    sample_set: int
    sample_index: int
    volume: int
    uninherited: bool
    effects: int

class TimingPointsSection(Section):
    points: List[TimingPoint]

    def __init__(self):
        self.points = []

    def parse(self, content: str) -> None:
        for line in content.split('\n'):
            if ',' in line:
                parts = line.split(',')
                time = (float(parts[0].strip()))
                beat_length = float(parts[1].strip())
                meter = int(parts[2].strip())
                sample_set = int(parts[3].strip())
                sample_index = int(parts[4].strip())
                volume = int(parts[5].strip())
                uninherited = int(parts[6].strip()) == 1
                effects = int(parts[7].strip())
                self.points.append(TimingPoint(time, beat_length, meter, sample_set, sample_index, volume, uninherited, effects))

    def __repr__(self) -> str:
        return f"TimingPointsSection(points={len(self.points)})"

@dataclass
class Colour:
    name: str
    r: int
    g: int
    b: int

class ColoursSection(Section):
    colours: List[Colour]

    def __init__(self):
        self.colours = []

    def parse(self, content: str) -> None:
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key.startswith("Combo"):
                    rgb = value.split(',')
                    self.colours.append(Colour(key, int(rgb[0]), int(rgb[1]), int(rgb[2])))

    def __repr__(self) -> str:
        return f"ColoursSection(colours={len(self.colours)})"


class HitObjectsSection(Section):
    objects: List[base.HitObject]

    def __init__(self):
        self.objects = []

    def parse(self, content: str, approach_time: float) -> None:
        for line in content.split('\n'):
            if ',' in line:
                parts = line.split(',')
                x = int(parts[0].strip())
                y = int(parts[1].strip())

                time = int(parts[2].strip())
                type = int(parts[3].strip())
                hit_sound = int(parts[4].strip())
                params = [p.strip() for p in parts[5:]]

                if type & base.HitObjectType.HIT_CIRCLE.value:  
                    self.objects.append(HitCircle(x, y, time, base.HitObjectType.HIT_CIRCLE,  approach_time, hit_sound))
                elif type & base.HitObjectType.SLIDER.value:  
                    curve_data = params[0].split('|')
                    curve_type = curve_data[0]
                    curve_points = [(int(p.split(':')[0]), int(p.split(':')[1])) for p in curve_data[1:]]
                    slides = int(params[1])
                    length = float(params[2])
                    edge_sounds = [int(s) for s in params[3].split('|')] if len(params) > 3 else []
                    edge_sets = [s.split(':') for s in params[4].split('|')] if len(params) > 4 else []
                    hit_sample = params[5].split(':') if len(params) > 5 else []
                    self.objects.append(Slider(x, y, time, base.HitObjectType.SLIDER, hit_sound, approach_time,
                                            curve_type, curve_points, slides, length, 
                                            edge_sounds, edge_sets, hit_sample))
                elif type & base.HitObjectType.SPINNER.value: 
                    end_time = int(params[0])
                    hit_sample = params[1].split(':') if len(params) > 1 else []
                    self.objects.append(Spinner(256, 192, time, base.HitObjectType.SPINNER, hit_sound,
                                             end_time, hit_sample))

    def __repr__(self) -> str:
        return f"HitObjectsSection(objects={len(self.objects)})"
