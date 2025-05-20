from dataclasses import dataclass
from typing import Optional
from .beatmap import Beatmap
from .difficulty import Difficulty

@dataclass
class DanserConfig:
    """Configuration for danser video generation"""
    width: int = 1920
    height: int = 1080
    fps: int = 60
    
    speed: float = 1.0
    pitch: float = 1.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    skip_intro: bool = False
    quickstart: bool = False
    
    record: bool = True
    output_name: Optional[str] = None
    output_dir: str = "videos"
        
    cursors: int = 1
    tag: int = 1
    
    mods: Optional[str] = None
    skin: Optional[str] = None
    
    circle_size: Optional[float] = None
    approach_rate: Optional[float] = None
    overall_difficulty: Optional[float] = None
    hp_drain: Optional[float] = None
    
    dataset_dir: str = "dataset_yolo"
    
    def to_danser_args(self, beatmap: Beatmap, difficulty: Difficulty) -> str:
        """Convert config to danser command line arguments"""
        args = [
            f'-title="{beatmap.title}"',
            f'-difficulty="{difficulty.metadata.version}"',
            "-noupdatecheck",
            "-preciseprogress"
        ]
        
        if self.speed != 1.0:
            args.append(f'-speed={self.speed}')
        if self.pitch != 1.0:
            args.append(f'-pitch={self.pitch}')
        if self.start_time is not None:
            args.append(f'-start={self.start_time}')
        if self.end_time is not None:
            args.append(f'-end={self.end_time}')
        if self.skip_intro:
            args.append('-skip')
        if self.quickstart:
            args.append('-quickstart')
        if self.record:
            args.append('-record')
        if self.output_name:
            args.append(f'-out="..\\..\\{self.output_dir}\\{self.output_name.replace(".mp4", "")}"')
        if self.cursors > 1:
            args.append(f'-cursors={self.cursors}')
        if self.tag > 1:
            args.append(f'-tag={self.tag}')
        if self.mods:
            args.append(f'-mods={self.mods}')
        if self.skin:
            args.append(f'-skin={self.skin}')
        if self.circle_size is not None:
            args.append(f'-cs={self.circle_size}')
        if self.approach_rate is not None:
            args.append(f'-ar={self.approach_rate}')
        if self.overall_difficulty is not None:
            args.append(f'-od={self.overall_difficulty}')
        if self.hp_drain is not None:
            args.append(f'-hp={self.hp_drain}')
            
        return " ".join(args) 