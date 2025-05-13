from typing import List, Tuple, Union
from .difficulty import Difficulty
import logging
import re
import os

BEATMAP_PATH_PATTERN = r"^(\d+)\s+([^-]+?)\s+-\s+(.+)$"

def _parse_beatmap_path(beatmap_path: str) -> Tuple[int, str, str]:
    """
    Parses the beatmap path to extract the beatmap ID, author and title .
    """
    match = re.match(BEATMAP_PATH_PATTERN, beatmap_path.split('/')[-1])
    if match:
        beatmap_id = int(match.group(1))
        author = match.group(2).strip()
        title = match.group(3).strip()

        return beatmap_id, title, author
    else:
        raise ValueError(f"Invalid beatmap path format: {beatmap_path}")

class Beatmap:
    difficulties: List[Difficulty]
    title: str 
    beatmap_id: int
    author: str
    folder_path: str
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        (self.beatmap_id, self.title, self.author) = _parse_beatmap_path(os.path.basename(folder_path))
        self.difficulties = []
        logging.info(f"Initialized Beatmap with ID: {self.beatmap_id}, Title: {self.title}, Author: {self.author}")    
        
    def add_difficulty(self, difficulty: Difficulty) -> None:
        """Adds a difficulty to the beatmap."""
        self.difficulties.append(difficulty)
        logging.info(f"Added difficulty: {difficulty} to beatmap: {self.title}")
    
    def parse_difficulties(self, difficulty_names: Union[List[str] | None] = None) -> None:
        """
        Parse the difficulties of the beatmap.
        Note: if not difficulty_names is provided, all difficulties will be parsed.
        """
        difficulty_names = difficulty_names or [difficulty.difficulty_name for difficulty in self.difficulties]
        for difficulty in self.difficulties:
            if difficulty.difficulty_name in difficulty_names:
                difficulty.parse()
                
    def parse_difficulties_from_index(self, index: List[int]) -> None:
        """Parses the difficulty of the beatmap from the index."""
        self.parse_difficulties([self.difficulties[i].difficulty_name for i in index])
    
    def get_difficulty(self, difficulty_name: Union[str, int]) -> Difficulty:
        """Returns the difficulty of the beatmap."""
        if isinstance(difficulty_name, int):
            return self.difficulties[difficulty_name]
        
        for difficulty in self.difficulties:
            if difficulty.difficulty_name == difficulty_name:
                return difficulty
        
        raise ValueError(f"Difficulty {difficulty_name} not found in beatmap: {self.title}")

    
    def __repr__(self) -> str:
        return f"Beatmap(Title: {self.title}, Difficulties: {len(self.difficulties)})"