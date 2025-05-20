from emulator.beatmap import Beatmap, _parse_beatmap_path
import zipfile
from typing import List, Optional
import os
import logging
from emulator.difficulty import Difficulty

def _extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extracts a zip file to the specified directory if it contains .osu files.
    Raises FileNotFoundError if the archive does not contain .osu files.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if any(file.endswith('.osu') for file in zip_ref.namelist()):
            zip_ref.extractall(extract_to)
            logging.info(f"Extracted {zip_path} to {extract_to}.")
        else:
            raise FileNotFoundError(f"Archive not valid. Path {zip_path}.")

def _find_difficulties(folder_path: str) -> List[str]:
    """
    Finds all .osu files in the specified directory and its subdirectories.
    Returns a list of file paths.
    """
    difficulties = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.osu'):
                difficulties.append(os.path.join(root, file))
    return difficulties

def find_beatmap(search_term: str, is_full_path: bool = False, beatmaps_dir: str = "beatmaps") -> Optional[str]:
    """
    Search for a beatmap by different criteria.
    """
    if not os.path.exists(beatmaps_dir):
        logging.error(f"Beatmaps directory {beatmaps_dir} does not exist")
        return None

    for item in os.listdir(beatmaps_dir):
        item_path = os.path.join(beatmaps_dir, item)
        if not (os.path.isdir(item_path) or item_path.lower().endswith(('.osz', '.zip'))):
            continue
        try:
            beatmap_id, title, _= _parse_beatmap_path(os.path.basename(item_path))
            if not is_full_path and str(beatmap_id) == search_term or not is_full_path and search_term.lower() in title.lower() or search_term.lower() in title.lower():
                return item_path
        except Exception as e:
            logging.error(f"{e}")
            continue
    return None

def extract_beatmap(beatmap_path_or_search: str, is_full_path: bool = False, beatmaps_dir: str = "beatmaps") -> Optional[Beatmap]:
    """
    Extracts a beatmap from a .osz or .zip file, or uses the provided path if it is already a folder.
    Can also search for beatmaps by name or id.
    """
    if not is_full_path:
        beatmap_path = find_beatmap(beatmap_path_or_search, beatmaps_dir)
        if not beatmap_path:
            logging.error(f"No beatmap found matching {beatmap_path_or_search}")
            return None
    else:
        beatmap_path = beatmap_path_or_search
        
    beatmap_folder_path = beatmap_path
    if beatmap_path.endswith('.osz') or beatmap_path.endswith('.zip'):
        beatmap_folder_path = beatmap_path[:-4]
        try:
            _extract_zip(beatmap_path, beatmap_folder_path)
        except Exception as e:
            logging.error(f"Failed to extract {beatmap_path}: {e}")
            return None
            
    difficulties = _find_difficulties(beatmap_folder_path)
    logging.info(f"Found {len(difficulties)} difficulties files in {beatmap_folder_path}.")
    if not difficulties:
        logging.error(f"No difficulties found in {beatmap_folder_path}.")
        return None
    
    beatmap = Beatmap(beatmap_folder_path)
    for difficulty_path in difficulties:
        beatmap.add_difficulty(Difficulty(difficulty_path=difficulty_path))
    return beatmap