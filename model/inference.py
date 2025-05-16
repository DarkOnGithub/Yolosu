from model.reinforcement_learning.config import RL_Config
from model.reinforcement_learning.tracker import Tracker
from model.reinforcement_learning.game_state import GameState
import time

class Inference:
    def __init__(self, config: RL_Config):
        self.config = config
        self.tracker = Tracker(config=config)
        self.game_state = GameState(config=config)
        
    
    def train_on_beatmap(self, beatmap_name: str):
        pass
    
    def run(self):
        while True:
            pass