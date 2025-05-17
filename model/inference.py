from model.reinforcement_learning.config import RL_Config
from model.reinforcement_learning.tracker import Tracker
from model.reinforcement_learning.game_state import GameState
import time
from enum import Enum
import threading
from model.reinforcement_learning.session import Session
from model.reinforcement_learning.process_manager import ProcessManager

class InferenceState(Enum):
    pass
    
class Inference:
    def __init__(self, config: RL_Config):
        self.config = config
        self.game_state = GameState(config=config)
        self.game_state.start_capture()
        self.played_beatmaps = []
        self.process_manager = ProcessManager()
        self.process_manager.bind_to_process()

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        
    def train_on_beatmap(self, beatmap_name: str):
        pass
            
    def run(self):
        while True:
            info = self.game_state.get_last_game_info()
            if info is None:
                time.sleep(1/10)
                continue
            if info.game_state == 5:
                self.process_manager.send_key("f2")

            time.sleep(1/10)