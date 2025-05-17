from model.reinforcement_learning.config import RL_Config
from model.reinforcement_learning.game_state import GameState
from model.reinforcement_learning.tracker import Tracker
import utils
from enum import Enum

class SessionState(Enum):
    TRACKING = 0
    
class Session:
    def __init__(self, config: RL_Config, game_state: GameState):
        self.config = config
        self.game_state = game_state
        
    def capture_frames(self):
        output_objects = []
        tracker = Tracker(config=self.config, output_objects=output_objects)
        while True:
            if not utils.get_active_window_name() == "osu!" or self.game_state.last_game_info.game_state != 1:
                tracker.pause()
            else:
                tracker.resume()

    def start_session(self):
        pass
        
