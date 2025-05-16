from model.reinforcement_learning.config import RL_Config
from model.reinforcement_learning.tracker import Tracker
import time

class Inference:
    def __init__(self, config: RL_Config):
        self.config = config
        self.tracker = Tracker(config=config)
        
    def run(self):
        while True:
            time.sleep(0.001)
