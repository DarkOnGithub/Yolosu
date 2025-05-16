from .socket import Socket
import time
from dataclasses import dataclass
import queue
from typing import List, Dict
from model.reinforcement_learning.config import RL_Config
from utils.time_queue import TimeQueue
WEBSOCKET_URL = "ws://localhost:24050/websocket/v2/precise"

@dataclass
class GameInfo:
    current_time: int
    epoch: int
    three_hundred: int
    one_hundred: int
    fifty: int
    miss: int
    slider_break: int
    score: int
    accuracy: float
    unstable_rate: float
    max_combo: int
    current_combo: int
    title: str
    difficulty: str
    checksum: str
    hit_errors: List[float]
    game_state: int
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GameInfo':
        return cls(
            current_time=data['currentTime'],
            epoch=data['epoch'],
            three_hundred=data['three_hundred'],
            one_hundred=data['one_hundred'],
            fifty=data['fifty'],
            miss=data['miss'],
            slider_break=data['slider_break'],
            score=data['score'],
            accuracy=data['accuracy'],
            unstable_rate=data['unstableRate'],
            max_combo=data['max_combo'],
            current_combo=data['current_combo'],
            title=data['title'],
            difficulty=data['difficulty'],
            checksum=data['checksum'],
            hit_errors=data['hitErrors'],
            game_state=data['gameState']
        )


class GameState:
    def __init__(self, config: RL_Config):
        self.socket_receiver = Socket(WEBSOCKET_URL, self.on_message_callback)
        self.socket_receiver.connect()
        self.info_queue = TimeQueue()
        self.class_to_id = {name: idx for idx, name in enumerate(config.classes)}

        self.accuracy = 0
    

    def on_message_callback(self, message):
        info_socket = GameInfo.from_dict(message)
        self.info_queue.add(info_socket, timestamp=info_socket.epoch)

