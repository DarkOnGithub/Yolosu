from .socket import Socket
import time
from dataclasses import dataclass
from typing import List, Dict
import queue
import time

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
            hit_errors=data['hitErrors']
        )

class GameState:
    def __init__(self):       
        self.socket_receiver = Socket(WEBSOCKET_URL, self.on_message_callback)
        self.socket_receiver.connect()
        self.last_packet_epoch = int(time.time() * 1000)

        self.socket_queue = queue.Queue(maxsize=10)
        
        while True:
            time.sleep(1)

    def on_message_callback(self, message):
        info_socket = GameInfo.from_dict(message)
        print(info_socket.epoch, info_socket.accuracy, "time between packet", f"{info_socket.epoch - self.last_packet_epoch}ms", end="\r")
        self.last_packet_epoch = info_socket.epoch
        print(len(str(info_socket)))
    
