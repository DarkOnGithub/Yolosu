from .socket import Socket
import time
from dataclasses import dataclass
from typing import List, Dict
import queue

WEBSOCKET_URL = "ws://localhost:24050/websocket/v2"
WEBSOCKET_PRECISE_URL = "ws://localhost:24050/websocket/v2/precise"

@dataclass
class Hits:
    three_hundred: int = 0
    one_hundred: int = 0
    fifty: int = 0
    miss: int = 0
    slider_break: int = 0

    def __repr__(self):
        return f"Hits(300={self.three_hundred}, 100={self.one_hundred}, 50={self.fifty}, miss={self.miss}, slider_break={self.slider_break})"

@dataclass
class Scores:
    score: int = 0
    pp: int = 0
    
    def __repr__(self):
        return f"Scores(score={self.score}, pp={self.pp})"
    
@dataclass
class Precisions:
    precision: float = 0.0
    accuracy: float = 0.0
    unstableRate: float = 0.0
    max_combo: int = 0
    current_combo: int = 0
    current_score: int = 0
    
    def __repr__(self):
        return f"Precisions(precision={self.precision}, accuracy={self.accuracy}, unstableRate={self.unstableRate}, max_combo={self.max_combo}, current_combo={self.current_combo}, current_score={self.current_score})"
@dataclass
class Beatmap:
    title: str = ""
    difficulty: str = ""
    checksum: str = ""
    time: int = 0
    first_object_time: int = 0
    last_object_time: int = 0

    def __repr__(self):
        return f"Beatmap(title={self.title}, difficulty={self.difficulty}, checksum={self.checksum}, time={self.time}, first_object_time={self.first_object_time}, last_object_time={self.last_object_time})"
    
@dataclass
class InfoSocket:
    hits: Hits
    scores: Scores
    precisions: Precisions
    beatmap: Beatmap

class GameState:
    def __init__(self):
        # self.socket = Socket(WEBSOCKET_URL, self.callback_message)
        # self.socket.connect()
        
        self.precise_socket = Socket(WEBSOCKET_PRECISE_URL, self.callback_precise_message)
        self.precise_socket.connect()
        
        self.socket_queue = queue.Queue(maxsize=10)
        
        while True:
            time.sleep(1)

    def callback_message(self, message):
        print(message["beatmap"]["time"])
        if 'play' in message:
            play_data = message['play']
            hits = Hits(
                three_hundred=play_data['hits'].get('300', 0),
                one_hundred=play_data['hits'].get('100', 0),
                fifty=play_data['hits'].get('50', 0),
                miss=play_data['hits'].get('0', 0),
                slider_break=play_data['hits'].get('sliderBreaks', 0)
            )

            scores = Scores(
                score=play_data.get('score', 0),
            )

            precisions = Precisions(
                precision=play_data.get('accuracy', 0.0),
                accuracy=play_data.get('accuracy', 0.0),
                unstableRate=play_data.get('unstableRate', 0.0),
                max_combo=play_data.get('combo', {}).get('max', 0),
                current_combo=play_data.get('combo', {}).get('current', 0),
                current_score=play_data.get('score', 0)
            )

        if 'beatmap' in message:
            beatmap_data = message['beatmap']
            beatmap = Beatmap(
                title=beatmap_data.get('title', ''),
                difficulty=beatmap_data.get('version', ''),
                checksum=beatmap_data.get('checksum', ''),
                time=beatmap_data.get('time', {}).get('live', 0),
                first_object_time=beatmap_data.get('time', {}).get('firstObject', 0),
                last_object_time=beatmap_data.get('time', {}).get('lastObject', 0)
            )
        info_socket = InfoSocket(hits, scores, precisions, beatmap)
        self.socket_queue.put(info_socket)
    
    def callback_precise_message(self, message):
        print(message)
    
