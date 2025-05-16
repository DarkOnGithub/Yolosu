from dataclasses import dataclass, field

@dataclass
class RL_Config:
    classes: list[str]
    device_idx: int = 0
    weight_path: str = r"runs\detect\train6\weights\best.engine"
    yolo_config: dict = field(default_factory=lambda: {
        "conf": 0.25,
        "iou": 0.5,
        "tracker": "bytetrack.yaml",
        "show": False,
        "imgsz": 416,
    })
    
