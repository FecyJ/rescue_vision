from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True, slots=True)
class CameraFrame:
    sequence: int
    timestamp: float   # ns
    image_bgr: np.ndarray