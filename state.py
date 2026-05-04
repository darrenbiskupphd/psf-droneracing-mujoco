from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class DroneState:
    position: np.ndarray      # [x, y, z]
    velocity: np.ndarray      # [vx, vy, vz]
    quaternion: np.ndarray    # [qw, qx, qy, qz]
    angular_rate: np.ndarray  # [wx, wy, wz]

    @property
    def euler(self):
        # SciPy expects [qx, qy, qz, qw]
        # MuJoCo is [qw, qx, qy, qz]
        q = [self.quaternion[1], self.quaternion[2], self.quaternion[3], self.quaternion[0]]
        r = Rotation.from_quat(q)
        
        yaw, pitch, roll = r.as_euler('zyx', degrees=False)
        return np.array([roll, pitch, yaw])
