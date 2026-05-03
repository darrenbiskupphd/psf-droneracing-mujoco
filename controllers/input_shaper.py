import numpy as np

class InputShaper:
    def __init__(self, mass: float = 1.325, max_roll_pitch: float = np.deg2rad(15), max_yaw_rate: float = 0.75):
        self.mass = mass
        self.g = 9.81
        
        self.max_roll_pitch = max_roll_pitch
        self.max_yaw_rate = max_yaw_rate
        self.max_vz = 2.0  # m/s target vertical velocity
        
        # Desired states
        self.vz_des = 0.0
        self.phi_des = 0.0
        self.theta_des = 0.0
        self.psi_dot_des = 0.0

        # Input states
        self.keys = {}
        
    def process_inputs(self, dt: float):
        """Bang-Bang control targeting pure states based on keys"""
        # Vertical Velocity (Z) logic
        up = self.keys.get('space', False)
        down = self.keys.get('shift', False)
        if up and not down:
            self.vz_des = self.max_vz
        elif down and not up:
            self.vz_des = -self.max_vz
        else:
            self.vz_des = 0.0

        w = self.keys.get('w', False)
        s = self.keys.get('s', False)
        if w and not s:
            self.theta_des = -self.max_roll_pitch # Nose down to go forward (+X axis is back)
        elif s and not w:
            self.theta_des = self.max_roll_pitch
        else:
            self.theta_des = 0.0

        a = self.keys.get('a', False)
        d = self.keys.get('d', False)
        if a and not d:
            self.phi_des = self.max_roll_pitch # Roll left
        elif d and not a:
            self.phi_des = -self.max_roll_pitch  # Roll right
        else:
            self.phi_des = 0.0

        # Yaw rate (Q/E)
        q = self.keys.get('q', False)
        e = self.keys.get('e', False)
        if q and not e:
            self.psi_dot_des = self.max_yaw_rate
        elif e and not q:
            self.psi_dot_des = -self.max_yaw_rate
        else:
            self.psi_dot_des = 0.0

    def get_desired_state(self):
        """Returns [vz_des, phi_des, theta_des, psi_dot_des]"""
        return np.array([self.vz_des, self.phi_des, self.theta_des, self.psi_dot_des])