import numpy as np
from controllers.state import DroneState

class PDController:
    def __init__(self):
        # PD Gains for attitude, P for yaw and vertical velocity
        self.kp_roll, self.kd_roll = 5.0, 1.0
        self.kp_pitch, self.kd_pitch = 5.0, 1.0
        self.kp_yaw = 0.5
        
        self.kp_vz = 5.0  # P gain for vertical velocity
        self.mass = 1.325
        self.gravity = 9.81

        # Mixer matrix constructed from motor positions based on the xml
        # Rotor 1: thrust1 pos="-.14 -.18 .05" gear="... -.0201"
        # Rotor 2: thrust2 pos="-.14  .18 .05" gear="...  .0201"
        # Rotor 3: thrust3 pos=" .14  .18 .08" gear="... -.0201"
        # Rotor 4: thrust4 pos=" .14 -.18 .08" gear="...  .0201"
        r_motors = np.array([
            [-0.14, -0.18, 0],
            [-0.14,  0.18, 0],
            [ 0.14,  0.18, 0],
            [ 0.14, -0.18, 0]
        ])
        c_motors = np.array([-0.0201, 0.0201, -0.0201, 0.0201])
        
        M = np.zeros((4, 4))
        for i in range(4):
            M[0, i] = 1.0                     # thrust   = sum(u)
            M[1, i] = r_motors[i, 1] * 1.0    # tau_x = sum(y * Fz)
            M[2, i] = -r_motors[i, 0] * 1.0   # tau_y = sum(-x * Fz)
            M[3, i] = c_motors[i]             # tau_z = sum(c * u)
            
        self.M_inv = np.linalg.inv(M)

    def compute_control(self, state: DroneState, desired_state: np.ndarray):
        """
        Takes in current DroneState and desired vector [vz, phi, theta, psi_dot].
        Returns motor commands [u1, u2, u3, u4].
        """
        vz_des, phi_des, theta_des, psi_dot_des = desired_state

        phi, theta, psi = state.euler
        omega_x, omega_y, omega_z = state.angular_rate
        vz = state.velocity[2]

        # Z-axis velocity control -> commanded thrust
        thrust = self.mass * self.gravity + self.kp_vz * (vz_des - vz)

        # PD attitude control
        tau_x = self.kp_roll * (phi_des - phi) + self.kd_roll * (0.0 - omega_x)
        tau_y = self.kp_pitch * (theta_des - theta) + self.kd_pitch * (0.0 - omega_y)
        tau_z = self.kp_yaw * (psi_dot_des - omega_z)

        V = np.array([thrust, tau_x, tau_y, tau_z])
        U = self.M_inv @ V

        # Clamp against the ctrlrange defined in x2.xml (0 to 13)
        U = np.clip(U, 0, 13.0)
        return U