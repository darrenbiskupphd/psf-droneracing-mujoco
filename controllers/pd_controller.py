import numpy as np
from state import DroneState

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
        vz_cmd, phi_cmd, theta_cmd, psi_dot_cmd = desired_state

        phi_global, theta_global, psi_global = state.euler
        omega_x_local, omega_y_local, omega_z_local = state.angular_rate
        vz_global = state.velocity[2]

        # Transform global observed angles to local body frame
        c_psi = np.cos(psi_global)
        s_psi = np.sin(psi_global)
        
        phi_obs_local = phi_global * c_psi + theta_global * s_psi
        theta_obs_local = -phi_global * s_psi + theta_global * c_psi

        # Transform global velocity to local frame and implement synthetic braking
        vx_global, vy_global, vz_global = state.velocity
        vx_local = vx_global * c_psi + vy_global * s_psi
        vy_local = -vx_global * s_psi + vy_global * c_psi
        k_brake = -0.15 # Tuning parameter: radians of tilt per m/s of drift
        
        if phi_cmd == 0.0:
            # If drifting left (+vy_local), roll right (-phi) to brake
            phi_cmd = -k_brake * vy_local 
            
        if theta_cmd == 0.0:
            # If drifting forward (+vx_local), pitch up (+theta) to brake
            theta_cmd = k_brake * vx_local 

        # Compute errors strictly in the local frame
        err_phi = phi_cmd - phi_obs_local
        err_theta = theta_cmd - theta_obs_local
        
        # Z-axis velocity control
        thrust = self.mass * self.gravity + self.kp_vz * (vz_cmd - vz_global)

        # PD attitude control (Local Proportional Error - Local Derivative)
        tau_x = self.kp_roll * err_phi + self.kd_roll * (0.0 - omega_x_local)
        tau_y = self.kp_pitch * err_theta + self.kd_pitch * (0.0 - omega_y_local)
        tau_z = self.kp_yaw * (psi_dot_cmd - omega_z_local)

        V = np.array([thrust, tau_x, tau_y, tau_z])
        U = self.M_inv @ V

        # Clamp against the ctrlrange defined in x2.xml (0 to 13)
        U = np.clip(U, 0, 13.0)
        return U