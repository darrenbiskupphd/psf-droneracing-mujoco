# Incremental Implementation Plan: MuJoCo Quadrotor + PSF

## Objective
Build a continuous-time quadrotor simulation in MuJoCo to test a Predictive Safety Filter (PSF). Implementation will be strictly phased. Do not implement downstream phases until the underlying control theory has been derived and validated.

---

## Phase 1: Scaffolding and Physics
Establish the plant and the visualizer.

1.  **MJCF (XML) Setup:**
    * Define the global bounding box (translucent, `rgba` alpha < 1.0).
    * Define the quadrotor (`freejoint`) with 4 spatial actuators (motors).
    * **Actuation Model:** Use abstract spatial force/torque vectors via `site` and `gear` properties. Do not simulate physical spinning blades. Couple Z-thrust with reactive Z-torque using a coefficient $c$:
    ```xml
    <actuator>
        <motor name="m1" site="s1" gear="0 0 1 0 0  0.05"/> 
        <motor name="m2" site="s2" gear="0 0 1 0 0 -0.05"/> 
    </actuator>
    ```
    * Attach a fixed tracking camera (Fortnite view) to the drone body.
2.  **Simulation Loop:**
    * Initialize `mujoco.viewer.launch_passive`.
    * Run `mj_step()` at a fixed $dt$.
    * Verify the drone falls and collides with the box floor, and verify camera switching via the viewer.

---

## Phase 2: State Estimation, Human Input, and Inner Loop
Decouple the controller from the plant, shape inputs, and compute motor commands.

1.  **State Estimator Architecture:**
    * Implement a strict data contract (`DroneState`) to decouple the controller from MuJoCo's ground truth, enabling future EKF integration without refactoring.
    ```python
    @dataclass
    class DroneState:
        position: np.ndarray      # [x, y, z]
        velocity: np.ndarray      # [vx, vy, vz]
        quaternion: np.ndarray    # [qw, qx, qy, qz]
        angular_rate: np.ndarray  # [wx, wy, wz]
    ```

2.  **Input Capture (Separate GUI):**
    * Create a small, separate UI window popping up over the MuJoCo viewer (e.g., via `glfw`, `pygame`, or `tkinter`) featuring virtual buttons or a joystick.
    * When this window is focused, it captures the keyboard inputs (Q, W, E, A, S, D, Space, Shift) to prevent conflicts with MuJoCo's native viewer keybindings (which use these keys for actions like wireframes, hide objects, camera indicators, etc.).

3.  **Setpoint Translator:**
    * **Thrust ($T_{des}$):** Baseline hover ($m \cdot g$). Spacebar integrates thrust upward ($+\Delta T$); Shift integrates downward ($-\Delta T$) to saturation limits. No key applies first-order decay back to $m \cdot g$.
    * **Attitude ($\phi_{des}, \theta_{des}$):** WASD integrates target angles to a saturation limit (e.g., $\pm 30^\circ$). No key applies rapid exponential decay to $0^\circ$ (auto-leveling).
    * **Yaw Rate ($\dot{\psi}_{des}$):** Q/E sets a fixed scalar target (e.g., $\pm \pi/2$ rad/s). No key resets immediately to $0$ rad/s.

4.  **Inner Controller (PD):**
    * Compute body torques required to track the shaped inputs. Extract Euler angles from the `DroneState` quaternion.
    * **Roll ($\tau_x$) & Pitch ($\tau_y$):** PD control on angle errors.
    * **Yaw ($\tau_z$):** P control on angular rate error.
    * Output vector: $V = [T_{des}, \tau_x, \tau_y, \tau_z]^T$.

5.  **Actuator Mixer:**
    * Map desired body forces $V$ to motor inputs $U = [u_1, u_2, u_3, u_4]^T$.
    * Given arm length $L$ and torque coefficient $c$, construct the mixer matrix $M$:
    $$
    \begin{bmatrix} T_{des} \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix} = 
    \begin{bmatrix} 
    1 & 1 & 1 & 1 \\
    L & L & -L & -L \\
    -L & L & L & -L \\
    c & -c & c & -c 
    \end{bmatrix}
    \begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ u_4 \end{bmatrix}
    $$
    * **Implementation:** Pre-compute $M^{-1}$ at initialization. Solve $U = M^{-1} V$ continuously. Clamp $U \ge 0$ before writing to `d.ctrl[:]`.

---

## Phase 3: The Predictive Safety Filter (Math TBD)
Intercept the nominal commands to prevent constraint violations.

1.  **Functional Requirements (Ready):**
    * Must be toggleable via a hotkey (e.g., `F`).
    * When off: Drone can crash into the box walls.
    * When on: Filter overrides user input to prevent boundary penetration.
2.  **State and Dynamics Definitions:**
    
    **State Variables:**
    
    | Variable | Description | Reference Frame | MuJoCo Extraction (`data.`) |
    |----------|-------------|-----------------|---------------------------|
    | $p$ | Position (X, Y, Z) | World | `qpos[0:3]` |
    | $v$ | Linear Velocity (X, Y, Z) | World | `qvel[0:3]` |
    | $\Theta$ | Euler Angles ($\phi, \theta, \psi$) | World (intrinsic Z-Y-X) | `qpos[3:7]` (requires quaternion conversion via scipy) |
    | $\omega$ | Angular Velocity ($\omega_x, \omega_y, \omega_z$) | Local Body | `qvel[3:6]` |
    
    **System Dynamics:**
    
    $$
    \begin{bmatrix}
    \dot{p} \\
    \dot{v} \\
    \dot{\Theta} \\
    \dot{\omega}
    \end{bmatrix} = 
    \begin{bmatrix}
    v \\
    \begin{bmatrix} 0 \\ 0 \\ -g \end{bmatrix} + \frac{1}{m} R(\Theta) \begin{bmatrix} 0 \\ 0 \\ \sum u_i \end{bmatrix} \\
    W(\Theta) \omega \\
    J^{-1} (M_{2:4} U - \omega \times (J \omega))
    \end{bmatrix}
    $$
    
    where $W(\Theta) = T(\Theta)^{-1}$ is the rotation rate transformation matrix:
    
    $$
    W(\Theta) = 
    \begin{bmatrix}
    1 & \sin \phi \tan \theta & \cos \phi \tan \theta \\
    0 & \cos \phi & -\sin \phi \\
    0 & \frac{\sin \phi}{\cos \theta} & \frac{\cos \phi}{\cos \theta}
    \end{bmatrix}
    $$
    
    The rotation matrix $R(\Theta)$ for intrinsic Z-Y-X Euler angles ($\psi, \theta, \phi$) is:
    
    $$
    R(\Theta) = \begin{bmatrix}
    \cos\psi \cos\theta & \sin\phi \sin\theta \cos\psi - \sin\psi \cos\phi & \sin\phi \sin\psi + \sin\theta \cos\phi \cos\psi \\
    \sin\psi \cos\theta & \sin\phi \sin\psi \sin\theta + \cos\phi \cos\psi & -\sin\phi \cos\psi + \sin\psi \sin\theta \cos\phi \\
    -\sin\theta & \sin\phi \cos\theta & \cos\phi \cos\theta
    \end{bmatrix}
    $$


---

## Phase 4: Race Track and Waypoints (Ready to Implement, Low Priority)
Visual coaching layer. 

1.  **Pre-allocation:** Define tubular gates in the XML (`contype="0"`, highly transparent).
2.  **State Machine:** Track target gate via $L_2$ norm distance.
3.  **Visual Feedback:** Mutate the active gate's `rgba` properties in real-time to guide the user. No dynamic mesh generation.