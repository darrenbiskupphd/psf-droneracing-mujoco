import time
import mujoco
import mujoco.viewer
import numpy as np

from state import DroneState
from controllers.pd_controller import PDController
from controllers.input_shaper import InputShaper
from controllers.gui import DroneGUI
from safety.x2_psf_jax import PredictiveSafetyFilter

def get_drone_state(model, data) -> DroneState:
    # Use qpos offsets for freejoint: 3 for pos, 4 for quat
    # qvel offsets: 3 for lin_vel, 3 for ang_vel
    return DroneState(
        position=data.qpos[:3].copy(),
        quaternion=data.qpos[3:7].copy(),
        velocity=data.qvel[:3].copy(),
        angular_rate=data.qvel[3:6].copy()
    )

def main():
    # Load the model from the modified XML
    model = mujoco.MjModel.from_xml_path('x2.xml')
    data = mujoco.MjData(model)

    # Initialize PD controller and Input Shaper
    controller = PDController()
    
    total_mass = 1.325 # We derived this sum (0.325 ellipsoid + 4 * 0.25 rotors) from xml
    input_shaper = InputShaper(mass=total_mass)

    # Initialize Predictive Safety Filter (JAX-Free variant)
    J = np.diag(model.body_inertia[1])  # Extract real inertia vector from MuJoCo
    M = np.linalg.inv(controller.M_inv) # Recover the forward mixer matrix
    psf = PredictiveSafetyFilter(mass=total_mass, J=J, M=M, horizon=10, dt=0.03, use_rk4=True)

    # Initialize Tkinter Control Panel
    gui = DroneGUI(input_shaper)

    # Simulation loop setup
    physics_dt = model.opt.timestep # configured to 0.001 in xml
    control_dt = 1/60             # 100 Hz Control Rate
    steps_per_control = int(control_dt // physics_dt)
    
    # Optional logic for setting the initial state from keyframe
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hover")
    if keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        data.ctrl[:] = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance += 5
        
        while viewer.is_running() and gui.is_running():
            step_start = time.time()
            
            # --- 100 Hz CONTROL LOOP ---
            gui.update()
            input_shaper.process_inputs(control_dt)
            desired_state = input_shaper.get_desired_state()

            current_state = get_drone_state(model, data)
            u_nom = controller.compute_control(current_state, desired_state)
            
            # Apply predictive safety filter if enabled by the user
            if input_shaper.psf_enabled:
                motor_commands = psf.solve(current_state, u_nom)
            else:
                motor_commands = u_nom
                
            data.ctrl[:] = motor_commands
            
            if input_shaper.chase_cam_active:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = current_state.position
                viewer.cam.azimuth = np.degrees(current_state.euler[2]) + 180
                viewer.cam.elevation = -20
                viewer.cam.distance = 3.0

            # --- 1000 Hz PHYSICS LOOP ---
            for _ in range(steps_per_control):
                mujoco.mj_step(model, data)
            
            viewer.sync()

            # Real-time synchronization
            time_until_next_step = control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
