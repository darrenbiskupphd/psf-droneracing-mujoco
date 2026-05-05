import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

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
    model = mujoco.MjModel.from_xml_path('x2_race.xml')
    data = mujoco.MjData(model)

    # Initialize PD controller and Input Shaper
    controller = PDController()
    
    total_mass = 1.325 # derived this sum (0.325 ellipsoid + 4 * 0.25 rotors) from xml
    input_shaper = InputShaper(mass=total_mass)

    J = np.diag(model.body_inertia[1])  # Extract real inertia vector from MuJoCo
    M = np.linalg.inv(controller.M_inv) # Recover the forward mixer matrix
    psf = PredictiveSafetyFilter(mass=total_mass, J=J, M=M, horizon=10, dt=0.03, use_rk4=True)

    # Initialize Tkinter Control Panel
    gui = DroneGUI(input_shaper)

    # Simulation loop setup
    physics_dt = model.opt.timestep # configured to 0.001 in xml
    control_dt = 1/100             # 100 Hz Control Rate
    steps_per_control = int(control_dt // physics_dt)
    
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hover")
    if keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        data.ctrl[:] = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance += 5
        
        # Precompute site ids for the prediction dots
        pred_site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"pred{i}") for i in range(10)]
        goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        
        # Logging for plotting
        t_history = []
        u_nom_history = []
        u_filt_history = []
        sim_start_time = time.time()
        
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
                traj = psf.get_trajectory(current_state)
                for i, site_id in enumerate(pred_site_ids):
                    model.site_pos[site_id] = traj[i]
            else:
                motor_commands = u_nom
                for site_id in pred_site_ids:
                    model.site_pos[site_id] = [0, 0, -10] # Hide them below the floor
                
            data.ctrl[:] = motor_commands
            
            # Log data
            if input_shaper.psf_enabled:
                t_history.append(time.time() - sim_start_time)
                u_nom_history.append(u_nom.copy())
                u_filt_history.append(motor_commands.copy())
            
            # Camera chasing if enabled
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

    # Plot results after simulation closes
    if len(t_history) > 0:
        t_history = np.array(t_history)
        u_nom_history = np.array(u_nom_history)
        u_filt_history = np.array(u_filt_history)
        
        plt.figure(figsize=(10, 8))
        for i in range(4):
            plt.subplot(4, 1, i+1)
            plt.plot(t_history, u_nom_history[:, i], label='Commanded (u_nom)', linestyle='--')
            plt.plot(t_history, u_filt_history[:, i], label='Filtered (u_filt)', alpha=0.7)
            plt.ylabel(f'Motor {i+1} Thrust')
            plt.grid(True)
            if i == 0:
                plt.legend()
        plt.xlabel('Time (s)')
        plt.suptitle('Commanded vs. Filtered Motor Inputs')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
