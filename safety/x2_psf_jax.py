import os
import numpy as np

USE_JAX_GPU = False

if not USE_JAX_GPU:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from state import DroneState

# Optional: Force JAX to use 64-bit floats for solver precision stability
jax.config.update("jax_enable_x64", True)

def build_jax_functions(mass: float, J_mat: jnp.ndarray, J_inv: jnp.ndarray, M_torque: jnp.ndarray, 
                        N: int, dt: float, box_min: jnp.ndarray, box_max: jnp.ndarray, 
                        use_rk4: bool = False):
    """
    Factory function that creates and JIT-compiles the exact physics 
    and gradients required by the SLSQP solver.
    """
    g = 9.81
    
    def dynamics(x, u):
        p, v, theta, omega = x[0:3], x[3:6], x[6:9], x[9:12]
        phi, th, psi = theta[0], theta[1], theta[2]
        
        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_th, s_th = jnp.cos(th), jnp.sin(th)
        c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
        
        # Z-column of R(Theta)
        R_z = jnp.array([
            c_psi * s_th * c_phi + s_psi * s_phi,
            s_psi * s_th * c_phi - c_psi * s_phi,
            c_th * c_phi
        ])
        
        T = jnp.sum(u)
        v_dot = jnp.array([0.0, 0.0, -g]) + (T / mass) * R_z
        
        W = jnp.array([
            [1.0, s_phi * jnp.tan(th), c_phi * jnp.tan(th)],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi / c_th, c_phi / c_th]
        ])
        theta_dot = W @ omega
        
        tau = M_torque @ u
        omega_dot = J_inv @ (tau - jnp.cross(omega, J_mat @ omega))
        
        return jnp.concatenate([v, v_dot, theta_dot, omega_dot])

    def euler_step(x, u):
        return x + dt * dynamics(x, u)

    def rk4_step(x, u):
        k1 = dynamics(x, u)
        k2 = dynamics(x + 0.5 * dt * k1, u)
        k3 = dynamics(x + 0.5 * dt * k2, u)
        k4 = dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Alias the chosen integrator
    step_fn = rk4_step if use_rk4 else euler_step

    def rollout_constraints(U_flat, x0):
        """
        Unrolls the trajectory and returns the constraint margins.
        Returns array of shape (2 * 3 * N,) -> (min_margins, max_margins)
        """
        U = U_flat.reshape((N, 4))
        x = x0
        
        obs_centers = jnp.array([[0.0, -1.0, 2.0], [0.0, 2.0, 2.5], [4.0, -2.0, 3.0]])
        obs_radii = jnp.array([1.0, 1.0, 1.0])

        def scan_body(carry_x, u_k):
            next_x = step_fn(carry_x, u_k)
            # We only need the position (first 3 elements) for the box constraints
            p = next_x[0:3]
            margin_min = p - box_min
            margin_max = box_max - p

            # Obstacle constraints: (distance from center)^2 - radius^2 >= 0
            obs_margins = jnp.sum((p - obs_centers)**2, axis=1) - (obs_radii**2)
            
            margins = jnp.concatenate([margin_min, margin_max, obs_margins])
            return next_x, margins
        
        # jax.lax.scan is a highly optimized JAX loop
        _, margins_seq = jax.lax.scan(scan_body, x, U)
        
        # Flatten the margins for SciPy
        return margins_seq.flatten()

    def rollout_states(U_flat, x0):
        U = U_flat.reshape((N, 4))
        def scan_body(carry_x, u_k):
            next_x = step_fn(carry_x, u_k)
            return next_x, next_x
        _, states = jax.lax.scan(scan_body, x0, U)
        return states

    def cost(U_flat, u_nom):
        U = U_flat.reshape((N, 4))
        # Broadcast u_nom across the horizon
        u_nom_seq = jnp.tile(u_nom, (N, 1))
        # Simple L2 tracking cost
        return jnp.sum((U - u_nom_seq)**2)

    # We use jacfwd (forward mode) because the input (4N) and output (6N) sizes are similar.
    jit_cost = jax.jit(cost)
    jit_grad = jax.jit(jax.grad(cost, argnums=0))
    jit_cons = jax.jit(rollout_constraints)
    jit_cons_jac = jax.jit(jax.jacfwd(rollout_constraints, argnums=0))
    jit_rollout = jax.jit(rollout_states)

    return jit_cost, jit_grad, jit_cons, jit_cons_jac, jit_rollout


class PredictiveSafetyFilter:
    def __init__(self, mass: float, J: np.ndarray, M: np.ndarray, 
                 horizon: int = 10, dt: float = 0.05, use_rk4: bool = False):
        
        self.N = horizon
        self.u_min = 0.0
        self.u_max = 13.0
        
        # SciPy boundaries
        self.bounds = [(self.u_min, self.u_max) for _ in range(self.N * 4)]
        self.U_prev = np.zeros(self.N * 4)

        # Prevents high-speed discretization tunneling through the walls or roof
        box_min = jnp.array([-9.6, -4.6, 0.1])
        box_max = jnp.array([ 9.6,  4.6, 4.8])
        J_mat = jnp.array(J)
        J_inv = jnp.linalg.inv(J_mat)
        M_torque = jnp.array(M[1:4, :])
        
        # Generate and cache the compiled JAX functions
        self._cost, self._grad, self._cons, self._cons_jac, self._rollout = build_jax_functions(
            mass, J_mat, J_inv, M_torque, self.N, dt, box_min, box_max, use_rk4=use_rk4
        )

        self._warmup_compilation()

    def _warmup_compilation(self) -> None:
        U0 = jnp.zeros((self.N * 4,), dtype=jnp.float64)
        x0 = jnp.zeros((12,), dtype=jnp.float64)
        u_nom = jnp.full((4,), 0.5 * (self.u_min + self.u_max), dtype=jnp.float64)

        self._cost(U0, u_nom).block_until_ready()
        self._grad(U0, u_nom).block_until_ready()
        self._cons(U0, x0).block_until_ready()
        self._cons_jac(U0, x0).block_until_ready()

    def solve(self, state: DroneState, u_nom: np.ndarray) -> np.ndarray:
        x0 = np.concatenate([state.position, state.velocity, state.euler, state.angular_rate])
        
        j_x0 = jnp.array(x0)
        j_unom = jnp.array(u_nom)
        
        U_guess = np.roll(self.U_prev, -4)
        U_guess[-4:] = u_nom
        
        # SciPy wrapper functions to handle JAX -> NumPy memory conversion
        def cost_wrapper(U): return float(self._cost(U, j_unom))
        def grad_wrapper(U): return np.asarray(self._grad(U, j_unom)).astype(np.float64)
        def cons_wrapper(U): return np.asarray(self._cons(U, j_x0)).astype(np.float64)
        def cons_jac_wrapper(U): return np.asarray(self._cons_jac(U, j_x0)).astype(np.float64)

        cons = {
            'type': 'ineq', 
            'fun': cons_wrapper, 
            'jac': cons_jac_wrapper # Explicit analytical Jacobian passed to SLSQP
        }
        
        res = minimize(
            cost_wrapper, 
            U_guess, 
            method='SLSQP', 
            jac=grad_wrapper, # Explicit analytical Gradient
            bounds=self.bounds, 
            constraints=cons,
            options={'maxiter': 15, 'ftol': 1e-3, 'disp': False}
        )
        
        if res.success or res.status == 9:
            self.U_prev = res.x
            
        return self.U_prev[:4]

    def get_trajectory(self, state: DroneState) -> np.ndarray:
        x0 = np.concatenate([state.position, state.velocity, state.euler, state.angular_rate])
        # Returns shape (N, 12), we only need first 3 elements for position
        traj = self._rollout(self.U_prev, jnp.array(x0))
        return np.asarray(traj)[:, :3]
