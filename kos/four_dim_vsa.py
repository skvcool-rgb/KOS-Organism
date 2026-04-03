"""
KOS 4D Spatiotemporal VSA — Continuous Time via Fourier Phase Shifts

Standard HDC/VSA uses integer rolls (np.roll(vec, 17)) for discrete spatial steps.
But time and space are continuous. To represent video, trajectories, and motion,
we need fractional shifts (e.g., 0.003 seconds, 2.7 pixels).

Mathematical Breakthrough: In the frequency domain (FFT), shifting a vector
by a fractional amount is just a phase rotation:

    shifted = IFFT( FFT(vec) * exp(-2j*pi*freq*shift/dim) )

This gives us exact continuous interpolation in 10,000-D hyperspace.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict


class ContinuousVSA:
    """
    Hyperdimensional Physics Engine that operates in continuous 4D Spacetime.
    Uses Fractional Phase Shifting to encode continuous velocity and time.
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.memory: Dict[str, np.ndarray] = {}
        # Pre-calculate the frequency array for rapid Fourier phase shifts
        self.frequencies = np.arange(self.dim, dtype=np.float64)
        # Cache FFT plans for common operations
        self._phase_cache: Dict[float, np.ndarray] = {}

    def create_concept(self, name: str) -> np.ndarray:
        """Create a bipolar random hypervector for a base concept."""
        vec = np.random.choice([-1.0, 1.0], size=self.dim)
        self.memory[name] = vec
        return vec

    def get_or_create(self, name: str) -> np.ndarray:
        """Get existing concept or create new one."""
        if name not in self.memory:
            self.create_concept(name)
        return self.memory[name]

    def shift_continuous(self, vec: np.ndarray, shift_amount: float) -> np.ndarray:
        """
        Fractional Binding: Moves an object through continuous space and time.

        shift_amount can be 0.001 (a microsecond) or 50.4 (a spatial leap).
        Uses FFT phase rotation for exact continuous interpolation.

        Math:
            shifted = IFFT( FFT(vec) * exp(-2j * pi * freq * shift / dim) )
        """
        # Use numpy's FFT (scipy not required)
        vec_fft = np.fft.fft(vec)

        # Compute phase shift (cache for repeated shifts)
        cache_key = round(shift_amount, 10)
        if cache_key not in self._phase_cache:
            self._phase_cache[cache_key] = np.exp(
                -2j * np.pi * self.frequencies * shift_amount / self.dim
            )
        phase_shift = self._phase_cache[cache_key]

        # Apply phase rotation and return to physical hyperspace
        shifted_vec = np.real(np.fft.ifft(vec_fft * phase_shift))

        # Bipolar threshold to preserve VSA algebraic properties
        return np.where(shifted_vec >= 0, 1.0, -1.0)

    def shift_continuous_soft(self, vec: np.ndarray, shift_amount: float) -> np.ndarray:
        """
        Soft fractional shift — returns continuous values (no thresholding).
        Useful for intermediate computations where precision matters.
        """
        vec_fft = np.fft.fft(vec)
        phase_shift = np.exp(
            -2j * np.pi * self.frequencies * shift_amount / self.dim
        )
        return np.real(np.fft.ifft(vec_fft * phase_shift))

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiplication (bipolar binding)."""
        return a * b

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Superposition with majority threshold."""
        total = np.sum(vectors, axis=0)
        return np.where(total >= 0, 1.0, -1.0)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two hypervectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    # ================================================================
    # 4D SPACETIME ENCODING
    # ================================================================

    def encode_position_4d(self, x: float, y: float, z: float, t: float) -> np.ndarray:
        """
        Encode a continuous 4D spacetime coordinate as a hypervector.

        Each dimension gets its own base vector, shifted by the continuous
        coordinate value. The 4D position is their binding (intersection).

        Position(x,y,z,t) = shift(X_base, x) * shift(Y_base, y) * shift(Z_base, z) * shift(T_base, t)
        """
        x_vec = self.shift_continuous(self.get_or_create("_DIM_X"), x)
        y_vec = self.shift_continuous(self.get_or_create("_DIM_Y"), y)
        z_vec = self.shift_continuous(self.get_or_create("_DIM_Z"), z)
        t_vec = self.shift_continuous(self.get_or_create("_DIM_T"), t)
        return self.bind(self.bind(x_vec, y_vec), self.bind(z_vec, t_vec))

    def encode_trajectory(self, object_vec: np.ndarray, velocity: float,
                          frames: int, dt: float = 0.1) -> np.ndarray:
        """
        Encode a moving object as a single trajectory manifold.

        Takes a static object vector and smears it across the time dimension,
        creating a single hypervector that represents the entire motion.

        Args:
            object_vec: The static object's identity vector
            velocity: Speed (spatial units per second)
            frames: Number of time steps
            dt: Time step size (seconds)

        Returns:
            A single hypervector encoding the full trajectory
        """
        trajectory_manifold = np.zeros(self.dim)

        for t in range(frames):
            # Position = Velocity * Time
            current_position = velocity * (t * dt)
            # Bind object identity to its position at time t
            time_vec = self.shift_continuous(self.get_or_create("_DIM_T"), t * dt)
            pos_vec = self.shift_continuous(object_vec, current_position)
            state_in_time = self.bind(pos_vec, time_vec)
            trajectory_manifold += state_in_time

        return np.where(trajectory_manifold >= 0, 1.0, -1.0)

    def encode_trajectory_2d(self, object_vec: np.ndarray,
                              vx: float, vy: float,
                              frames: int, dt: float = 0.1,
                              gravity: float = 0.0) -> np.ndarray:
        """
        Encode a 2D trajectory with optional gravity.

        Simulates parabolic motion: x(t) = vx*t, y(t) = vy*t + 0.5*g*t^2
        Each frame is bound to its time stamp and bundled.
        """
        trajectory = np.zeros(self.dim)

        for i in range(frames):
            t = i * dt
            x = vx * t
            y = vy * t + 0.5 * gravity * t * t

            # Encode spatial position at this time
            x_vec = self.shift_continuous(self.get_or_create("_DIM_X"), x)
            y_vec = self.shift_continuous(self.get_or_create("_DIM_Y"), y)
            t_vec = self.shift_continuous(self.get_or_create("_DIM_T"), t)

            frame_vec = self.bind(self.bind(object_vec, self.bind(x_vec, y_vec)), t_vec)
            trajectory += frame_vec

        return np.where(trajectory >= 0, 1.0, -1.0)

    def detect_velocity(self, trajectory: np.ndarray, object_vec: np.ndarray,
                        dt: float = 0.1, max_frames: int = 20,
                        velocity_range: Tuple[float, float] = (-10.0, 10.0),
                        resolution: float = 0.5) -> Optional[float]:
        """
        Given a trajectory manifold and the object's identity, recover the velocity.

        Scans candidate velocities and finds which one produces the highest
        resonance with the trajectory.
        """
        best_sim = -1.0
        best_vel = None

        v = velocity_range[0]
        while v <= velocity_range[1]:
            candidate = self.encode_trajectory(object_vec, v, max_frames, dt)
            sim = self.similarity(trajectory, candidate)
            if sim > best_sim:
                best_sim = sim
                best_vel = v
            v += resolution

        # Refine with finer resolution around best
        if best_vel is not None:
            v = best_vel - resolution
            while v <= best_vel + resolution:
                candidate = self.encode_trajectory(object_vec, v, max_frames, dt)
                sim = self.similarity(trajectory, candidate)
                if sim > best_sim:
                    best_sim = sim
                    best_vel = v
                v += resolution / 10

        return best_vel if best_sim > 0.1 else None

    def predict_future(self, trajectory: np.ndarray, object_vec: np.ndarray,
                       velocity: float, current_time: float,
                       future_time: float) -> np.ndarray:
        """
        Given a trajectory and known velocity, predict the object's state
        at a future time point.

        Returns the predicted position-bound object vector.
        """
        future_pos = velocity * future_time
        pos_vec = self.shift_continuous(object_vec, future_pos)
        t_vec = self.shift_continuous(self.get_or_create("_DIM_T"), future_time)
        return self.bind(pos_vec, t_vec)

    # ================================================================
    # PHYSICS DISCOVERY
    # ================================================================

    def discover_law(self, observations: List[Tuple[float, float]],
                     object_vec: np.ndarray) -> Dict:
        """
        Given (time, position) observations, discover the governing equation.

        Tries: constant velocity, constant acceleration, and static.
        Returns the best-fit law as a dict.
        """
        if len(observations) < 2:
            return {"law": "static", "params": {}}

        times = [o[0] for o in observations]
        positions = [o[1] for o in observations]

        # Fit linear: pos = v*t + p0
        dt_vals = [times[i+1] - times[i] for i in range(len(times)-1)]
        dp_vals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

        if all(abs(dt) > 1e-10 for dt in dt_vals):
            velocities = [dp / dt for dp, dt in zip(dp_vals, dt_vals)]
            mean_v = sum(velocities) / len(velocities)
            v_variance = sum((v - mean_v)**2 for v in velocities) / len(velocities)

            if v_variance < 0.01:
                # Constant velocity
                return {
                    "law": "constant_velocity",
                    "params": {"velocity": mean_v, "initial_position": positions[0]},
                    "equation": f"x(t) = {mean_v:.3f} * t + {positions[0]:.3f}"
                }

            # Try acceleration: v changes linearly
            if len(velocities) >= 2:
                dv_vals = [velocities[i+1] - velocities[i] for i in range(len(velocities)-1)]
                dt_v = [dt_vals[i] for i in range(len(dv_vals))]
                if all(abs(dt) > 1e-10 for dt in dt_v):
                    accels = [dv / dt for dv, dt in zip(dv_vals, dt_v)]
                    mean_a = sum(accels) / len(accels)
                    a_variance = sum((a - mean_a)**2 for a in accels) / len(accels)

                    if a_variance < 0.1:
                        return {
                            "law": "constant_acceleration",
                            "params": {
                                "acceleration": mean_a,
                                "initial_velocity": velocities[0],
                                "initial_position": positions[0]
                            },
                            "equation": f"x(t) = 0.5 * {mean_a:.3f} * t^2 + {velocities[0]:.3f} * t + {positions[0]:.3f}"
                        }

        return {"law": "unknown", "params": {}}
