"""
KOS Epistemic Drive -- Fristonian Free Energy Minimization Engine

Pillar 1 of the Genesis Engine: The "Desire to Grow"

A machine without desire sits idle. The Epistemic Drive gives KOS a
mathematical mandate to minimize Surprise (Prediction Error). When its
predictions fail, it registers Mathematical Entropy and is physically
forced to launch Swarms, spawn hypotheses, and write new modules until
the prediction error reaches 0.0.

The Drive runs as a continuous background thread, accumulating:
- Curiosity: grows during idle time, triggers exploration
- Frustration: grows on task failure, triggers self-repair
- Compression: grows on redundant solutions, triggers consolidation
- Surprise: spikes on new input, triggers MCTS hypothesis generation

Usage:
    drive = EpistemicDrive()
    drive.start()  # Launches background thread
    drive.inject_stimulus(task_data)  # External input = sensory disruption
    drive.stop()   # Graceful shutdown
"""

import os
import json
import time
import threading
import math
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque


@dataclass
class DriveState:
    """The organism's emotional/motivational state."""
    # Core Fristonian tensions (all in [0.0, infinity))
    curiosity: float = 0.0       # Grows during idle time
    frustration: float = 0.0     # Grows on task failure
    compression: float = 0.0     # Drive to simplify/consolidate
    surprise: float = 0.0        # Spikes on new input
    entropy: float = 0.0         # Global prediction error

    # Metabolic state
    energy: float = 100.0        # Depleted by computation
    temperature: float = 1.0     # Exploration vs exploitation
    cycles: int = 0              # Total heartbeat cycles

    # Performance tracking
    tasks_attempted: int = 0
    tasks_solved: int = 0
    dream_successes: int = 0
    last_solve_time: float = 0.0

    def free_energy(self) -> float:
        """Friston's Free Energy: the quantity the organism MUST minimize."""
        return (self.surprise + self.entropy +
                self.curiosity * 0.5 +
                self.frustration * 0.3)

    def to_dict(self) -> dict:
        return {
            'curiosity': round(self.curiosity, 4),
            'frustration': round(self.frustration, 4),
            'compression': round(self.compression, 4),
            'surprise': round(self.surprise, 4),
            'entropy': round(self.entropy, 4),
            'energy': round(self.energy, 2),
            'temperature': round(self.temperature, 4),
            'free_energy': round(self.free_energy(), 4),
            'cycles': self.cycles,
            'tasks_attempted': self.tasks_attempted,
            'tasks_solved': self.tasks_solved,
            'solve_rate': round(self.tasks_solved / max(self.tasks_attempted, 1), 4),
        }


class EpistemicDrive:
    """
    The Fristonian Drive Engine -- gives KOS autonomous motivation.

    Runs a continuous background loop that:
    1. Accumulates curiosity during idle time
    2. Decays frustration and surprise over time
    3. Triggers actions when tensions exceed thresholds
    4. Maintains energy budget (prevents runaway computation)
    """

    # Tension thresholds that trigger autonomous actions
    CURIOSITY_DREAM_THRESHOLD = 2.0    # Triggers dreaming
    FRUSTRATION_REPAIR_THRESHOLD = 3.0  # Triggers self-repair
    COMPRESSION_CONSOLIDATE = 5.0       # Triggers REM Sleep
    SURPRISE_INVESTIGATE = 1.5          # Triggers hypothesis generation

    # Decay rates (per second)
    CURIOSITY_GROWTH = 0.02     # Idle curiosity accumulation
    FRUSTRATION_DECAY = 0.005   # Natural frustration cooling
    SURPRISE_DECAY = 0.05       # Surprise fades quickly
    ENERGY_REGEN = 0.5          # Energy regeneration per second

    def __init__(self):
        self.state = DriveState()
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Action callbacks (wired by the organism)
        self._on_dream: Optional[Callable] = None
        self._on_repair: Optional[Callable] = None
        self._on_consolidate: Optional[Callable] = None
        self._on_investigate: Optional[Callable] = None

        # Event log (circular buffer)
        self._event_log = deque(maxlen=1000)

        # Prediction model (simple: track solve rate trends)
        self._recent_results = deque(maxlen=50)

        # Persistence
        self._state_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "drive_state.json"
        )
        self._load_state()

    def _load_state(self):
        """Load persisted drive state."""
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path) as f:
                    data = json.load(f)
                for key, val in data.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, val)
        except Exception:
            pass

    def _save_state(self):
        """Persist drive state to disk."""
        try:
            with open(self._state_path, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception:
            pass

    # ================================================================
    # EXTERNAL INTERFACE
    # ================================================================

    def start(self, tick_rate: float = 1.0):
        """Start the background drive loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._drive_loop, args=(tick_rate,), daemon=True)
        self._thread.start()
        self._log("DRIVE_START", "Epistemic Drive engaged")

    def stop(self):
        """Stop the drive loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._save_state()
        self._log("DRIVE_STOP", "Epistemic Drive disengaged")

    def inject_stimulus(self, stimulus_type: str, intensity: float = 1.0,
                         metadata: dict = None):
        """
        External sensory input -- disrupts equilibrium.

        stimulus_type: "new_task", "task_solved", "task_failed",
                       "dream_success", "dream_failure", "new_data"
        """
        with self._lock:
            if stimulus_type == "new_task":
                self.state.surprise += intensity * 2.0
                self.state.curiosity = max(0, self.state.curiosity - 0.5)
                self.state.tasks_attempted += 1
                self.state.energy -= 5.0

            elif stimulus_type == "task_solved":
                self.state.surprise = max(0, self.state.surprise - 1.0)
                self.state.frustration = max(0, self.state.frustration - 0.5)
                self.state.entropy = max(0, self.state.entropy - 0.2)
                self.state.tasks_solved += 1
                self.state.last_solve_time = time.time()
                self._recent_results.append(1)

            elif stimulus_type == "task_failed":
                self.state.frustration += intensity * 0.5
                self.state.entropy += 0.1
                self._recent_results.append(0)

            elif stimulus_type == "dream_success":
                self.state.compression += 0.3
                self.state.curiosity = max(0, self.state.curiosity - 1.0)
                self.state.dream_successes += 1

            elif stimulus_type == "dream_failure":
                self.state.frustration += 0.2
                self.state.entropy += 0.05

            elif stimulus_type == "new_data":
                self.state.surprise += intensity
                self.state.curiosity += 0.3

        self._log(f"STIMULUS_{stimulus_type.upper()}",
                  f"intensity={intensity}", metadata)

    def wire_actions(self, on_dream=None, on_repair=None,
                     on_consolidate=None, on_investigate=None):
        """Wire callback functions for autonomous actions."""
        self._on_dream = on_dream
        self._on_repair = on_repair
        self._on_consolidate = on_consolidate
        self._on_investigate = on_investigate

    def get_state(self) -> dict:
        """Get current drive state as dict."""
        with self._lock:
            return self.state.to_dict()

    def get_temperature(self) -> float:
        """Get exploration temperature for MCTS/swarm decisions."""
        with self._lock:
            return self.state.temperature

    def get_event_log(self, n: int = 50) -> List[dict]:
        """Get recent events."""
        return list(self._event_log)[-n:]

    # ================================================================
    # THE HEARTBEAT
    # ================================================================

    def _drive_loop(self, tick_rate: float):
        """
        The continuous subconscious heartbeat.

        Every tick:
        1. Update tensions (curiosity grows, frustration decays)
        2. Compute free energy
        3. Trigger autonomous actions when thresholds exceeded
        4. Update temperature (exploration vs exploitation)
        """
        while self._running:
            t0 = time.time()

            with self._lock:
                self.state.cycles += 1

                # ---- TENSION DYNAMICS ----

                # Curiosity grows when idle (no recent stimuli)
                idle_time = time.time() - self.state.last_solve_time if self.state.last_solve_time > 0 else 0
                if idle_time > 10:
                    self.state.curiosity += self.CURIOSITY_GROWTH * tick_rate

                # Frustration decays naturally (cooling)
                self.state.frustration = max(
                    0, self.state.frustration - self.FRUSTRATION_DECAY * tick_rate)

                # Surprise decays fast
                self.state.surprise = max(
                    0, self.state.surprise - self.SURPRISE_DECAY * tick_rate)

                # Energy regenerates
                self.state.energy = min(
                    100.0, self.state.energy + self.ENERGY_REGEN * tick_rate)

                # ---- PREDICTION ERROR (Entropy) ----
                # Compute based on recent solve rate vs expectation
                if len(self._recent_results) >= 5:
                    recent_rate = sum(self._recent_results) / len(self._recent_results)
                    # Entropy = -log(solve_rate) = high when we're failing
                    self.state.entropy = max(0, -math.log(max(recent_rate, 0.01)))

                # ---- TEMPERATURE (Exploration/Exploitation) ----
                # High frustration → high temperature (explore more)
                # Low entropy → low temperature (exploit what works)
                self.state.temperature = 0.5 + (
                    self.state.frustration * 0.3 +
                    self.state.curiosity * 0.1 -
                    min(self.state.tasks_solved * 0.01, 0.3)
                )
                self.state.temperature = max(0.1, min(3.0, self.state.temperature))

                # ---- FREE ENERGY ----
                fe = self.state.free_energy()

                # ---- AUTONOMOUS ACTIONS ----
                # Only trigger if we have energy
                if self.state.energy > 10:

                    # Curiosity → Dream (explore unsolved tasks)
                    if (self.state.curiosity > self.CURIOSITY_DREAM_THRESHOLD
                            and self._on_dream):
                        self._log("ACTION_DREAM",
                                  f"Curiosity={self.state.curiosity:.2f} "
                                  f"triggered dreaming")
                        self.state.curiosity *= 0.5  # Partially satisfy
                        self.state.energy -= 20.0
                        try:
                            self._on_dream()
                        except Exception as e:
                            self._log("DREAM_ERROR", str(e))

                    # Frustration → Self-repair
                    elif (self.state.frustration > self.FRUSTRATION_REPAIR_THRESHOLD
                          and self._on_repair):
                        self._log("ACTION_REPAIR",
                                  f"Frustration={self.state.frustration:.2f} "
                                  f"triggered self-repair")
                        self.state.frustration *= 0.5
                        self.state.energy -= 15.0
                        try:
                            self._on_repair()
                        except Exception as e:
                            self._log("REPAIR_ERROR", str(e))

                    # Compression → Consolidate (REM Sleep)
                    elif (self.state.compression > self.COMPRESSION_CONSOLIDATE
                          and self._on_consolidate):
                        self._log("ACTION_CONSOLIDATE",
                                  f"Compression={self.state.compression:.2f} "
                                  f"triggered consolidation")
                        self.state.compression *= 0.3
                        self.state.energy -= 10.0
                        try:
                            self._on_consolidate()
                        except Exception as e:
                            self._log("CONSOLIDATE_ERROR", str(e))

                # Periodic state save
                if self.state.cycles % 60 == 0:
                    self._save_state()

            # Maintain tick rate
            elapsed = time.time() - t0
            sleep_time = max(0, tick_rate - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _log(self, event_type: str, message: str, metadata: dict = None):
        """Log an event."""
        entry = {
            'time': time.time(),
            'cycle': self.state.cycles,
            'type': event_type,
            'message': message,
            'free_energy': round(self.state.free_energy(), 4),
        }
        if metadata:
            entry['metadata'] = metadata
        self._event_log.append(entry)
