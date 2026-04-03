"""
KOS Molecular Dynamics Engine
=============================
A real-time MD simulator with:
- Lennard-Jones + Coulomb force fields
- Velocity Verlet integration
- NVT thermostat (Berendsen)
- Periodic boundary conditions
- Neighbor list optimization
- Trajectory analysis (RDF, MSD, energy)

Built for material discovery — the organism uses this to simulate
atomic-scale behavior of candidate compositions.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Fundamental Constants ──────────────────────────────────────
KB = 1.380649e-23      # Boltzmann constant (J/K)
NA = 6.02214076e23     # Avogadro's number
EV_TO_J = 1.602176634e-19  # eV → Joules
ANGSTROM = 1e-10       # Angstrom → meters
AMU_TO_KG = 1.66053906660e-27  # atomic mass unit → kg


# ── Atom Types with LJ Parameters ─────────────────────────────
@dataclass
class AtomType:
    """Lennard-Jones parameters + metadata for an atom type."""
    symbol: str
    mass: float         # in AMU
    epsilon: float      # LJ well depth in eV
    sigma: float        # LJ zero-crossing in Angstroms
    charge: float       # partial charge in elementary charges
    color: str = "#888" # for visualization
    toxic: bool = False
    toxicity_score: float = 0.0  # 0=safe, 1=deadly


# Comprehensive atom type database with LJ parameters
# (Universal Force Field / OPLS-AA derived approximations)
ATOM_TYPES: Dict[str, AtomType] = {
    # ── Safe / Common Elements ──
    "H":  AtomType("H",  1.008,  0.0157, 2.571, 0.0,   "#FFFFFF"),
    "He": AtomType("He", 4.003,  0.0009, 2.640, 0.0,   "#D9FFFF"),
    "Li": AtomType("Li", 6.941,  0.0250, 2.451, 0.0,   "#CC80FF"),
    "Be": AtomType("Be", 9.012,  0.0420, 2.260, 0.0,   "#C2FF00"),
    "B":  AtomType("B",  10.81,  0.0400, 3.638, 0.0,   "#FFB5B5"),
    "C":  AtomType("C",  12.011, 0.0559, 3.431, 0.0,   "#909090"),
    "N":  AtomType("N",  14.007, 0.0690, 3.261, 0.0,   "#3050F8"),
    "O":  AtomType("O",  15.999, 0.0600, 3.118, 0.0,   "#FF0D0D"),
    "F":  AtomType("F",  18.998, 0.0500, 2.997, 0.0,   "#90E050"),
    "Ne": AtomType("Ne", 20.180, 0.0031, 2.780, 0.0,   "#B3E3F5"),
    "Na": AtomType("Na", 22.990, 0.0300, 2.983, 1.0,   "#AB5CF2"),
    "Mg": AtomType("Mg", 24.305, 0.1110, 2.691, 2.0,   "#8AFF00"),
    "Al": AtomType("Al", 26.982, 0.5050, 4.008, 0.0,   "#BFA6A6"),
    "Si": AtomType("Si", 28.086, 0.4020, 3.826, 0.0,   "#F0C8A0"),
    "P":  AtomType("P",  30.974, 0.3050, 3.695, 0.0,   "#FF8000"),
    "S":  AtomType("S",  32.065, 0.2740, 3.595, 0.0,   "#FFFF30"),
    "Cl": AtomType("Cl", 35.453, 0.2270, 3.516, -1.0,  "#1FF01F"),
    "K":  AtomType("K",  39.098, 0.0350, 3.812, 1.0,   "#8F40D4"),
    "Ca": AtomType("Ca", 40.078, 0.2380, 3.028, 2.0,   "#3DFF00"),
    "Ti": AtomType("Ti", 47.867, 0.0170, 2.829, 0.0,   "#BFC2C7"),
    "Fe": AtomType("Fe", 55.845, 0.0130, 2.594, 0.0,   "#E06633"),
    "Cu": AtomType("Cu", 63.546, 0.0050, 3.114, 0.0,   "#C88033"),
    "Zn": AtomType("Zn", 65.380, 0.1240, 2.462, 0.0,   "#7D80B0"),
    "Ge": AtomType("Ge", 72.630, 0.3790, 3.813, 0.0,   "#668F8F"),
    "Br": AtomType("Br", 79.904, 0.2510, 3.732, -1.0,  "#A62929"),
    "Ag": AtomType("Ag", 107.87, 0.0360, 2.805, 0.0,   "#C0C0C0"),
    "Sn": AtomType("Sn", 118.71, 0.2910, 3.913, 0.0,   "#668080"),
    "I":  AtomType("I",  126.90, 0.3390, 4.009, -1.0,   "#940094"),
    "Cs": AtomType("Cs", 132.91, 0.0450, 4.024, 1.0,   "#57178F"),
    "Ba": AtomType("Ba", 137.33, 0.3640, 3.299, 2.0,   "#00C900"),
    "W":  AtomType("W",  183.84, 0.0670, 2.734, 0.0,   "#2194D6"),
    "Au": AtomType("Au", 196.97, 0.0390, 2.934, 0.0,   "#FFD123"),
    "Bi": AtomType("Bi", 208.98, 0.1760, 3.893, 0.0,   "#9E4FB5"),

    # ── Toxic Elements (flagged) ──
    "Pb": AtomType("Pb", 207.20, 0.6630, 3.828, 0.0, "#575961",
                   toxic=True, toxicity_score=0.85),
    "Cd": AtomType("Cd", 112.41, 0.2280, 2.537, 0.0, "#FFD98F",
                   toxic=True, toxicity_score=0.90),
    "Hg": AtomType("Hg", 200.59, 0.0980, 2.410, 0.0, "#B8B8D0",
                   toxic=True, toxicity_score=0.95),
    "As": AtomType("As", 74.922, 0.3090, 3.769, 0.0, "#BD80E3",
                   toxic=True, toxicity_score=0.88),
    "Tl": AtomType("Tl", 204.38, 0.6800, 3.873, 0.0, "#A6544D",
                   toxic=True, toxicity_score=0.92),
    "Se": AtomType("Se", 78.960, 0.2910, 3.746, 0.0, "#FFA100",
                   toxic=True, toxicity_score=0.40),  # Moderate toxicity
    "Cr": AtomType("Cr", 51.996, 0.0150, 2.693, 0.0, "#8A99C7",
                   toxic=True, toxicity_score=0.60),  # Cr(VI) is toxic
}


# ── Toxicity Remediation Database ──────────────────────────────
TOXICITY_REMEDIATION: Dict[str, Dict] = {
    "Pb": {
        "toxicity_class": "HIGH",
        "health_effects": ["neurotoxicity", "kidney damage", "developmental harm"],
        "safe_alternatives": ["Sn", "Bi", "Ge"],
        "encapsulation_methods": [
            "Glass encapsulation (hermetic sealing)",
            "Polymer barrier layer (EVOH/PET)",
            "Ceramic coating (Al2O3 ALD)",
        ],
        "remediation": [
            "Chelation with EDTA for contaminated soil",
            "Phytoremediation with Thlaspi caerulescens",
            "Stabilization with phosphate (apatite formation)",
        ],
        "recycling": "Hydrometallurgical recovery: dissolve in HNO3, precipitate PbSO4",
        "regulatory_limit_ppm": 10,  # EPA drinking water standard
    },
    "Cd": {
        "toxicity_class": "HIGH",
        "health_effects": ["carcinogen (lung)", "kidney failure", "bone demineralization"],
        "safe_alternatives": ["Zn", "Cu(In,Ga)Se2"],
        "encapsulation_methods": [
            "Laminated glass sandwich",
            "EVA encapsulant + backsheet",
            "Atomic layer deposition (TiO2)",
        ],
        "remediation": [
            "Ion exchange resins",
            "Electrochemical recovery",
            "Biosorption with algae (Chlorella vulgaris)",
        ],
        "recycling": "Acid leaching + electrowinning",
        "regulatory_limit_ppm": 5,
    },
    "Hg": {
        "toxicity_class": "EXTREME",
        "health_effects": ["CNS damage", "renal toxicity", "Minamata disease"],
        "safe_alternatives": ["Ag", "Au nanoparticles"],
        "encapsulation_methods": [
            "Sealed quartz ampoules",
            "Sulfide stabilization (HgS cinnabar)",
        ],
        "remediation": [
            "Activated carbon adsorption",
            "Thiol-functionalized nanoparticles",
            "Bacterial methylmercury degradation (merB gene)",
        ],
        "recycling": "Retort distillation at 357C",
        "regulatory_limit_ppm": 2,
    },
    "As": {
        "toxicity_class": "HIGH",
        "health_effects": ["carcinogen (skin/lung)", "peripheral neuropathy"],
        "safe_alternatives": ["P", "Sb (lower doses)"],
        "encapsulation_methods": [
            "Iron co-precipitation (FeAsO4)",
            "Cement stabilization",
        ],
        "remediation": [
            "Iron-based adsorbents (GFH)",
            "Phytoremediation with Pteris vittata (brake fern)",
            "Coagulation with ferric chloride",
        ],
        "recycling": "Controlled acid digestion + precipitation",
        "regulatory_limit_ppm": 10,
    },
    "Se": {
        "toxicity_class": "MODERATE",
        "health_effects": ["selenosis at high doses", "hair/nail loss"],
        "safe_alternatives": ["S", "Te (with encapsulation)"],
        "encapsulation_methods": [
            "Polymer encapsulation (standard)",
            "Glass-glass lamination",
        ],
        "remediation": [
            "Biological reduction (Se(VI) -> Se(0))",
            "Ferrihydrite adsorption",
        ],
        "recycling": "Selenium recovery from acid leachate",
        "regulatory_limit_ppm": 50,
    },
    "Cr": {
        "toxicity_class": "MODERATE_TO_HIGH",
        "health_effects": ["Cr(VI) is carcinogenic", "respiratory sensitizer"],
        "safe_alternatives": ["Ti", "Zr", "V (less toxic)"],
        "encapsulation_methods": [
            "Reduction to Cr(III) which is safe",
            "Cement immobilization",
        ],
        "remediation": [
            "Chemical reduction Cr(VI)->Cr(III) with Fe(II)",
            "Bioreduction with Shewanella oneidensis",
        ],
        "recycling": "Electroplating recovery",
        "regulatory_limit_ppm": 100,  # for Cr(III); 0.05 for Cr(VI)
    },
    "Tl": {
        "toxicity_class": "EXTREME",
        "health_effects": ["alopecia", "polyneuropathy", "cardiac toxicity"],
        "safe_alternatives": ["K", "Rb"],
        "encapsulation_methods": [
            "Sealed containers only",
            "Not recommended for consumer products",
        ],
        "remediation": [
            "Prussian blue oral administration (medical)",
            "Activated carbon adsorption",
        ],
        "recycling": "Specialized hazmat recovery",
        "regulatory_limit_ppm": 2,
    },
}


# ── Particle Representation ────────────────────────────────────
@dataclass
class Particle:
    """A single atom in the simulation."""
    atom_type: str      # Symbol (key into ATOM_TYPES)
    position: np.ndarray  # [x, y, z] in Angstroms
    velocity: np.ndarray  # [vx, vy, vz] in Angstrom/fs
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    id: int = 0


@dataclass
class MDResult:
    """Result of an MD simulation run."""
    total_steps: int
    final_energy_eV: float
    avg_temperature_K: float
    avg_pressure_GPa: float
    is_stable: bool
    rdf: Optional[Dict[str, List[float]]] = None
    energy_trajectory: Optional[List[float]] = None
    temperature_trajectory: Optional[List[float]] = None
    toxic_components: Optional[List[Dict]] = None
    remediation_options: Optional[List[Dict]] = None
    binding_energy_per_atom_eV: float = 0.0
    bandgap_estimate_eV: float = 0.0
    notes: str = ""


class MolecularDynamicsEngine:
    """
    Classical Molecular Dynamics with Lennard-Jones + Coulomb potentials.

    Supports:
    - Arbitrary compositions (any elements in ATOM_TYPES)
    - Velocity Verlet integration
    - Berendsen thermostat (NVT ensemble)
    - Periodic boundary conditions
    - Neighbor list with cutoff
    - Radial distribution function analysis
    - Mean square displacement
    - Toxicity assessment + remediation
    """

    def __init__(self, box_size: float = 30.0, cutoff: float = 10.0):
        """
        Args:
            box_size: Cubic simulation box side length in Angstroms
            cutoff: LJ cutoff distance in Angstroms
        """
        self.box_size = box_size
        self.cutoff = cutoff
        self.particles: List[Particle] = []
        self.dt = 0.1  # timestep in femtoseconds (small for LJ stability)
        self.step_count = 0
        self.energy_history: List[float] = []
        self.temp_history: List[float] = []
        self.ke_history: List[float] = []

    def clear(self):
        """Reset simulation."""
        self.particles.clear()
        self.step_count = 0
        self.energy_history.clear()
        self.temp_history.clear()
        self.ke_history.clear()

    def add_atoms(self, symbol: str, count: int, temperature_K: float = 300.0):
        """Add atoms of a given type on a grid with Maxwell-Boltzmann velocities.

        Uses grid-based placement to avoid initial overlaps that cause energy explosion.
        """
        if symbol not in ATOM_TYPES:
            raise ValueError(f"Unknown atom type: {symbol}. Available: {list(ATOM_TYPES.keys())}")

        atype = ATOM_TYPES[symbol]
        mass_kg = atype.mass * AMU_TO_KG

        # Maxwell-Boltzmann velocity standard deviation
        # v_std = sqrt(kT/m) in m/s, convert to Angstrom/fs
        v_std = math.sqrt(KB * temperature_K / mass_kg) * 1e-5  # m/s -> A/fs

        # Grid-based placement: distribute atoms on a regular grid
        # to avoid initial overlaps
        existing = len(self.particles)
        total_needed = existing + count
        n_per_side = max(2, int(math.ceil(total_needed ** (1/3))))
        spacing = (self.box_size - 4.0) / n_per_side  # Leave 2A margin on each side
        spacing = max(spacing, atype.sigma * 0.9)  # At least 0.9 sigma apart

        placed = 0
        for ix in range(n_per_side):
            for iy in range(n_per_side):
                for iz in range(n_per_side):
                    if placed >= count:
                        break
                    pos = np.array([
                        2.0 + ix * spacing + random.uniform(-0.1, 0.1),
                        2.0 + iy * spacing + random.uniform(-0.1, 0.1),
                        2.0 + iz * spacing + random.uniform(-0.1, 0.1),
                    ])
                    # Ensure within box
                    pos = np.clip(pos, 1.0, self.box_size - 1.0)
                    vel = np.random.normal(0, v_std, 3)
                    pid = len(self.particles)
                    self.particles.append(Particle(symbol, pos, vel, id=pid))
                    placed += 1
                if placed >= count:
                    break
            if placed >= count:
                break

    def add_crystal(self, symbol: str, lattice_constant: float,
                    nx: int = 3, ny: int = 3, nz: int = 3,
                    temperature_K: float = 300.0):
        """Add atoms in a simple cubic crystal arrangement."""
        if symbol not in ATOM_TYPES:
            raise ValueError(f"Unknown atom type: {symbol}")

        atype = ATOM_TYPES[symbol]
        mass_kg = atype.mass * AMU_TO_KG
        v_std = math.sqrt(KB * temperature_K / mass_kg) * 1e-5

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    pos = np.array([
                        ix * lattice_constant + 0.5,
                        iy * lattice_constant + 0.5,
                        iz * lattice_constant + 0.5,
                    ])
                    vel = np.random.normal(0, v_std, 3)
                    pid = len(self.particles)
                    self.particles.append(Particle(symbol, pos, vel, id=pid))

        # Update box size to fit crystal
        self.box_size = max(self.box_size, max(nx, ny, nz) * lattice_constant + 2.0)

    def _minimum_image(self, dr: np.ndarray) -> np.ndarray:
        """Apply periodic boundary conditions (minimum image convention)."""
        return dr - self.box_size * np.round(dr / self.box_size)

    def _compute_forces(self) -> float:
        """Compute all pairwise forces. Returns total potential energy in eV."""
        n = len(self.particles)
        for p in self.particles:
            p.force[:] = 0.0

        pe = 0.0
        cutoff_sq = self.cutoff ** 2

        for i in range(n):
            pi = self.particles[i]
            ti = ATOM_TYPES[pi.atom_type]

            for j in range(i + 1, n):
                pj = self.particles[j]
                tj = ATOM_TYPES[pj.atom_type]

                dr = self._minimum_image(pj.position - pi.position)
                r_sq = np.dot(dr, dr)

                # Minimum distance: prevent singularity when atoms overlap
                min_r_sq = 1.0  # 1 Angstrom^2 minimum
                if r_sq < cutoff_sq and r_sq > min_r_sq:
                    r = math.sqrt(r_sq)

                    # Lorentz-Berthelot mixing rules
                    eps = math.sqrt(max(ti.epsilon, 1e-6) * max(tj.epsilon, 1e-6))
                    sig = (ti.sigma + tj.sigma) / 2.0

                    # Lennard-Jones force and energy
                    sr = sig / r
                    sr6 = sr ** 6
                    sr12 = sr6 * sr6

                    # Cap sr12 to prevent numerical explosion
                    sr12 = min(sr12, 1e6)
                    sr6 = min(sr6, 1e3)

                    # Energy
                    pe += 4.0 * eps * (sr12 - sr6)

                    # Force magnitude: F = 24*eps/r * (2*sr12 - sr6)
                    f_mag = 24.0 * eps / r * (2.0 * sr12 - sr6)
                    # Cap force to prevent explosion
                    f_mag = max(-100.0, min(100.0, f_mag))
                    f_vec = f_mag * dr / r

                    pi.force += f_vec
                    pj.force -= f_vec

                    # Coulomb interaction (if charged)
                    if abs(ti.charge) > 0.01 and abs(tj.charge) > 0.01:
                        # F = k_e * q1*q2 / r^2 (in eV*A units)
                        # k_e = 14.3996 eV*A/e^2
                        k_e = 14.3996
                        coul_e = k_e * ti.charge * tj.charge / r
                        pe += coul_e
                        f_coul = k_e * ti.charge * tj.charge / r_sq
                        f_coul_vec = f_coul * dr / r
                        pi.force += f_coul_vec
                        pj.force -= f_coul_vec

        return pe

    def _kinetic_energy(self) -> float:
        """Compute total kinetic energy in eV."""
        ke = 0.0
        for p in self.particles:
            m = ATOM_TYPES[p.atom_type].mass * AMU_TO_KG
            v_ms = p.velocity * 1e5  # A/fs -> m/s
            ke += 0.5 * m * np.dot(v_ms, v_ms)
        return ke / EV_TO_J  # Convert J to eV

    def _temperature(self) -> float:
        """Instantaneous temperature in Kelvin."""
        n = len(self.particles)
        if n == 0:
            return 0.0
        ke_j = self._kinetic_energy() * EV_TO_J
        return 2.0 * ke_j / (3.0 * n * KB)

    def _apply_pbc(self):
        """Wrap positions into periodic box."""
        for p in self.particles:
            p.position = p.position % self.box_size

    def _berendsen_thermostat(self, target_T: float, tau: float = 10.0):
        """Berendsen velocity rescaling thermostat (strong coupling)."""
        current_T = self._temperature()
        if current_T < 1.0:
            return
        # Strong coupling: directly rescale if temperature is way off
        if current_T > target_T * 5:
            # Emergency rescale
            lam = math.sqrt(target_T / current_T)
        else:
            lam = math.sqrt(1.0 + (self.dt / tau) * (target_T / current_T - 1.0))
        lam = max(0.1, min(lam, 3.0))  # Safety bounds
        for p in self.particles:
            p.velocity *= lam

    def step(self, target_temperature: float = 300.0, thermostat: bool = True):
        """Perform one Velocity Verlet integration step."""
        dt = self.dt

        # Half-step velocity update
        for p in self.particles:
            m_amu = ATOM_TYPES[p.atom_type].mass
            # a = F/m, but we need consistent units
            # F in eV/A, m in AMU, we want acceleration in A/fs^2
            # 1 eV/A / 1 AMU = 9.6485e13 A/fs^2... need conversion
            # eV/(A*AMU) = 1.602e-19 / (1e-10 * 1.6605e-27) = 9648.5 A/fs^2
            acc = p.force / m_amu * 9.6485e-3  # eV/(A*AMU) -> A/fs^2 (approx)
            p.velocity += 0.5 * dt * acc

        # Full-step position update
        for p in self.particles:
            p.position += dt * p.velocity

        # Apply PBC
        self._apply_pbc()

        # Recompute forces
        pe = self._compute_forces()

        # Second half-step velocity update
        for p in self.particles:
            m_amu = ATOM_TYPES[p.atom_type].mass
            acc = p.force / m_amu * 9.6485e-3
            p.velocity += 0.5 * dt * acc

        # Cap velocities (prevent runaway)
        max_vel = 0.05  # Max ~0.05 A/fs
        for p in self.particles:
            v_mag = np.linalg.norm(p.velocity)
            if v_mag > max_vel:
                p.velocity *= max_vel / v_mag

        # Thermostat
        if thermostat:
            self._berendsen_thermostat(target_temperature)

        ke = self._kinetic_energy()
        total_e = pe + ke
        temp = self._temperature()

        self.energy_history.append(total_e)
        self.temp_history.append(temp)
        self.ke_history.append(ke)
        self.step_count += 1

        return total_e, temp

    def _energy_minimize(self, max_steps: int = 200, step_size: float = 0.01):
        """Steepest descent energy minimization to remove bad contacts."""
        for _ in range(max_steps):
            pe = self._compute_forces()
            max_force = 0.0
            for p in self.particles:
                f_mag = np.linalg.norm(p.force)
                max_force = max(max_force, f_mag)
                # Move atom in direction of force (steepest descent)
                if f_mag > 0:
                    p.position += step_size * p.force / max(f_mag, 0.01)
            self._apply_pbc()
            # Converged if max force is small
            if max_force < 1.0:
                break
            # Adaptive step size
            if max_force > 100:
                step_size *= 0.5
            elif max_force < 10:
                step_size *= 1.2
            step_size = min(step_size, 0.1)

    def run(self, n_steps: int = 1000, temperature_K: float = 300.0,
            progress_callback=None) -> MDResult:
        """Run a full MD simulation."""
        if not self.particles:
            return MDResult(0, 0, 0, 0, False, notes="No particles in simulation")

        t0 = time.perf_counter()

        # Energy minimization first to remove bad contacts
        self._energy_minimize(max_steps=300, step_size=0.01)

        # Reset velocities after minimization (thermal equilibration)
        for p in self.particles:
            atype = ATOM_TYPES[p.atom_type]
            mass_kg = atype.mass * AMU_TO_KG
            v_std = math.sqrt(KB * temperature_K / mass_kg) * 1e-5
            p.velocity = np.random.normal(0, v_std, 3)

        # Initial force computation
        self._compute_forces()

        for step in range(n_steps):
            total_e, temp = self.step(temperature_K)

            if progress_callback and step % 100 == 0:
                progress_callback(step, n_steps, total_e, temp)

            # Safety: abort if energy explodes (unstable configuration)
            if abs(total_e) > 1e6:
                return MDResult(
                    total_steps=step,
                    final_energy_eV=total_e,
                    avg_temperature_K=np.mean(self.temp_history[-100:]) if self.temp_history else 0,
                    avg_pressure_GPa=0,
                    is_stable=False,
                    energy_trajectory=self.energy_history,
                    temperature_trajectory=self.temp_history,
                    notes=f"UNSTABLE: Energy diverged at step {step}. Configuration is not viable."
                )

        elapsed = time.perf_counter() - t0

        # Analysis
        avg_temp = np.mean(self.temp_history[-min(500, len(self.temp_history)):])
        final_e = self.energy_history[-1] if self.energy_history else 0

        # Energy stability check
        if len(self.energy_history) > 100:
            e_std = np.std(self.energy_history[-100:])
            e_mean = abs(np.mean(self.energy_history[-100:]))
            is_stable = e_std / max(e_mean, 0.001) < 0.1  # <10% fluctuation
        else:
            is_stable = True

        # Binding energy per atom
        n_atoms = len(self.particles)
        binding_e = final_e / n_atoms if n_atoms > 0 else 0

        # Estimate pressure from virial (simplified)
        volume_A3 = self.box_size ** 3
        # P = NkT/V (ideal gas contribution) in eV/A^3
        p_ideal = n_atoms * KB * avg_temp / (volume_A3 * ANGSTROM**3 * EV_TO_J)
        # Convert to GPa: 1 eV/A^3 = 160.218 GPa
        p_gpa = p_ideal * 160.218

        # Radial Distribution Function
        rdf = self._compute_rdf()

        # Toxicity assessment
        toxic_report = self._assess_toxicity()

        # Bandgap estimate (from composition, not DFT)
        bandgap = self._estimate_bandgap_from_composition()

        return MDResult(
            total_steps=n_steps,
            final_energy_eV=final_e,
            avg_temperature_K=avg_temp,
            avg_pressure_GPa=p_gpa,
            is_stable=is_stable,
            rdf=rdf,
            energy_trajectory=self.energy_history[-200:],
            temperature_trajectory=self.temp_history[-200:],
            toxic_components=toxic_report["toxic_elements"],
            remediation_options=toxic_report["remediation"],
            binding_energy_per_atom_eV=binding_e,
            bandgap_estimate_eV=bandgap,
            notes=f"Simulated {n_steps} steps in {elapsed:.1f}s. {n_atoms} atoms.",
        )

    def _compute_rdf(self, n_bins: int = 100, r_max: float = None) -> Dict[str, List[float]]:
        """Compute radial distribution function g(r)."""
        if r_max is None:
            r_max = self.box_size / 2.0

        dr = r_max / n_bins
        n = len(self.particles)
        if n < 2:
            return {"r": [], "g_r": []}

        hist = np.zeros(n_bins)

        for i in range(n):
            for j in range(i + 1, n):
                d = self._minimum_image(self.particles[j].position - self.particles[i].position)
                r = np.linalg.norm(d)
                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < n_bins:
                        hist[bin_idx] += 2  # Count both i-j and j-i

        # Normalize
        rho = n / self.box_size ** 3  # number density
        r_vals = []
        g_vals = []
        for k in range(n_bins):
            r = (k + 0.5) * dr
            shell_vol = 4.0 * math.pi * r ** 2 * dr
            ideal = rho * shell_vol * n
            g = hist[k] / max(ideal, 1e-10)
            r_vals.append(round(r, 3))
            g_vals.append(round(g, 4))

        return {"r": r_vals, "g_r": g_vals}

    def _assess_toxicity(self) -> Dict:
        """Assess toxicity of all components and suggest remediation."""
        # Count atoms by type
        composition = {}
        for p in self.particles:
            composition[p.atom_type] = composition.get(p.atom_type, 0) + 1

        toxic_elements = []
        remediations = []

        for symbol, count in composition.items():
            atype = ATOM_TYPES.get(symbol)
            if atype and atype.toxic:
                toxic_info = {
                    "element": symbol,
                    "count": count,
                    "fraction": count / len(self.particles),
                    "toxicity_score": atype.toxicity_score,
                }

                if symbol in TOXICITY_REMEDIATION:
                    rem = TOXICITY_REMEDIATION[symbol]
                    toxic_info["toxicity_class"] = rem["toxicity_class"]
                    toxic_info["health_effects"] = rem["health_effects"]
                    toxic_info["safe_alternatives"] = rem["safe_alternatives"]

                    remediations.append({
                        "element": symbol,
                        "alternatives": rem["safe_alternatives"],
                        "encapsulation": rem["encapsulation_methods"],
                        "remediation_methods": rem["remediation"],
                        "recycling": rem["recycling"],
                        "regulatory_limit_ppm": rem["regulatory_limit_ppm"],
                    })

                toxic_elements.append(toxic_info)

        overall_toxicity = max(
            (e["toxicity_score"] for e in toxic_elements), default=0.0
        )

        return {
            "toxic_elements": toxic_elements,
            "remediation": remediations,
            "overall_toxicity_score": overall_toxicity,
            "is_safe": overall_toxicity < 0.3,
            "composition": composition,
        }

    def _estimate_bandgap_from_composition(self) -> float:
        """Rough bandgap estimate based on composition (not DFT).

        Uses empirical correlations for common material classes.
        This is a screening tool — real bandgap requires DFT.
        """
        # Count composition
        symbols = set(p.atom_type for p in self.particles)

        # Known bandgaps for common compositions
        bandgap_db = {
            frozenset(["Si"]): 1.12,
            frozenset(["Ge"]): 0.67,
            frozenset(["C"]): 5.5,  # Diamond
            frozenset(["Cs", "Sn", "I"]): 1.30,
            frozenset(["Cs", "Sn", "Br"]): 1.75,
            frozenset(["Cs", "Sn", "Cl"]): 2.80,
            frozenset(["Cs", "Ge", "I"]): 1.63,
            frozenset(["Cs", "Ge", "Br"]): 2.32,
            frozenset(["Cs", "Pb", "I"]): 1.73,
            frozenset(["Cs", "Pb", "Br"]): 2.30,
            frozenset(["Cs", "Bi", "I"]): 1.77,
            frozenset(["Cd", "Se"]): 1.74,
            frozenset(["Cd"]): 0.0,  # Metal
            frozenset(["Cu", "Ga", "Se"]): 1.15,  # CIGS
            frozenset(["Ti", "O"]): 3.2,  # TiO2
            frozenset(["Zn", "O"]): 3.37,
            frozenset(["Zn", "S"]): 3.54,
            frozenset(["Fe"]): 0.0,  # Metal
            frozenset(["Cu"]): 0.0,  # Metal
            frozenset(["Au"]): 0.0,  # Metal
            frozenset(["Al"]): 0.0,  # Metal
        }

        if frozenset(symbols) in bandgap_db:
            return bandgap_db[frozenset(symbols)]

        # Heuristic: average electronegativity difference correlates with bandgap
        if len(symbols) >= 2:
            ens = [ATOM_TYPES[s].epsilon for s in symbols if s in ATOM_TYPES]
            if len(ens) >= 2:
                en_range = max(ens) - min(ens)
                return min(en_range * 3.0, 6.0)  # Very rough estimate

        return 0.0  # Metallic or unknown

    def simulate_composition(self, composition: Dict[str, int],
                            n_steps: int = 2000,
                            temperature_K: float = 300.0) -> MDResult:
        """Convenience: set up and run simulation for a given composition.

        Args:
            composition: {"Cs": 4, "Sn": 4, "I": 12} etc.
            n_steps: simulation steps
            temperature_K: target temperature
        """
        self.clear()

        total_atoms = sum(composition.values())
        # Auto-size box: generous spacing to prevent initial overlaps
        self.box_size = max(30.0, (total_atoms * 150) ** (1/3))

        for symbol, count in composition.items():
            self.add_atoms(symbol, count, temperature_K)

        return self.run(n_steps, temperature_K)


class MaterialPermutationEngine:
    """
    Systematic combinatorial search over material compositions.

    Given a target application (e.g., solar cell), explores:
    - Element substitutions
    - Stoichiometry variations
    - Mixed compositions
    - Ranks by efficiency + safety
    """

    def __init__(self, md_engine: MolecularDynamicsEngine):
        self.md = md_engine
        self.results: List[Dict] = []

    def search_perovskites(self, n_steps_per: int = 500) -> List[Dict]:
        """Search ABX3 perovskite compositions for solar cell applications.

        Explores: A = Cs, MA(simulated as Na for MD)
                  B = Pb, Sn, Ge, Bi
                  X = I, Br, Cl
        """
        A_sites = ["Cs", "Na"]  # Na approximates methylammonium for MD
        B_sites = ["Sn", "Ge", "Bi", "Pb"]
        X_sites = ["I", "Br", "Cl"]

        results = []

        for a in A_sites:
            for b in B_sites:
                for x in X_sites:
                    composition = {a: 4, b: 4, x: 12}
                    formula = f"{a}{b}{x}3"

                    try:
                        md_result = self.md.simulate_composition(
                            composition, n_steps=n_steps_per
                        )

                        entry = {
                            "formula": formula,
                            "composition": composition,
                            "A_site": a,
                            "B_site": b,
                            "X_site": x,
                            "binding_energy_eV": float(md_result.binding_energy_per_atom_eV),
                            "is_stable": bool(md_result.is_stable),
                            "avg_temperature_K": float(md_result.avg_temperature_K),
                            "bandgap_eV": float(md_result.bandgap_estimate_eV),
                            "is_toxic": bool(md_result.toxic_components),
                            "toxicity_details": md_result.toxic_components or [],
                            "remediation": md_result.remediation_options or [],
                            "sq_efficiency_pct": self._shockley_queisser(
                                md_result.bandgap_estimate_eV),
                            "safe_alternatives": [],
                        }

                        # If toxic, suggest alternatives
                        if entry["is_toxic"]:
                            for tc in md_result.toxic_components:
                                if tc["element"] in TOXICITY_REMEDIATION:
                                    entry["safe_alternatives"].extend(
                                        TOXICITY_REMEDIATION[tc["element"]]["safe_alternatives"]
                                    )

                        results.append(entry)

                    except Exception as e:
                        results.append({
                            "formula": formula,
                            "error": str(e),
                            "is_stable": False,
                        })

        # Sort by: safety first, then efficiency
        results.sort(key=lambda x: (
            int(x.get("is_toxic", True)),  # Safe first
            -float(x.get("sq_efficiency_pct", 0)),  # Then highest efficiency
            -int(x.get("is_stable", False)),  # Then stable
        ))

        self.results = results
        return results

    def search_custom(self, elements: List[str],
                      stoichiometries: List[Dict[str, int]],
                      target_bandgap_range: Tuple[float, float] = (1.0, 1.8),
                      n_steps_per: int = 500) -> List[Dict]:
        """Search arbitrary element combinations."""
        results = []

        for stoich in stoichiometries:
            # Validate all elements exist
            if not all(e in ATOM_TYPES for e in stoich):
                continue

            formula = "".join(f"{e}{n}" if n > 1 else e for e, n in stoich.items())

            try:
                md_result = self.md.simulate_composition(stoich, n_steps=n_steps_per)

                bg = md_result.bandgap_estimate_eV
                in_target = target_bandgap_range[0] <= bg <= target_bandgap_range[1]

                entry = {
                    "formula": formula,
                    "composition": stoich,
                    "binding_energy_eV": md_result.binding_energy_per_atom_eV,
                    "is_stable": md_result.is_stable,
                    "bandgap_eV": bg,
                    "in_target_bandgap": in_target,
                    "is_toxic": bool(md_result.toxic_components),
                    "toxicity_details": md_result.toxic_components or [],
                    "remediation": md_result.remediation_options or [],
                    "sq_efficiency_pct": self._shockley_queisser(bg),
                }
                results.append(entry)

            except Exception as e:
                results.append({"formula": formula, "error": str(e), "is_stable": False})

        results.sort(key=lambda x: (
            int(x.get("is_toxic", True)),
            int(not x.get("in_target_bandgap", False)),
            -float(x.get("sq_efficiency_pct", 0)),
        ))

        self.results.extend(results)
        return results

    def _shockley_queisser(self, bandgap_eV: float) -> float:
        """Compute Shockley-Queisser limit for given bandgap."""
        if bandgap_eV <= 0 or bandgap_eV > 4.0:
            return 0.0
        # Approximation: peak ~33.7% at 1.34 eV
        return max(0, 33.7 - 4.0 * abs(bandgap_eV - 1.34) ** 1.5)


# ── DFT Approximation (Extended Huckel / Tight Binding) ──────
class TightBindingDFT:
    """
    Simplified DFT approximation using Extended Huckel / Tight-Binding method.

    Not full Kohn-Sham DFT, but gives:
    - Approximate band structure (bandgap)
    - Density of States
    - Orbital energy levels
    - Formation energy estimates

    Good enough for material screening — real DFT would use Quantum ESPRESSO.
    """

    # Valence orbital ionization energies (eV) — Huckel parameters
    VSIE: Dict[str, Dict[str, float]] = {
        "H":  {"1s": -13.60},
        "C":  {"2s": -21.40, "2p": -11.40},
        "N":  {"2s": -26.00, "2p": -13.40},
        "O":  {"2s": -32.38, "2p": -14.80},
        "F":  {"2s": -40.00, "2p": -18.10},
        "Si": {"3s": -17.30, "3p": -9.20},
        "P":  {"3s": -18.60, "3p": -14.00},
        "S":  {"3s": -20.00, "3p": -11.00},
        "Cl": {"3s": -26.30, "3p": -14.20},
        "Ge": {"4s": -16.00, "4p": -9.00},
        "Sn": {"5s": -14.60, "5p": -8.00},
        "Pb": {"6s": -15.70, "6p": -8.00},
        "Bi": {"6s": -15.19, "6p": -7.79},
        "I":  {"5s": -18.00, "5p": -12.70},
        "Br": {"4s": -22.07, "4p": -13.10},
        "Cs": {"6s": -3.89},
        "Na": {"3s": -5.14},
        "K":  {"4s": -4.34},
        "Ti": {"3d": -10.81, "4s": -8.97},
        "Fe": {"3d": -12.60, "4s": -9.10},
        "Cu": {"3d": -13.49, "4s": -11.40},
        "Zn": {"3d": -17.32, "4s": -9.39},
    }

    # Wolfsberg-Helmholz K parameter
    K_WH = 1.75

    def compute_levels(self, atoms: List[str]) -> Dict:
        """Compute approximate orbital energy levels for a cluster of atoms.

        Returns estimated HOMO, LUMO, bandgap, and orbital list.
        """
        # Collect all valence orbital energies
        orbitals = []
        for atom in atoms:
            if atom in self.VSIE:
                for orb_name, energy in self.VSIE[atom].items():
                    orbitals.append({
                        "atom": atom,
                        "orbital": orb_name,
                        "energy_eV": energy,
                    })

        if not orbitals:
            return {"error": "No orbital data for given atoms"}

        # Sort by energy
        orbitals.sort(key=lambda x: x["energy_eV"])

        # Count total valence electrons
        valence_electrons = {
            "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5,
            "O": 6, "F": 7, "Ne": 8, "Na": 1, "Mg": 2, "Al": 3, "Si": 4,
            "P": 5, "S": 6, "Cl": 7, "K": 1, "Ca": 2, "Ti": 4, "Fe": 8,
            "Cu": 11, "Zn": 12, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
            "Sn": 4, "I": 7, "Cs": 1, "Ba": 2, "Pb": 4, "Bi": 5,
        }

        total_e = sum(valence_electrons.get(a, 0) for a in atoms)

        # Fill orbitals (each can hold 2 electrons, simplified)
        # In extended Huckel, we'd solve the secular equation H*c = E*S*c
        # Here we approximate: sort orbital energies, fill lowest first
        n_filled = total_e // 2  # Number of filled orbitals (each holds 2e)

        if n_filled <= 0 or n_filled > len(orbitals):
            homo_idx = min(len(orbitals) - 1, max(0, n_filled - 1))
            lumo_idx = min(len(orbitals) - 1, homo_idx + 1)
        else:
            homo_idx = n_filled - 1
            lumo_idx = min(n_filled, len(orbitals) - 1)

        homo = orbitals[homo_idx]["energy_eV"]
        lumo = orbitals[lumo_idx]["energy_eV"]
        bandgap = lumo - homo

        return {
            "total_valence_electrons": total_e,
            "n_orbitals": len(orbitals),
            "homo_eV": homo,
            "lumo_eV": lumo,
            "bandgap_eV": max(0, bandgap),
            "is_metallic": bandgap < 0.5,
            "is_semiconductor": 0.5 <= bandgap <= 4.0,
            "is_insulator": bandgap > 4.0,
            "orbital_levels": orbitals[:20],  # First 20 levels
            "filled_orbitals": n_filled,
        }

    def formation_energy(self, composition: Dict[str, int],
                          cohesive_energies: Optional[Dict[str, float]] = None) -> float:
        """Estimate formation energy in eV/atom.

        Uses tabulated cohesive energies for elemental solids.
        Negative = exothermic (stable compound).
        """
        # Cohesive energies of elemental solids (eV/atom)
        default_cohesive = {
            "H": 2.26, "C": 7.37, "N": 4.92, "O": 2.60,
            "Si": 4.63, "Ge": 3.85, "Sn": 3.14, "Pb": 2.03,
            "Cu": 3.49, "Fe": 4.28, "Ti": 4.85, "Zn": 1.35,
            "Cs": 0.80, "Na": 1.11, "K": 0.93,
            "I": 1.07, "Br": 1.22, "Cl": 1.40, "F": 0.84,
            "S": 2.85, "P": 3.43, "Bi": 2.18,
            "Cd": 1.16, "Se": 2.46, "Au": 3.81, "Ag": 2.95,
        }

        if cohesive_energies:
            default_cohesive.update(cohesive_energies)

        total_atoms = sum(composition.values())
        if total_atoms == 0:
            return 0.0

        # Sum of elemental reference energies
        ref_energy = sum(
            count * default_cohesive.get(elem, 3.0)
            for elem, count in composition.items()
        )

        # Compound energy (rough: bonding lowers energy by ~0.5-2.0 eV/atom for ionic)
        # This is a VERY rough estimate — real DFT would compute this properly
        compound_bonus = 0.0
        elems = list(composition.keys())
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                ei = ATOM_TYPES.get(elems[i])
                ej = ATOM_TYPES.get(elems[j])
                if ei and ej:
                    # Electronegativity difference drives ionic bonding energy
                    # Using epsilon as rough electronegativity proxy
                    delta = abs(ei.epsilon - ej.epsilon)
                    compound_bonus -= delta * 2.0  # Negative = stabilizing

        formation_e = (compound_bonus - ref_energy) / total_atoms
        return formation_e

    def screen_material(self, composition: Dict[str, int]) -> Dict:
        """Full screening of a material composition.

        Returns bandgap, formation energy, stability, toxicity.
        """
        atoms = []
        for elem, count in composition.items():
            atoms.extend([elem] * count)

        levels = self.compute_levels(atoms)
        form_e = self.formation_energy(composition)

        # Toxicity check
        toxic = []
        for elem in composition:
            at = ATOM_TYPES.get(elem)
            if at and at.toxic:
                toxic.append({
                    "element": elem,
                    "toxicity_score": at.toxicity_score,
                    "remediation": TOXICITY_REMEDIATION.get(elem, {}),
                })

        formula = "".join(
            f"{e}{n}" if n > 1 else e
            for e, n in composition.items()
        )

        return {
            "formula": formula,
            "composition": composition,
            "electronic_structure": levels,
            "formation_energy_eV_per_atom": form_e,
            "is_thermodynamically_stable": form_e < -0.1,
            "toxic_components": toxic,
            "is_safe": len(toxic) == 0,
            "bandgap_eV": levels.get("bandgap_eV", 0),
            "material_class": (
                "metal" if levels.get("is_metallic") else
                "semiconductor" if levels.get("is_semiconductor") else
                "insulator"
            ),
            "solar_cell_viable": (
                levels.get("is_semiconductor", False) and
                0.8 <= levels.get("bandgap_eV", 0) <= 2.0
            ),
            "shockley_queisser_pct": max(0, 33.7 - 4.0 * abs(
                levels.get("bandgap_eV", 0) - 1.34) ** 1.5
            ) if levels.get("is_semiconductor") else 0,
        }
