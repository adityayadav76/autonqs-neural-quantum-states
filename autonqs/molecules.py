from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import torch

from .constants import ATOMIC_NUMBERS, BOHR_PER_ANGSTROM


@dataclass(frozen=True)
class Molecule:
    name: str
    symbols: tuple[str, ...]
    coords_angstrom: tuple[tuple[float, float, float], ...]
    charge: int = 0
    spin: int = 0
    active_electrons: int | None = None
    description: str = ""
    cell_angstrom: tuple[tuple[float, float, float], ...] | None = None
    pseudopotentials: tuple[str, ...] | None = None

    @property
    def nuclear_charges(self) -> tuple[int, ...]:
        return tuple(ATOMIC_NUMBERS[s] for s in self.symbols)

    @property
    def total_electrons(self) -> int:
        return sum(self.nuclear_charges) - self.charge

    @property
    def electron_count(self) -> int:
        return self.active_electrons or self.total_electrons

    @property
    def spin_counts(self) -> tuple[int, int]:
        n = self.electron_count
        n_up = (n + self.spin) // 2
        n_down = n - n_up
        if n_up < 0 or n_down < 0 or n_up + n_down != n:
            raise ValueError(f"Invalid spin={self.spin} for {n} active electrons")
        return n_up, n_down

    def tensors(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        coords = torch.tensor(self.coords_angstrom, dtype=torch.get_default_dtype(), device=device)
        charges = torch.tensor(self.nuclear_charges, dtype=torch.get_default_dtype(), device=device)
        return coords * BOHR_PER_ANGSTROM, charges

    def cell_tensor(self, device: torch.device | str) -> torch.Tensor | None:
        if self.cell_angstrom is None:
            return None
        return torch.tensor(self.cell_angstrom, dtype=torch.get_default_dtype(), device=device) * BOHR_PER_ANGSTROM


def linear(symbols: list[str], spacing: float) -> tuple[tuple[float, float, float], ...]:
    offset = 0.5 * spacing * (len(symbols) - 1)
    return tuple((i * spacing - offset, 0.0, 0.0) for i in range(len(symbols)))


def ring(symbol: str, count: int, radius: float, z: float = 0.0) -> list[tuple[str, tuple[float, float, float]]]:
    return [
        (symbol, (radius * cos(2 * pi * i / count), radius * sin(2 * pi * i / count), z))
        for i in range(count)
    ]


def _specs() -> dict[str, Molecule]:
    benzene_c = ring("C", 6, 1.397)
    benzene_h = ring("H", 6, 2.48)
    pentacene_c = [("C", (1.4 * (i % 11 - 5), 1.2 * (i // 11), 0.0)) for i in range(22)]
    pentacene_h = [("H", (1.4 * (i % 7 - 3), -1.1 + 3.4 * (i // 7), 0.0)) for i in range(14)]
    data = [
        Molecule("h2", ("H", "H"), ((-0.37, 0, 0), (0.37, 0, 0)), description="Hydrogen"),
        Molecule("ch2", ("C", "H", "H"), ((0, 0, 0), (0, 1.08, 0), (1.02, -0.36, 0)), spin=2, description="Methylene triplet"),
        Molecule("nh3", ("N", "H", "H", "H"), ((0, 0, 0.11), (0.94, 0, -0.27), (-0.47, 0.81, -0.27), (-0.47, -0.81, -0.27)), description="Ammonia"),
        Molecule("h2o", ("O", "H", "H"), ((0, 0, 0), (0.958, 0, 0), (-0.24, 0.927, 0)), description="Water"),
        Molecule("c2", ("C", "C"), ((-0.621, 0, 0), (0.621, 0, 0)), description="Dicarbon"),
        Molecule("n2", ("N", "N"), ((-0.55, 0, 0), (0.55, 0, 0)), description="Nitrogen"),
        Molecule("c2h4", ("C", "C", "H", "H", "H", "H"), ((-0.67, 0, 0), (0.67, 0, 0), (-1.23, 0.93, 0), (-1.23, -0.93, 0), (1.23, 0.93, 0), (1.23, -0.93, 0)), description="Ethylene"),
        Molecule("o3", ("O", "O", "O"), ((0, 0, 0), (1.08, 0.58, 0), (-1.08, 0.58, 0)), description="Ozone"),
        Molecule("ch2s", ("C", "S", "H", "H"), ((0, 0, 0), (1.61, 0, 0), (-0.58, 0.94, 0), (-0.58, -0.94, 0)), description="Thioformaldehyde"),
        Molecule("benzene", tuple(x[0] for x in benzene_c + benzene_h), tuple(x[1] for x in benzene_c + benzene_h), active_electrons=30, description="Benzene, valence active space"),
        Molecule("pyrazine", ("N", "C", "C", "N", "C", "C", "H", "H", "H", "H"), ((1.4,0,0),(0.7,1.21,0),(-0.7,1.21,0),(-1.4,0,0),(-0.7,-1.21,0),(0.7,-1.21,0),(1.25,2.16,0),(-1.25,2.16,0),(-1.25,-2.16,0),(1.25,-2.16,0)), active_electrons=32, description="Pyrazine, valence active space"),
        Molecule("mnc", ("Mn", "C"), ((-0.8, 0, 0), (0.8, 0, 0)), spin=1, active_electrons=14, description="Manganese carbide active model"),
        Molecule("tio2", ("Ti", "O", "O"), ((0, 0, 0), (-1.62, 0, 0), (1.62, 0, 0)), active_electrons=16, description="Titanium oxide representative"),
        Molecule("cr2", ("Cr", "Cr"), ((-0.84, 0, 0), (0.84, 0, 0)), active_electrons=24, description="Chromium dimer active model"),
        Molecule("fe2s2", ("Fe", "Fe", "S", "S"), ((-1.35, 0, 0), (1.35, 0, 0), (0, 1.45, 0), (0, -1.45, 0)), active_electrons=28, description="Iron-sulfur cluster representative"),
        Molecule("pentacene", tuple(x[0] for x in pentacene_c + pentacene_h), tuple(x[1] for x in pentacene_c + pentacene_h), active_electrons=64, description="Pentacene active model"),
        Molecule("oxo_mn_salen", ("Mn", "O", "N", "N", "C", "C", "C", "C", "H", "H", "H", "H"), ((0,0,0),(0,0,1.6),(-1.8,0,0),(1.8,0,0),(-1.2,1.4,0),(1.2,1.4,0),(-1.2,-1.4,0),(1.2,-1.4,0),(-2.1,1.9,0),(2.1,1.9,0),(-2.1,-1.9,0),(2.1,-1.9,0)), spin=1, active_electrons=34, description="oxo-Mn(salen) representative"),
        Molecule("ir_complex", ("Ir", "C", "C", "N", "N", "H", "H", "H", "H"), ((0,0,0),(1.9,0,0),(-1.9,0,0),(0,1.9,0),(0,-1.9,0),(2.7,0.8,0),(-2.7,-0.8,0),(0.8,2.7,0),(-0.8,-2.7,0)), active_electrons=28, description="Iridium complex active model"),
        Molecule("fe_porphyrin", ("Fe", "N", "N", "N", "N", "C", "C", "C", "C", "H", "H", "H", "H"), ((0,0,0),(1.9,0,0),(-1.9,0,0),(0,1.9,0),(0,-1.9,0),(2.8,0,0),(-2.8,0,0),(0,2.8,0),(0,-2.8,0),(3.5,0.8,0),(-3.5,-0.8,0),(0.8,3.5,0),(-0.8,-3.5,0)), spin=2, active_electrons=34, description="Iron porphyrin active model"),
        Molecule("femoco", ("Fe", "Fe", "Fe", "Fe", "S", "S", "S", "S", "C", "Mo"), ((-1.4,0,0),(1.4,0,0),(0,-1.4,0),(0,1.4,0),(-1,1,1),(1,-1,1),(-1,-1,-1),(1,1,-1),(0,0,0),(0,0,1.8)), spin=3, active_electrons=54, description="FeMoco cluster representative"),
    ]
    return {m.name: m for m in data}


MOLECULES = _specs()


def list_molecules() -> list[str]:
    return list(MOLECULES)


def get_molecule(name: str) -> Molecule:
    key = name.lower().replace("-", "_")
    if key not in MOLECULES:
        raise KeyError(f"Unknown molecule {name!r}. Options: {', '.join(list_molecules())}")
    return MOLECULES[key]
