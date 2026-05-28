from __future__ import annotations

from dataclasses import dataclass

from .molecules import Molecule, get_molecule


@dataclass(frozen=True)
class Benchmark:
    molecule: Molecule
    reference_energy_hartree: float
    source: str
    tolerance_hartree: float = 0.01
    smoke_tolerance_hartree: float = 5.0


BENCHMARKS: dict[str, Benchmark] = {
    "h": Benchmark(
        Molecule("h", ("H",), ((0.0, 0.0, 0.0),), spin=1),
        -0.5,
        "Analytic non-relativistic hydrogen atom ground state",
        0.005,
        2.0,
    ),
    "he": Benchmark(
        Molecule("he", ("He",), ((0.0, 0.0, 0.0),)),
        -2.903724377,
        "Non-relativistic helium benchmark energy",
        0.02,
        5.0,
    ),
    "lih": Benchmark(
        Molecule("lih", ("Li", "H"), ((0.0, 0.0, -0.7975), (0.0, 0.0, 0.7975))),
        -7.882,
        "Approximate all-electron LiH equilibrium benchmark",
        0.05,
        15.0,
    ),
    "be": Benchmark(
        Molecule("be", ("Be",), ((0.0, 0.0, 0.0),)),
        -14.66736,
        "Non-relativistic beryllium atom benchmark energy",
        0.05,
        20.0,
    ),
    "h2": Benchmark(
        get_molecule("h2"),
        -1.174475,
        "Near-exact H2 equilibrium energy around 0.74 Angstrom",
        0.02,
        5.0,
    ),
    "n2": Benchmark(
        get_molecule("n2"),
        -109.542,
        "Approximate all-electron N2 equilibrium benchmark",
        0.1,
        120.0,
    ),
    "h2o": Benchmark(
        get_molecule("h2o"),
        -76.438,
        "Approximate all-electron water equilibrium benchmark",
        0.1,
        90.0,
    ),
}


def list_benchmarks() -> list[str]:
    return list(BENCHMARKS)


def get_benchmark(name: str) -> Benchmark:
    key = name.lower().replace("-", "_")
    if key not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}. Options: {', '.join(list_benchmarks())}")
    return BENCHMARKS[key]
