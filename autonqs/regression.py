from __future__ import annotations

from dataclasses import dataclass

from .analysis import compare_reference
from .benchmarks import get_benchmark, list_benchmarks
from .training import TrainConfig, train


@dataclass(frozen=True)
class RegressionCase:
    name: str
    steps: int = 5
    walkers: int = 16
    hidden: int = 24
    layers: int = 1
    hidden_density: int = 2
    optimizer: str = "adam"
    max_abs_error_hartree: float | None = None


def reference_table() -> list[dict]:
    rows = []
    for name in list_benchmarks():
        bench = get_benchmark(name)
        rows.append(
            {
                "name": name,
                "electrons": bench.molecule.electron_count,
                "spin_counts": bench.molecule.spin_counts,
                "reference_energy_hartree": bench.reference_energy_hartree,
                "tolerance_hartree": bench.tolerance_hartree,
                "smoke_tolerance_hartree": bench.smoke_tolerance_hartree,
                "source": bench.source,
            }
        )
    return rows


def validate_reference_table() -> None:
    for row in reference_table():
        if row["electrons"] <= 0:
            raise AssertionError(f"{row['name']} has invalid electron count")
        if row["reference_energy_hartree"] >= 0:
            raise AssertionError(f"{row['name']} reference energy should be negative")
        if row["tolerance_hartree"] <= 0 or row["smoke_tolerance_hartree"] <= 0:
            raise AssertionError(f"{row['name']} tolerances must be positive")


def run_regression_case(case: RegressionCase, device: str = "cuda", seed: int = 7) -> dict:
    bench = get_benchmark(case.name)
    cfg = TrainConfig(
        steps=case.steps,
        walkers=case.walkers,
        hidden=case.hidden,
        layers=case.layers,
        hidden_density=case.hidden_density,
        optimizer=case.optimizer,
        burn_in=max(1, min(10, case.steps)),
        mcmc_steps=2,
        seed=seed,
        deterministic=True,
        device=device,
    )
    result = train(bench.molecule, cfg)
    result.pop("model")
    energy = result["analysis"]["energy_blocks"]["mean"]
    comparison = compare_reference(case.name, energy)
    threshold = case.max_abs_error_hartree or bench.smoke_tolerance_hartree
    passed = comparison["has_reference"] and comparison["abs_error_hartree"] <= threshold
    return {
        "case": case.name,
        "passed": passed,
        "threshold_hartree": threshold,
        "comparison": comparison,
        "result": result,
    }


def run_regression_suite(cases: list[RegressionCase], device: str = "cuda", seed: int = 7) -> dict:
    results = [run_regression_case(case, device=device, seed=seed) for case in cases]
    return {
        "passed": all(item["passed"] for item in results),
        "total": len(results),
        "failed": [item["case"] for item in results if not item["passed"]],
        "results": results,
    }


def default_smoke_cases() -> list[RegressionCase]:
    return [
        RegressionCase("h", max_abs_error_hartree=2.0),
        RegressionCase("h2", max_abs_error_hartree=5.0),
        RegressionCase("he", max_abs_error_hartree=5.0),
    ]
