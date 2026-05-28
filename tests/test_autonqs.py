import math

import pytest
import torch

from autonqs.hamiltonian import local_energy
from autonqs.analysis import analyze_history, compare_reference
from autonqs.benchmarks import get_benchmark, list_benchmarks
from autonqs.config import RunConfig, run_config_from_dict
from autonqs.forces import estimate_forces
from autonqs.optim import OptimizerConfig, build_optimizer
from autonqs.periodic import minimum_image_displacement, wrap_positions
from autonqs.pseudopotentials import get_pseudopotential, local_pseudopotential_energy
from autonqs.molecules import get_molecule, list_molecules
from autonqs.network import AutoNQS
from autonqs.profiler import compare_profile, profile_training
from autonqs.regression import RegressionCase, reference_table, run_regression_case, validate_reference_table
from autonqs.sampler import adapt_step_size, burn_in, make_state, metropolis_step, metropolis_sweep
from autonqs.training import TrainConfig, train


def test_twenty_molecule_specs_are_available():
    names = list_molecules()
    assert len(names) == 20
    assert {"h2", "h2o", "benzene", "femoco"}.issubset(names)
    for name in names:
        mol = get_molecule(name)
        assert len(mol.symbols) == len(mol.coords_angstrom)
        assert mol.electron_count > 0
        assert sum(mol.spin_counts) == mol.electron_count


def test_nqs_has_pauli_nodes_for_same_spin_swap():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol = get_molecule("h2")
    nuclei, charges = mol.tensors(device)
    model = AutoNQS(2, 2, nuclei, charges, hidden=32, layers=1, hidden_density=2, orbital_jitter=0.0).to(device)
    x = torch.randn(4, 2, 3, device=device)
    sign, logabs = model.slog_psi(x)
    swapped = x[:, [1, 0], :]
    sign_swapped, logabs_swapped = model.slog_psi(swapped)
    assert torch.allclose(logabs, logabs_swapped, atol=1e-5)
    assert torch.allclose(sign, -sign_swapped, atol=1e-5)


def test_backflow_starts_as_identity_and_preserves_exchange_symmetry():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol = get_molecule("h2")
    nuclei, charges = mol.tensors(device)
    model = AutoNQS(
        2,
        2,
        nuclei,
        charges,
        hidden=24,
        layers=1,
        hidden_density=2,
        orbital_reference=True,
        orbital_jitter=0.0,
    ).to(device)
    x = torch.randn(4, 2, 3, device=device)
    assert torch.allclose(model.backflow_coordinates(x), x, atol=1e-7)

    sign, logabs = model.slog_psi(x)
    swapped = x[:, [1, 0], :]
    sign_swapped, logabs_swapped = model.slog_psi(swapped)
    assert torch.allclose(logabs, logabs_swapped, atol=1e-5)
    assert torch.allclose(sign, -sign_swapped, atol=1e-5)


def test_orbital_reference_and_backflow_are_trainable():
    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol = get_molecule("h2")
    nuclei, charges = mol.tensors(device)
    model = AutoNQS(
        mol.electron_count,
        mol.spin_counts[0],
        nuclei,
        charges,
        hidden=16,
        pair_hidden=8,
        layers=1,
        hidden_density=1,
        orbital_reference=True,
    ).to(device)
    x = torch.randn(5, mol.electron_count, 3, device=device)
    _, logabs = model.slog_psi(x)
    logabs.mean().backward()
    assert model.up_orbital_coeff.grad is not None
    assert torch.isfinite(model.up_orbital_coeff.grad).all()
    assert model.backflow_net[-1].weight.grad is not None
    assert torch.isfinite(model.backflow_net[-1].weight.grad).all()


def test_cusp_initialization_rbm_weights_and_translation_invariance():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol = get_molecule("h2")
    nuclei, charges = mol.tensors(device)
    model = AutoNQS(mol.electron_count, mol.spin_counts[0], nuclei, charges, hidden=16, pair_hidden=8, layers=1, hidden_density=2).to(device)
    assert torch.allclose(torch.nn.functional.softplus(model.en_cusp_scale), torch.tensor(1.0, device=device), atol=1e-6)
    assert torch.allclose(torch.nn.functional.softplus(model.ee_cusp_same), torch.tensor(0.25, device=device), atol=1e-6)
    assert torch.allclose(torch.nn.functional.softplus(model.ee_cusp_opposite), torch.tensor(0.5, device=device), atol=1e-6)
    assert model.hidden_linear.weight.shape[0] == model.rbm_hidden

    x = torch.randn(3, mol.electron_count, 3, device=device)
    _, logabs = model.slog_psi(x)
    logabs.sum().backward()
    assert model.hidden_linear.weight.grad is not None
    assert torch.isfinite(model.hidden_linear.weight.grad).all()
    assert model.hidden_linear.weight.grad.abs().sum() > 0

    with torch.no_grad():
        shift = torch.tensor([1.25, -0.5, 0.75], device=device)
        _, before = model.slog_psi(x)
        model.nuclei.add_(shift)
        _, after = model.slog_psi(x + shift)
    assert torch.allclose(before, after, atol=1e-5)


def test_local_energy_and_sampler_are_finite():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol = get_molecule("h2")
    nuclei, charges = mol.tensors(device)
    model = AutoNQS(mol.electron_count, mol.spin_counts[0], nuclei, charges, hidden=24, layers=1, hidden_density=2).to(device)
    state = make_state(model, 8)
    state = metropolis_step(model, state, 0.03)
    state = metropolis_sweep(model, state, 0.03)
    energy = local_energy(model, state.positions)
    assert torch.isfinite(energy).all()
    assert 0.0 <= state.acceptance <= 1.0


def test_adaptive_sampler_and_benchmark_catalog():
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench = get_benchmark("h2")
    nuclei, charges = bench.molecule.tensors(device)
    model = AutoNQS(bench.molecule.electron_count, bench.molecule.spin_counts[0], nuclei, charges, hidden=24, layers=1, hidden_density=2).to(device)
    state = make_state(model, 8, step_size=0.03)
    state = burn_in(model, state, 2)
    state = adapt_step_size(state)
    assert state.diagnostics.total_proposals > 0
    assert state.step_size > 0
    assert {"h", "he", "lih", "be", "h2", "n2", "h2o"}.issubset(list_benchmarks())
    assert bench.reference_energy_hartree < 0


def test_config_and_optimizer_surfaces():
    cfg = run_config_from_dict(
        {
            "molecule": "h2",
            "network": {"hidden": 16, "pair_hidden": 8, "layers": 1, "hidden_density": 1},
            "optimizer": {"name": "natural", "lr": 1e-3},
        }
    )
    assert isinstance(cfg, RunConfig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol = get_molecule("h2")
    nuclei, charges = mol.tensors(device)
    model = AutoNQS(mol.electron_count, mol.spin_counts[0], nuclei, charges, hidden=16, pair_hidden=8, layers=1, hidden_density=1).to(device)
    opt = build_optimizer(model, OptimizerConfig(name="diag-natural", lr=1e-3))
    assert opt.__class__.__name__ == "DiagonalNaturalGradient"


def test_hydrogenic_local_energy_matches_analytic_ground_state():
    class HydrogenOneS(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.nuclei = torch.zeros(1, 3, device=device)
            self.charges = torch.ones(1, device=device)

        def forward(self, x):
            return -torch.linalg.norm(x[:, 0, :], dim=-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HydrogenOneS(device)
    x = torch.tensor([[[0.7, 0.2, -0.1]], [[1.2, -0.4, 0.3]]], device=device)
    energy = local_energy(model, x)
    assert torch.allclose(energy, torch.full_like(energy, -0.5), atol=2e-4)


def test_periodic_pseudopotential_forces_and_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell = torch.eye(3, device=device) * 4.0
    delta = torch.tensor([[[3.5, 0.0, 0.0]]], device=device)
    assert torch.allclose(minimum_image_displacement(delta, cell), torch.tensor([[[-0.5, 0.0, 0.0]]], device=device))
    wrapped = wrap_positions(torch.tensor([[[4.2, -0.1, 0.0]]], device=device), cell)
    assert (wrapped >= 0).all()

    mol = get_molecule("h2")
    nuclei, charges = mol.tensors(device)
    model = AutoNQS(mol.electron_count, mol.spin_counts[0], nuclei, charges, hidden=16, pair_hidden=8, layers=1, hidden_density=1).to(device)
    model.register_buffer("cell", cell)
    model.pseudopotentials = (get_pseudopotential("BFD-H"), get_pseudopotential("BFD-H"))
    positions = torch.randn(4, mol.electron_count, 3, device=device)
    pp = local_pseudopotential_energy(positions, nuclei, charges, model.pseudopotentials)
    forces = estimate_forces(model, positions)
    assert pp.shape == (4,)
    assert forces.shape == nuclei.shape
    assert torch.isfinite(forces).all()

    analysis = analyze_history([{"energy": -1.0}, {"energy": -1.2}, {"energy": -1.1}], "h2", block_size=2)
    assert analysis["best_energy"] == -1.2
    assert compare_reference("h2", -1.17)["has_reference"]


def test_reference_regression_metadata_and_profile_compare():
    validate_reference_table()
    rows = reference_table()
    assert {row["name"] for row in rows}.issuperset({"h", "he", "h2", "h2o"})
    current = {"steps_per_second": 8.0}
    baseline = {"steps_per_second": 10.0}
    assert compare_profile(current, baseline, max_slowdown=1.5)["passed"]
    assert not compare_profile(current, baseline, max_slowdown=1.1)["passed"]


def test_deterministic_reproducibility_on_cpu():
    cfg = TrainConfig(
        steps=1,
        walkers=4,
        hidden=8,
        layers=1,
        hidden_density=1,
        burn_in=1,
        mcmc_steps=1,
        seed=123,
        deterministic=True,
        device="cpu",
    )
    first = train(get_molecule("h2"), cfg)
    second = train(get_molecule("h2"), cfg)
    assert first["history"] == second["history"]
    assert first["analysis"] == second["analysis"]


def test_regression_case_and_profiler_smoke_on_cpu():
    regression = run_regression_case(RegressionCase("h2", steps=1, walkers=4, hidden=8, hidden_density=1), device="cpu", seed=5)
    assert regression["comparison"]["has_reference"]
    assert "passed" in regression
    profile = profile_training("h2", steps=1, walkers=4, device="cpu")
    assert profile["steps_per_second"] > 0
    assert profile["parameters"] > 0


def test_stochastic_reconfiguration_training_step_on_cpu():
    result = train(
        get_molecule("h2"),
        TrainConfig(
            steps=1,
            walkers=4,
            hidden=8,
            layers=1,
            hidden_density=1,
            burn_in=1,
            mcmc_steps=1,
            optimizer="natural",
            lr=1e-3,
            damping=1e-2,
            device="cpu",
            deterministic=True,
        ),
    )
    assert math.isfinite(result["final_energy"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_short_training_runs_on_cuda():
    result = train(get_molecule("h2"), TrainConfig(steps=3, walkers=8, hidden=16, layers=1, hidden_density=1, mcmc_steps=2, device="cuda"))
    assert result["device"] == "cuda"
    assert result["steps_per_second"] > 0
    assert math.isfinite(result["final_energy"])
