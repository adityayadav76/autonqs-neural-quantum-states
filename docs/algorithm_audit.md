# Algorithm Audit And Fixes

This document records algorithmic problems found during the research-grade
cleanup pass and the tests added to prevent regressions.

## Fixed Issues

| Area | Problem | Fix | Test Coverage |
|---|---|---|---|
| RBM hidden weights | Hidden-unit weights must participate in the variational gradient for the NQS ansatz to learn. | Test that the RBM hidden linear weights receive finite nonzero gradients through `log|psi|`. | `test_cusp_initialization_rbm_weights_and_translation_invariance` checks hidden-weight gradients. |
| Kato cusp parameters | Cusp values were passed through `softplus` after naive initialization, producing incorrect electron-nuclear and electron-electron cusp coefficients. | Initialize raw parameters with inverse-softplus so values are exactly `1.0`, `0.5`, and `0.25`; freeze by default. | `test_cusp_initialization_rbm_weights_and_translation_invariance`. |
| Translation invariance | Absolute electron coordinates were part of one-electron features, so shifting all nuclei/electrons changed the wavefunction. | Remove absolute coordinates from one-electron inputs; use electron-nucleus relative vectors and distances. | `test_cusp_initialization_rbm_weights_and_translation_invariance`. |
| MCMC scaling | Whole-walker proposals moved all electrons at once, which mixes poorly as electron count grows. | Add electron-wise Metropolis sweeps and make them the training default. | `test_local_energy_and_sampler_are_finite` exercises both whole-walker and electron-wise moves. |
| Natural gradient | `natural` was actually diagonal RMS-style gradient scaling, not VMC stochastic reconfiguration. | Add walker-space stochastic reconfiguration and route `natural`/`sr` to it; keep the old method as `diag-natural`. | `test_stochastic_reconfiguration_training_step_on_cpu`. |
| Local energy validation | No analytic local-energy check existed. | Add a hydrogen 1s analytic model test with expected local energy `-0.5 Ha`. | `test_hydrogenic_local_energy_matches_analytic_ground_state`. |
| Reproducibility | Deterministic controls were incomplete at the run-config/CLI surface. | Add `deterministic` to config, training, CLI, and regression tests. | `test_deterministic_reproducibility_on_cpu`. |

## Current Scientific Limitations

The framework is substantially less toy-like after these fixes, but still needs
more work before claiming production NQS parity:

- a full, scalable stochastic reconfiguration/K-FAC schedule for larger walker
  counts
- better visible-feature and RBM hidden-unit initialization from chemical priors
- robust local-energy clipping and thermalization policies
- nonlocal pseudopotential projectors
- Ewald/twist-averaged periodic Coulomb terms
- longer benchmark convergence studies with saved checkpoints
