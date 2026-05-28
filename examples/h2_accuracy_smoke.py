"""Short H2 run intended to verify CUDA execution and energy descent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.molecules import get_molecule
from autonqs.training import TrainConfig, train


cfg = TrainConfig(steps=80, walkers=96, hidden=64, layers=2, hidden_density=4, device="cuda", seed=11)
result = train(get_molecule("h2"), cfg)
result.pop("model")
print(json.dumps(result, indent=2))
