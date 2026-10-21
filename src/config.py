"""Single source of truth for paths/device so the same code runs on the
laptop (CPU, no GPU, calibration-only) and the remote GPU box (inference +
caching) without edits — override any of these with env vars.

Compute split (section 9.3 of the build guide): the remote GPU box runs
model inference once and writes caches; calibration experiments are pure
numpy and run on any CPU (remote or laptop).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("SPACAL_DATA_ROOT", REPO_ROOT / "data"))
CACHE_ROOT = Path(os.environ.get("SPACAL_CACHE_ROOT", REPO_ROOT / "cache"))
PREPARED_ROOT = Path(os.environ.get("SPACAL_PREPARED_ROOT", REPO_ROOT / "prepared"))
RESULTS_ROOT = Path(os.environ.get("SPACAL_RESULTS_ROOT", REPO_ROOT / "results"))
DEVICE = os.environ.get("SPACAL_DEVICE", "cpu")  # set SPACAL_DEVICE=cuda on the remote box
SEED = int(os.environ.get("SPACAL_SEED", "0"))
