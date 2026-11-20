"""Verify frozen environment checksums."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ENVIRONMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENT_ROOT / "src"))

from research_sim.freeze import verify_frozen_bundle


if __name__ == "__main__":
    print(json.dumps(
        verify_frozen_bundle(ENVIRONMENT_ROOT / "FROZEN_SHA256SUMS"),
        indent=2,
    ))
