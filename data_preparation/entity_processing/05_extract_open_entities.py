#!/usr/bin/env python3
"""Run the open-vocabulary entity extraction stage."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preparation.entity_processing.open_entity_extraction import main


if __name__ == "__main__":
    main()
