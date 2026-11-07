#!/usr/bin/env python3
"""Build the 12D composite lexicon from the three Vietnamese sources."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for candidate in (ROOT / "src", SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lexicon_build.build import main  # noqa: E402


if __name__ == "__main__":
    main()