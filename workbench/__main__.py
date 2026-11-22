"""python -m workbench — the same as ./waterfowl_detection.sh."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from workbench.app import main  # noqa: E402

raise SystemExit(main())
