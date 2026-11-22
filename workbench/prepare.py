#!/usr/bin/env python
"""
Make a data folder ready for every model, up front.

    WATERFOWL_DATASET=dataset_teal python -m workbench.prepare

Every model already does this when it starts, through `pipeline.auto.ensure_data`: read
the folder, filter the published crops to its photographs, tile any photograph that has
no crops yet, and build the on-disk layout its framework reads.  This calls the same
function once for each layout (Ultralytics and OWL), so the work — and its cache, keyed
on the folder's fingerprint — is exactly what the first model run would otherwise do.

It is used instead of `pipeline.cli select --from <folder>` because that command only
filters crops that already exist: photographs imported as a new training collection have
none, and would be left out.  For the configured default folder (the whole corpus) use
`python -m pipeline.cli prepare --mirrors`, which builds the published splits themselves.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_prep import config  # noqa: E402
from pipeline.auto import ensure_data  # noqa: E402


def main() -> int:
    print(f"Preparing {os.path.relpath(config.DATASET_ROOT, ROOT)} for every model", flush=True)
    name = ensure_data(mirror="yolo")
    ensure_data(mirror="mdo", verbose=False)
    if name is None:
        print("The folder holds exactly the published corpus: the committed splits are used.")
        return 0
    record = os.path.join(config.EXPERIMENTS_DIR, name, "selection.json")
    with open(record) as f:
        summary = json.load(f).get("summary", {})
    for split, counts in (summary.get("splits") or {}).items():
        print(f"  {split:6s} {counts.get('crops', 0):8,d} crops  {counts.get('birds', 0):8,d} birds  "
              f"{counts.get('photographs', 0):6,d} photographs")
    print(f"Ready: every model launched with WATERFOWL_DATASET={os.path.basename(config.DATASET_ROOT)} "
          f"reads output/experiments/{name}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
