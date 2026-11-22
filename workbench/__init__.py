"""
Waterfowl Workbench — a terminal interface over the whole pipeline.

    ./waterfowl_detection.sh        # open the main menu
    ./waterfowl_detection.sh jobs   # jump straight to one screen

It adds no second way of doing anything the pipeline already does.  Preparing data,
training, evaluation, baselines and the cross-model analysis are the existing entry
points (`pipeline.cli`, `<model>/main.py`, `baseline/main.py`, `analysis/main.py`),
launched with the same flags a person would type, as background jobs whose logs the
workbench can follow.  What it adds is the part of the workflow that had no command:
counting birds in *new* survey photographs (`workbench.detect`), reviewing and
correcting those counts (`workbench.reviewer`), and exporting the result
(`workbench.reports`).

See workbench/README.md for the menu map and the design decisions.
"""

__version__ = "1.0.0"
