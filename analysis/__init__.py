"""
Cross-model analysis for the waterfowl detection benchmark.

The set of models compared is *not* hard-coded here: it is read from a manifest
document (by default ``models_v1.docx``) listing one ``model, run_timestamp``
pair per line.  Point the CLI at a different manifest to compare a different
generation of runs without touching any code:

    ./analysis/main.py                          # models_v1.docx
    ./analysis/main.py -m models_v2.docx        # a later set

Outputs land in ``output/analysis/<manifest_stem>/``.
"""

__all__ = ["manifest", "registry", "collect", "figures"]
