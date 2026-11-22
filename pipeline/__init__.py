"""
Common dataset interface (Task 4.3).

Survey programmes reach the pipeline through an *adapter* — one per programme, which is
the only place that knows about that programme's folder layout, file naming, spreadsheet
columns, annotation format, label vocabulary and cameras.  Adapters yield canonical
records (`pipeline.records`), and everything above them works with fields rather than
file names:

    dataset_full/            ->  MdcAdapter        \\
                                                     ->  Photograph / Annotation records
    <AGFC delivery>/         ->  ArkansasAdapter   /              |
                                                                  v
                              splits, tiling, COCO, models, evaluation, analysis

The rule that makes the boundary testable: **above the interface, no code parses a file
name and no code names a collection or a programme.**

Entry points:

    from pipeline.adapters import MdcAdapter, ArkansasAdapter
    from pipeline import conformance, crops, splits

    python -m pipeline.cli describe   --source mdc
    python -m pipeline.cli coverage   --source mdc
    python -m pipeline.cli conform    --source mdc
    python -m pipeline.cli split      --source mdc
    python -m pipeline.verify                       # equivalence with the current pipeline
"""

__all__ = ["records", "vocabularies", "cameras", "adapters", "splits", "crops",
           "conformance"]
