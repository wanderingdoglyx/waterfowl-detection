#!/usr/bin/env python
"""
Cross-model analysis entry point.

    ./analysis/main.py                      # analyse models_v1.docx
    ./analysis/main.py -m models_v2.docx    # a different set of runs
    ./analysis/main.py --no-figures         # tables only

Which models get compared is decided entirely by the manifest document, so a new
generation of runs needs a new manifest, not a code change.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import config
from analysis import collect, figures, manifest, registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "models_v1.docx")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare evaluated model runs.")
    ap.add_argument("-m", "--manifest", default=DEFAULT_MANIFEST,
                    help="model list (.docx/.txt/.csv/.md). Default: models_v1.docx")
    ap.add_argument("-o", "--out", default=None,
                    help="output dir. Default: output/analysis/<manifest stem>/")
    ap.add_argument("--no-figures", action="store_true", help="skip the plots")
    ap.add_argument("--tasks", action="store_true",
                    help="also run Tasks 2.1-2.4 (efficiency, Pareto, size, density). "
                         "Re-runs inference for any model without a cached prediction set.")
    ap.add_argument("--force", action="store_true",
                    help="with --tasks: recompute predictions and scores from scratch")
    ap.add_argument("--conditions", action="store_true",
                    help="also run the survey-condition analysis: habitat, altitude, "
                         "other metadata, failure-case library, matching audit")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.manifest))[0]
    out_dir = args.out or os.path.join(config.OUTPUT_DIR, "analysis", stem)

    try:
        entries = manifest.load(args.manifest)
    except (FileNotFoundError, ValueError) as e:
        print(f"Manifest error: {e}", file=sys.stderr)
        return 2

    print(f"Manifest {os.path.basename(args.manifest)}: {len(entries)} models")

    records = []
    for e in entries:
        try:
            spec = registry.resolve(e.key)
        except KeyError as err:
            print(f"  {args.manifest}:{e.line_no}: {err}", file=sys.stderr)
            return 2
        rec = collect.load_run(spec, e.run_id)
        records.append(rec)
        status = "ok" if not rec.warnings else "; ".join(rec.warnings)
        ap_txt = f"AP {rec.ap * 100:.2f}%" if rec.ap is not None else "no metrics"
        print(f"  {rec.label:14s} {rec.run_id}  {ap_txt:16s} {status}")

    problems = collect.check_comparable(records)
    if problems:
        print("\nWARNING — these runs are not directly comparable:")
        for p in problems:
            print(f"  - {p}")

    csv_path = collect.write_csv(records, os.path.join(out_dir, "metrics.csv"))
    print(f"\nWrote {csv_path}")

    if not args.no_figures:
        made = figures.render_all(records, os.path.join(out_dir, "figures"))
        for p in made:
            print(f"Wrote {p}")

    if args.tasks:
        from analysis import figures_tasks, pareto, tasks

        print("\n=== Tasks 2.1-2.4 ===")
        specs = [(registry.resolve(e.key), e.run_id) for e in entries]
        try:
            rows, ctx = tasks.build(specs, out_dir, force=args.force)
        except FileNotFoundError as e:
            # A manifest naming a run that was never created is a user error, not
            # a crash — report it like the other manifest problems.
            print(f"\nError: {e}", file=sys.stderr)
            return 2

        for p in tasks.write_task_csvs(rows, ctx, out_dir):
            print(f"Wrote {p}")

        if not args.no_figures:
            fig_dir = os.path.join(out_dir, "figures")
            for x_axis, y_axis in pareto.TRADEOFFS:
                name = f"tradeoff_{x_axis.attr}_vs_{y_axis.attr}.png"
                p = figures_tasks.tradeoff(rows, x_axis, y_axis,
                                           os.path.join(fig_dir, name))
                if p:
                    print(f"Wrote {p}")
            per_model = {r.key: {"label": r.label, "kind": r.kind,
                                 "size": r.size, "density": r.density} for r in rows}
            for p in figures_tasks.size_curves(per_model, ctx["size_labels"], fig_dir):
                print(f"Wrote {p}")
            for p in figures_tasks.density_curves(per_model, ctx["dens_labels"], fig_dir):
                print(f"Wrote {p}")

    if args.conditions:
        from analysis import (failure_library, figures_survey_conditions,
                              matching_audit, survey_conditions)

        print("\n=== Survey-condition analysis ===")
        specs = [(registry.resolve(e.key), e.run_id) for e in entries]
        try:
            cond = survey_conditions.build(specs, force=args.force)
        except FileNotFoundError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 2

        tables = {"habitat": survey_conditions.habitat_table(cond),
                  "altitude": survey_conditions.altitude_table(cond),
                  **survey_conditions.other_tables(cond)}
        for p in survey_conditions.write_csvs(cond, tables, out_dir):
            print(f"Wrote {p}")

        library = failure_library.build(args.manifest, render=not args.no_figures)
        print(f"Failure library: {len(library)} cases")
        print(f"Wrote {failure_library.write_json(library, os.path.join(out_dir, 'failure_library.json'))}")

        if not args.no_figures:
            for p in figures_survey_conditions.render_all(
                    cond, tables, os.path.join(out_dir, "figures")):
                print(f"Wrote {p}")

        audits = {}
        for spec, run_id in specs:
            audits[spec.key] = matching_audit.ensure(spec, run_id, gt=cond.gt,
                                                     force=args.force)
        audit_csv = os.path.join(out_dir, "matching_audit.csv")
        with open(audit_csv, "w") as f:
            f.write("model,false_alarms,swaps,tp_greedy,tp_optimal,tp_lost,"
                    "recall_greedy,recall_optimal\n")
            for spec, _ in specs:
                a = audits[spec.key]
                f.write(f"{spec.key},{a['false_alarms']},{a['swaps']},{a['tp_greedy']},"
                        f"{a['tp_optimal']},{a['tp_lost']},{a['recall_greedy']:.4f},"
                        f"{a['recall_optimal']:.4f}\n")
        print(f"Wrote {audit_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
