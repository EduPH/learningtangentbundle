"""
Evaluate the sign-constancy criterion on every SAVED atlas -- no retraining.

This is the first thing the saved atlases buy us.  The criterion needs the
trained networks (it differentiates det g_ji), which is why it had previously
been run only on the manifolds we happened to retrain.  Every experiment in the
paper now stores its atlas, so the criterion can be applied to all of them
post hoc, at the cost of a few forward and backward passes.

Usage:
    python eval_saved.py                       # every saved atlas
    python eval_saved.py --glob '*cyclo*'      # a subset
    python eval_saved.py --max-per-run 1       # one seed per experiment (fast)
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, "..", "src"))


def find_atlases(pattern="*"):
    """All saved atlas directories, newest experiment first.

    `pattern` is matched against the whole relative path, so it can select
    either an experiment ('*cyclo*') or a chart name ('*seed42').

    Runs archived under experiments/_superseded/ are skipped: they are kept on
    disk for provenance but are not reported in the paper, and including them
    would silently mix superseded atlases into the margin totals.
    """
    import fnmatch
    out = []
    for root in sorted(glob.glob(os.path.join(EXP, "**", "atlases"),
                                 recursive=True),
                       key=os.path.getmtime, reverse=True):
        if "_superseded" in os.path.relpath(root, EXP).split(os.sep):
            continue
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isfile(os.path.join(d, "meta.json")):
                continue
            rel = os.path.relpath(d, EXP)
            if pattern == "*" or fnmatch.fnmatch(rel, pattern) \
               or fnmatch.fnmatch(os.path.basename(d), pattern):
                out.append(d)
    return out


def label(path):
    """A readable experiment name from the directory layout."""
    run = os.path.basename(os.path.dirname(os.path.dirname(path)))
    name = os.path.basename(path)
    if "cyclooctane" in run:
        return f"cyclo-octane {name.rsplit('_seed', 1)[0]}"
    if "patches" in run:
        return f"patches ({run.split('_')[-1]})"
    return name.rsplit("_seed", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*")
    ap.add_argument("--max-per-run", type=int, default=None,
                    help="evaluate at most this many atlases per experiment")
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--into", default=None,
                    help="write into this results dir, skipping atlases it "
                         "already contains; lets a long evaluation accumulate "
                         "across several invocations")
    ap.add_argument("--budget", type=float, default=None,
                    help="stop cleanly after this many seconds")
    ap.add_argument("--exclude", default="",
                    help="comma-separated substrings of run paths to skip; "
                         "used to leave out the one-seed runs kept only for "
                         "the eta validation, whose atlases are undertrained")
    a = ap.parse_args()

    from atlasae import load_atlas, sign_constancy_report, certify_verdict

    paths = find_atlases(a.glob)
    if a.max_per_run:
        seen, keep = {}, []
        for p in paths:
            k = os.path.dirname(p)
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= a.max_per_run:
                keep.append(p)
        paths = keep
    if a.exclude:
        bad = [x for x in a.exclude.split(",") if x]
        paths = [p for p in paths if not any(b in p for b in bad)]

    done, rows_prev = set(), []
    out_dir = os.path.join(HERE, a.into) if a.into else None
    if out_dir and os.path.isfile(os.path.join(out_dir, "results.json")):
        rows_prev = json.load(open(os.path.join(out_dir, "results.json")))
        done = {r["path"] for r in rows_prev}
        paths = [p for p in paths if os.path.relpath(p, EXP) not in done]
        print(f"resuming: {len(done)} already evaluated, {len(paths)} to go")
    if not paths:
        if done:
            print(f"nothing to do: all {len(done)} atlases are already in "
                  f"{a.into}.  --into resumes rather than recomputes; delete "
                  f"the directory, or pass a different --into, to redo them.")
        else:
            print("no saved atlases found.  Trained atlases are gitignored, so "
                  "a fresh clone has none: run ../run_all.sh to retrain, or "
                  "copy an existing <run>/atlases/ directory into place.")
        return

    print(f"{'experiment':26} {'seed':>5} {'charts':>7} {'overlaps':>9} "
          f"{'certified':>10} {'median':>8} {'worst':>8}  {'verdict':15} {'cert':>4}")
    print("-" * 104)
    import time
    t0 = time.time()
    rows = list(rows_prev)
    for p in paths:
        if a.budget and time.time() - t0 > a.budget:
            print(f"[budget] stopping after {len(rows)-len(rows_prev)} new; "
                  f"rerun the same command to continue")
            break
        meta = json.load(open(os.path.join(p, "meta.json")))
        try:
            system = load_atlas(p)
            r = sign_constancy_report(system, min_points=a.min_points)
        except Exception as e:                      # keep going on one failure
            print(f"{label(p):26} {'--':>5}  FAILED: {type(e).__name__}: {e}")
            continue
        if r["n_overlaps"] == 0:
            continue
        cert = certify_verdict(r["rows"])
        rows.append(dict(experiment=label(p), path=os.path.relpath(p, EXP),
                         seed=meta.get("seed"), n_charts=meta["n_charts"],
                         verdict=cert["verdict"],
                         verdict_certified=cert["certified"],
                         verdict_reason=cert["reason"],
                         witness=cert.get("witness"),
                         overlaps=r["rows"],
                         **{k: v for k, v in r.items() if k != "rows"}))
        flag = "CERT" if cert["certified"] else "--"
        print(f"{label(p):26} {str(meta.get('seed')):>5} {meta['n_charts']:>7} "
              f"{r['n_overlaps']:>9} "
              f"{r['n_certified']:>4}/{r['n_overlaps']:<5} "
              f"{r['median_margin']:8.3f} {r['worst_margin']:8.4f}  "
              f"{str(cert['verdict']):15} {flag}")

    if not rows:
        return
    if out_dir:
        d = out_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = os.path.join(HERE, f"results_saved_{stamp}")
    os.makedirs(d, exist_ok=True)
    json.dump(rows, open(os.path.join(d, "results.json"), "w"), indent=2)

    print("\naggregate by experiment:")
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["experiment"]].append(r)
    for k, v in by.items():
        cert = sum(x["n_certified"] for x in v)
        tot = sum(x["n_overlaps"] for x in v)
        print(f"  {k:26} {cert:>4}/{tot:<5} overlaps certified "
              f"({100*cert/tot:5.1f}%)   median margin "
              f"{np.median([x['median_margin'] for x in v]):.3f}")
    print(f"\nwrote {d}/results.json")


if __name__ == "__main__":
    main()
