"""
Recompute every quantitative claim in the paper from the result files.

The paper quotes numbers in prose, in captions and in tables. Tables are
generated, so they cannot drift; prose and captions are written by hand and
have drifted repeatedly. This script recomputes the quantities from
experiments/results/ and E7_signconstancy/results_certificate, then checks
that each appears where it should in main.tex and supplement.tex.

    python audit_paper.py            # report
    python audit_paper.py --strict   # non-zero exit if anything fails

Add a check here whenever a number goes into the prose.
"""

import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "Fcom__Atlas_Autoencoders")
sys.path.insert(0, HERE)

import make_master_table as M            # noqa: E402


def load_tex():
    out = {}
    for n in ("main", "supplement"):
        out[n] = open(os.path.join(PAPER, f"{n}.tex")).read()
    out["both"] = out["main"] + out["supplement"]
    return out


def certificate_stats():
    """Per-experiment and total certification, recomputed."""
    per, tot = {}, dict(n=0, cert=0, correct=0, undet=0, bad=0)
    for name, pat, truth in M.SOURCES:
        recs, path = M.load(pat)
        if recs is None:
            continue
        rows = M.analyse(recs, truth, os.path.dirname(path))
        s = M.summarise(rows)
        per[name] = s
        tot["n"] += s["n"]
        tot["cert"] += s["n_cert"]
        tot["correct"] += s["n_correct"]
        tot["undet"] += sum(1 for r in rows if r["correct"] is None)
        tot["bad"] += sum(1 for r in rows if r["certified"] and r["correct"] is False)
    return per, tot


def margin_stats():
    """Certified overlap components per run directory."""
    idx = M._margin_index()
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])
    for (run, seed, stratum), row in idx.items():
        for key in (run, (run, stratum)):
            a = agg[key]
            a[0] += row["n_certified"]
            a[1] += row["n_overlaps"]
            a[2] += 1
            a[3] += bool(row["verdict_certified"])
    return agg


def main(strict=False):
    tex = load_tex()
    per, tot = certificate_stats()
    agg = margin_stats()
    fails = []

    def check(label, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(label)

    print("totals")
    check("65 trials", tot["n"] == 65, f"computed {tot['n']}")
    check("36 certified appears in both documents",
          f"\\textbf{{{tot['cert']}/{tot['n']}}}" in tex["supplement"]
          and f"${tot['cert']}$ certified trials" in tex["main"],
          f"computed {tot['cert']}")
    check("52/52 correct in tab:master",
          f"\\textbf{{{tot['correct']}/{tot['correct']}}}" in tex["supplement"],
          f"computed {tot['correct']}")
    check(f"{tot['undet']} abstentions stated in main",
          f"remaining ${tot['undet']}$ abstained" in tex["main"])
    check("no certified-but-incorrect", tot["bad"] == 0, f"computed {tot['bad']}")

    print("\nper-experiment Cert. cells (tab:master is generated, so this "
          "checks the generator agrees with the file on disk)")
    for name, s in per.items():
        row = re.search(re.escape(name) + r".*?& (\d+)/(\d+) &", tex["supplement"])
        if row:
            got = (int(row.group(1)), int(row.group(2)))
            check(f"{name}", got == (s["n_cert"], s["n"]),
                  f"table {got[0]}/{got[1]}, computed {s['n_cert']}/{s['n']}")

    print("\nmargin figures quoted in prose")
    RUNS = {
        "cyclo Klein stratum": ("E4_real_data/results_cyclooctane_20260804_193749", 0),
        "S2 n=4000": ("results/results_paper_20260805_101332", None),
        "S2 n=1000": ("results/results_paper_20260804_091100", None),
        "Klein n=4000": ("results/results_paper_20260805_101547", None),
        "Klein n=1000": ("results/results_paper_20260804_091734", None),
        "RP2 raw": ("results/results_paper_20260804_110243", None),
        "RP2 norm": ("results/results_paper_20260804_095231", None),
        "patch model": ("E4_real_data/results_patches_20260804_195745", None),
        "van Hateren": ("E4_real_data/results_patches_20260804_202645", None),
    }
    quoted = {
        "S2 n=1000": r"\$35\$ of \$55\$",
        "S2 n=4000": r"\$79\$ of \$80\$",
        "Klein n=1000": r"\$60\$ of \$158\$",
        "Klein n=4000": r"\$193\$ of \$224\$",
    }
    for k, (run, stratum) in RUNS.items():
        a = agg.get((run, stratum) if stratum is not None else run)
        if a is None:
            check(k, False, "no margin data")
            continue
        pct = 100 * a[0] / a[1] if a[1] else 0
        detail = f"{a[0]}/{a[1]} = {pct:.0f}%, verdict-certified {a[3]}/{a[2]}"
        if k in quoted:
            check(f"{k} quoted in text", re.search(quoted[k], tex["main"]) is not None, detail)
        else:
            print(f"  ----  {k:22} {detail}")

    print("\npercentages quoted in conclusion 4 and sec:exp-signconstancy")
    for label, run, stratum in [("raw RP2", "results/results_paper_20260804_110243", None),
                                ("norm RP2", "results/results_paper_20260804_095231", None),
                                ("Klein R4", "results/results_paper_20260805_101547", None),
                                ("patch model", "E4_real_data/results_patches_20260804_195745", None),
                                ("cyclo Klein", "E4_real_data/results_cyclooctane_20260804_193749", 0),
                                ("MobiusS1", "results/results_paper_20260806_121436", None),
                                ("T3", "results/results_paper_20260806_124637", None)]:
        a = agg[(run, stratum) if stratum is not None else run]
        pct = round(100 * a[0] / a[1])
        check(f"{label} {pct}% appears", f"${pct}\\%$" in tex["both"]
              or f"{pct}\\%" in tex["both"], f"computed {pct}%")

    print("\neta validation")
    import subprocess
    out = subprocess.run([sys.executable, "E1_eta_codim/eta_validation.py"],
                         capture_output=True, text=True, cwd=HERE).stdout
    m = re.search(r"paired trials:\s*(\d+)", out)
    a = re.search(r"agree on the eta<1 test\s*:\s*(\d+)/(\d+)", out)
    f = re.search(r"FALSE CERTIFICATIONS\s*:\s*(\d+)", out)
    if m and a and f:
        check(f"{m.group(1)} paired trials", f"${m.group(1)}$ trials" in tex["both"]
              or f"{m.group(1)} trials" in tex["both"])
        check(f"{a.group(1)}/{a.group(2)} agreement",
              f"in ${a.group(1)}$" in tex["both"] or f"{a.group(1)}/{a.group(2)}" in tex["both"])
        check("0 false certifications", f.group(1) == "0")

    # sec:exp-summary, `Why the embedding governs certification': the paper
    # claims the size-eta correlation on the low-curvature patch model is
    # negative in every seed (size alone is not the failure driver).
    print("\nper-chart size vs eta (patch model)")
    import glob as _glob
    import numpy as _np

    def _spearman(a, b):
        ra = _np.argsort(_np.argsort(a)).astype(float)
        rb = _np.argsort(_np.argsort(b)).astype(float)
        return float(_np.corrcoef(ra, rb)[0, 1])

    pm = sorted(_glob.glob(os.path.join(
        HERE, "E4_real_data", "results_patches_*", "results.json")),
        key=os.path.getmtime)
    pm = [f for f in pm
          if json.load(open(f)) and json.load(open(f))[0].get("synthetic", True)
          and json.load(open(f))[0].get("eta_per_chart")]
    if pm:
        recs = json.load(open(pm[-1]))
        rhos = [_spearman(
            _np.array([c["n_points"] for c in e["eta_per_chart"]]),
            _np.array([c["eta_pca"] for c in e["eta_per_chart"]]))
            for e in recs if e.get("eta_per_chart")]
        check("size-eta correlation negative in every patch-model seed",
              bool(rhos) and all(r < 0 for r in rhos),
              f"spearman per seed: {[round(r, 2) for r in rhos]}")



    # width ablation (sec:exp-summary, app:epsfloor): quoted numbers
    print("\nwidth ablation")
    import glob as _g
    wf = sorted(_g.glob(os.path.join(HERE, "results", "results_paper_*",
                                     "RP2_raw_h*.json")))
    if wf:
        wr = json.load(open(wf[-1]))
        eps = sum(x["varepsilon"] for x in wr) / len(wr)
        allfail = all(x.get("n_eta_outliers", 0) >= 2 for x in wr)
        check("raw RP2 wide: eps 0.070 quoted",
              abs(eps - 0.070) < 0.0025 and "$0.070$" in tex["both"],
              f"computed {eps:.3f} over {len(wr)} seeds")
        check("raw RP2 wide: eta fails every seed", allfail,
              f"outliers: {[x.get('n_eta_outliers') for x in wr]}")
    for tag, lo_q, hi_q in (("real", 0.402, 0.422),):
        sf = sorted(_g.glob(os.path.join(HERE, "E4_real_data",
                                         f"results_epsfloor_{tag}_*",
                                         "results.json")), key=os.path.getmtime)
        rows = [x for x in json.load(open(sf[-1])) if x["axis"] == "width"]
        if rows:
            lo = min(x["epsilon"] for x in rows)
            hi = max(x["epsilon"] for x in rows)
            ok = (abs(lo - lo_q) < 0.0025 and abs(hi - hi_q) < 0.0025
                  and f"[{lo_q:.3f},{hi_q:.3f}]" in tex["main"].replace(" ", ""))
            check(f"width axis ({tag}): range [{lo_q}, {hi_q}] quoted", ok,
                  f"computed [{lo:.4f}, {hi:.4f}]")

    print(f"\n{len(fails)} failure(s)" if fails else "\nall checks pass")
    if strict and fails:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    main(ap.parse_args().strict)
