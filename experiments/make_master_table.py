"""
Master diagnostic table over every experiment.

For each run reports: the certificate quantities, whether the run is
certified, whether the classification is correct, and -- when a run is not
certified -- exactly which condition failed.

Certificate (all three required):
    C1  eps    <= EPS_MAX        sup reconstruction error
    C2  #{i : eta_pca^(i) > 1} <= 1   at most one chart outside the
                                 differential-error bound (the local theorem
                                 tolerates a single outlier via re-indexing)
    C3  delta  >  DELTA_MIN      non-degeneracy gap

A run may additionally return "undetermined": the coboundary test could not
be evaluated because the sign cocycle failed verification on the nerve. That
is not a certificate condition but is reported, since such a run yields no
classification at all.

Usage:
    python make_master_table.py                 # console table
    python make_master_table.py --latex         # LaTeX rows for the paper
"""

import argparse
import glob
import json
import os

import sys

import numpy as np

EPS_MAX, DELTA_MIN = 0.15, 0.005
HERE = os.path.dirname(os.path.abspath(__file__))

# experiment name -> (glob for results file, ground-truth orientability)
# ground truth None means "per-record" (cyclo-octane: depends on the stratum)
SOURCES = [
    ("$S^2$",                     "results/results_paper_*/S2.json",                 True),
    ("M\\\"obius band",           "results/results_paper_*/Mobius.json",             False),
    ("Klein bottle ($\\R^4$)",    "results/results_paper_*/Klein.json",              False),
    ("$\\mathbb{R}P^2$ raw",      "results/results_paper_*/RP2_raw.json",            False),
    ("$\\mathbb{R}P^2$ norm.",    "results/results_paper_*/RP2.json",                False),
    ("M\\\"obius band $\\times\\, S^1$", "results/results_paper_*/MobiusS1.json",    False),
    ("$T^3$",                     "results/results_paper_*/T3.json",                 True),
    ("image patches (model)",     ("E4_real_data/results_patches_*/results.json", "model"), False),
    ("image patches (van Hateren)", ("E4_real_data/results_patches_*/results.json", "real"), False),
    ("cyclo-octane",              "E4_real_data/results_cyclooctane_*/results.json", None),
]


def newest(pattern):
    hits = sorted(glob.glob(os.path.join(HERE, pattern)), key=os.path.getmtime)
    return hits[-1] if hits else None


def _is_synthetic(recs):
    """True if these records come from the analytic patch model."""
    if 'synthetic' in recs[0]:
        return bool(recs[0]['synthetic'])
    # runs predating the marker: the analytic model is generated at 3000 points
    # and reconstructs an order of magnitude better than the photographic data
    return recs[0].get('n_points') == 3000


EXPECTED_TRIALS = 5      # seeds per experiment (cyclo-octane is 5 per stratum)


def _load(pattern, expected=EXPECTED_TRIALS, strict=False):
    """pattern is a glob, or (glob, "model"/"real") to pick a patch variant.

    Picks the most recent matching file.  A run still in progress writes its
    results incrementally, so the newest file may hold fewer than `expected`
    trials; that would silently shrink a row and the totals.  We warn, and
    under `strict` fall back to the most recent *complete* run instead.
    """
    want = None
    if isinstance(pattern, tuple):
        pattern, want = pattern
    hits = sorted(glob.glob(os.path.join(HERE, pattern)), key=os.path.getmtime,
                  reverse=True)
    candidates = []
    for p in hits:
        r = json.load(open(p))
        r = r if isinstance(r, list) else [r]
        if want is not None and (want == "model") != _is_synthetic(r):
            continue
        candidates.append((r, os.path.relpath(p, HERE)))
    if not candidates:
        return None, None

    r, path = candidates[0]
    n = len(r) if "cyclooctane" not in path else len(r) // 4
    if n < expected:
        complete = next((c for c in candidates[1:]
                         if len(c[0]) >= expected), None)
        msg = (f"[warn] {path}: {n} of {expected} trials -- run still in "
               f"progress?")
        if strict and complete is not None:
            print(f"{msg}  falling back to {complete[1]}", file=sys.stderr)
            return complete
        print(msg, file=sys.stderr)
    return r, path



def _margin_index():
    """{(run_dir, seed, stratum): verdict-certificate row} from the newest
    E7_signconstancy/results_saved_* evaluation.

    This is the third condition of def:certificate.  It replaces the calibrated
    delta threshold used previously: delta only asserts that |det g_ji| avoids
    zero *at the sampled points*, whereas the margin proves the sign constant
    between them, and prop:odd-cycle turns per-overlap constancy into a
    certified verdict.
    """
    hits = sorted(glob.glob(os.path.join(
        HERE, "E7_signconstancy", "results_*", "results.json")),
        key=os.path.getmtime, reverse=True)
    idx = {}
    for h in hits:                     # newest wins; older only fill gaps
        for row in json.load(open(h)):
            # only evaluations that carry the verdict certificate; earlier
            # runs recorded per-overlap margins alone and must not be mixed in
            if "verdict_certified" not in row:
                continue
            rel = row.get("path", "")
            parts = rel.split(os.sep)
            if "atlases" not in parts:
                continue
            run = os.sep.join(parts[:parts.index("atlases")])
            name = parts[-1]
            # atlas dirs are '<tag>_seed42' (synthetic, cyclo strata) or bare
            # 'seed42' (the patch runs); handle both
            prefix, sep, seed = name.rpartition("_seed")
            if not sep:
                prefix, seed = "", name[4:] if name.startswith("seed") else ""
            if not seed.isdigit():
                continue
            stratum = None
            if prefix.startswith("stratum_"):
                stratum = int(prefix.split("_")[1])
            idx.setdefault((run, int(seed), stratum), row)
    return idx


_MARGINS = None


def margin_row(run_dir, rec):
    global _MARGINS
    if _MARGINS is None:
        _MARGINS = _margin_index()
    seed = rec.get("seed")
    for stratum in (rec.get("stratum_index"), None):
        hit = _MARGINS.get((run_dir, seed, stratum))
        if hit is not None:
            return hit
    return None



def margin_totals(run_dir, records):
    """(certified components, total components) over the trials of one run."""
    c = t = 0
    for r in records:
        m = margin_row(run_dir, r)
        if m:
            c += m["n_certified"]
            t += m["n_overlaps"]
    return c, t


def truth_of(rec, default):
    """Ground truth for one record; cyclo-octane depends on the stratum."""
    if default is not None:
        return default
    return rec.get("stratum_index", 0) != 0   # stratum 0 is the Klein piece


def failure_reasons(rec, run_dir=None):
    """The three conditions of def:certificate, in order."""
    out = []
    if rec["varepsilon"] > EPS_MAX:
        out.append(f"$\\varepsilon$={rec['varepsilon']:.2f}")
    if rec.get("n_eta_outliers", 0) > 1:
        out.append(f"{rec['n_eta_outliers']} charts $\\eta>1$")
    m = margin_row(run_dir, rec) if run_dir else None
    if m is None:
        out.append("margin not evaluated")
    elif not m.get("verdict_certified"):
        out.append(f"margin ({m['n_certified']}/{m['n_overlaps']} overlaps)")
    return out


def analyse(records, default_truth, run_dir=None):
    rows = []
    for r in records:
        orientable_truth = truth_of(r, default_truth)
        verdict = r["verdict"]
        reasons = failure_reasons(r, run_dir)
        certified = (len(reasons) == 0)
        if verdict == "undetermined":
            correct = None
        else:
            correct = ((verdict == "orientable") == orientable_truth)
        rows.append(dict(rec=r, certified=certified, correct=correct,
                         reasons=reasons, verdict=verdict))
    return rows


def summarise(rows):
    n = len(rows)
    cert = [x for x in rows if x["certified"]]
    det = [x for x in rows if x["correct"] is not None]
    return dict(
        n=n, n_cert=len(cert), n_undet=n - len(det),
        n_correct=sum(1 for x in det if x["correct"]),
        n_det=len(det),
        n_cert_correct=sum(1 for x in cert if x["correct"]),
        n_cert_wrong=sum(1 for x in cert if x["correct"] is False),
        eps=np.mean([x["rec"]["varepsilon"] for x in rows]),
        eta=np.mean([x["rec"]["eta_pca"] for x in rows]),
        delta=np.mean([x["rec"]["delta"] for x in rows]),
        reasons=[r for x in rows if not x["certified"] for r in x["reasons"]],
    )


# experiment -> (d, ambient, truth string) for the main-text summary table.
# cyclo-octane is split by stratum there, so it is handled separately.
SUMMARY_META = {
    "$S^2$":                       ("2", "$\\R^3$",     "Orient."),
    "M\\\"obius band":             ("2", "$\\R^3$",     "Non-orient."),
    "Klein bottle ($\\R^4$)":      ("2", "$\\R^4$",     "Non-orient."),
    "$\\mathbb{R}P^2$ raw":        ("2", "$\\R^{100}$", "Non-orient."),
    "$\\mathbb{R}P^2$ norm.":      ("2", "$\\R^{100}$", "Non-orient."),
    "M\\\"obius band $\\times\\, S^1$": ("3", "$\\R^5$", "Non-orient."),
    "$T^3$":                       ("3", "$\\R^6$",     "Orient."),
    "image patches (model)":       ("2", "$\\R^{8}$",   "Non-orient."),
    "image patches (van Hateren)": ("2", "$\\R^{8}$",   "Non-orient."),
}

SUMMARY_NAME = {
    "$\\mathbb{R}P^2$ raw":   "$\\mathbb{R}P^2$, raw",
    "$\\mathbb{R}P^2$ norm.": "$\\mathbb{R}P^2$, normalised",
    "Klein bottle ($\\R^4$)": "Klein bottle",
}


def summary_table():
    """LaTeX body of the main-text summary table (tab:summary_all).

    Kept in the same script as tab:master so the two cannot drift apart.
    'Detect' counts runs returning the *correct* verdict; undetermined runs
    count as non-detections, so Detect + undetermined <= trials.
    """
    print("% generated by make_master_table.py --summary -- do not edit by hand")
    print("\\multicolumn{8}{c}{\\textit{Synthetic manifolds}} \\\\")
    for name, pattern, truth in SOURCES:
        if name not in SUMMARY_META:
            continue
        recs, path = load(pattern)
        if recs is None:
            continue
        s = summarise(analyse(recs, truth, os.path.dirname(path)))
        d, amb, tr = SUMMARY_META[name]
        disp = SUMMARY_NAME.get(name, name)
        if name == "image patches (van Hateren)":
            continue  # printed in the real-data block below
        mc, mt = margin_totals(os.path.dirname(path), recs)
        print(f"{disp} & {d} & {amb} & {tr} & {s['eps']:.3f} & "
              f"{s['n_cert']}/{s['n']} & {s['n_correct']}/{s['n']} & "
              f"${mc}/{mt}$ \\\\")

    print("\\midrule")
    print("\\multicolumn{8}{c}{\\textit{Real data}} \\\\")

    # cyclo-octane, split into the Klein stratum and the three sphere strata
    recs, cpath = load("E4_real_data/results_cyclooctane_*/results.json")
    if recs is not None:
        klein = [r for r in recs if r.get("stratum_index", 0) == 0]
        sphere = [r for r in recs if r.get("stratum_index", 0) != 0]
        for lbl, sub, tr in (("cyclo-octane, Klein stratum", klein, "Non-orient."),
                             ("cyclo-octane, sphere ($\\times3$)", sphere, "Orient.")):
            s = summarise(analyse(sub, None, os.path.dirname(cpath)))
            mc, mt = margin_totals(os.path.dirname(cpath), sub)
            print(f"{lbl} & 2 & $\\R^{{24}}$ & {tr} & {s['eps']:.3f} & "
                  f"{s['n_cert']}/{s['n']} & {s['n_correct']}/{s['n']} & "
                  f"${mc}/{mt}$ \\\\")

    for name, pattern, truth in SOURCES:
        if name != "image patches (van Hateren)":
            continue
        recs, path = load(pattern)
        if recs is None:
            continue
        s = summarise(analyse(recs, truth, os.path.dirname(path)))
        d, amb, tr = SUMMARY_META[name]
        mc, mt = margin_totals(os.path.dirname(path), recs)
        print(f"{name} & {d} & {amb} & {tr} & {s['eps']:.3f} & "
              f"{s['n_cert']}/{s['n']} & {s['n_correct']}/{s['n']} & "
              f"${mc}/{mt}$ \\\\")


def main(latex=False):
    blocks = []
    for name, pattern, truth in SOURCES:
        recs, path = load(pattern)
        if recs is None:
            print(f"[skip] {name}: no results yet ({pattern})")
            continue
        rows = analyse(recs, truth, os.path.dirname(path))
        blocks.append((name, path, rows, summarise(rows)))

    if not latex:
        print(f"\ncertificate: eps<={EPS_MAX}, at most 1 chart with eta_pca>1, "
              f"and a certified verdict (sign-constancy margin)\n")
        hdr = (f"{'experiment':26} {'runs':>4} {'cert':>5} {'correct':>8} "
               f"{'cert&ok':>8} {'BAD':>4} {'eps':>6} {'eta':>6} {'delta':>7}")
        print(hdr); print("-" * len(hdr))
        for name, path, rows, s in blocks:
            plain = name.replace("$", "").replace("\\\"o", "o").replace("\\R", "R") \
                        .replace("\\mathbb{R}P^2", "RP2").replace("^4", "4")
            print(f"{plain:26} {s['n']:>4} {s['n_cert']:>5} "
                  f"{s['n_correct']}/{s['n_det']:<6} {s['n_cert_correct']}/{s['n_cert']:<6} "
                  f"{s['n_cert_wrong']:>4} {s['eps']:6.3f} {s['eta']:6.2f} {s['delta']:7.3f}")
        tot = dict(n=0, c=0, ok=0, det=0, cok=0, bad=0, und=0)
        for _, _, _, s in blocks:
            tot['n'] += s['n']; tot['c'] += s['n_cert']; tot['ok'] += s['n_correct']
            tot['det'] += s['n_det']; tot['cok'] += s['n_cert_correct']
            tot['bad'] += s['n_cert_wrong']; tot['und'] += s['n_undet']
        print("-" * len(hdr))
        print(f"{'TOTAL':26} {tot['n']:>4} {tot['c']:>5} {tot['ok']}/{tot['det']:<6} "
              f"{tot['cok']}/{tot['c']:<6} {tot['bad']:>4}")
        print(f"\ncertified but INCORRECT: {tot['bad']}   |   undetermined: {tot['und']}")
        print("\nreasons for non-certification (per run):")
        for name, path, rows, s in blocks:
            bad = [x for x in rows if not x["certified"]]
            if not bad:
                continue
            plain = name.replace("$", "")
            for x in bad:
                rs = ", ".join(r.replace("$", "").replace("\\varepsilon", "eps")
                               .replace("\\delta", "delta").replace("\\eta", "eta")
                               for r in x["reasons"])
                print(f"  {plain:24} seed {x['rec'].get('seed', '-')}"
                      f"{' ' + x['rec'].get('stratum', '') if 'stratum' in x['rec'] else ''}"
                      f": {rs}   (verdict {x['verdict']}, "
                      f"{'correct' if x['correct'] else 'incorrect' if x['correct'] is not None else 'n/a'})")
        print("\nsources:")
        for name, path, _, _ in blocks:
            print(f"  {name.replace('$','')}: {path}")
        return

    # ---- LaTeX ----
    print("% generated by make_master_table.py")
    for name, path, rows, s in blocks:
        det = f"{s['n_correct']}/{s['n_det']}"
        cert = f"{s['n_cert']}/{s['n']}"
        cok = f"{s['n_cert_correct']}/{s['n_cert']}" if s['n_cert'] else "--"
        reasons = s['reasons']
        rtxt = "---" if not reasons else ", ".join(sorted(set(
            ("$\\varepsilon$" if "varep" in r else
             "$\\eta$" if "eta" in r or "\\eta" in r else "margin")
            for r in reasons)))
        print(f"{name} & {s['n']} & {s['eps']:.3f} & {s['eta']:.2f} & "
              f"{s['delta']:.3f} & {cert} & {det} & {cok} & {rtxt} \\\\")




# short experiment names for the per-trial table (fits a longtable row)
PERTRIAL_NAME = {
    "$S^2$": "$S^2$",
    "M\\\"obius band": "M\\\"obius band",
    "Klein bottle ($\\R^4$)": "Klein bottle $\\R^4$",
    "$\\mathbb{R}P^2$ raw": "$\\mathbb{R}P^2$ raw",
    "$\\mathbb{R}P^2$ norm.": "$\\mathbb{R}P^2$ norm.",
    "M\\\"obius band $\\times\\, S^1$": "M\\\"ob.\\ $\\times S^1$",
    "$T^3$": "$T^3$",
    "image patches (model)": "patches (model)",
    "image patches (van Hateren)": "patches (vH)",
    "cyclo-octane": "cyclo-octane",
}


def pertrial_table():
    """LaTeX longtable body: one row per trial, every certificate quantity.

    Columns: experiment, seed, eps, eta_pca, #charts eta>1, delta,
    margin-certified components / total, verdict, correct, certified,
    failed certificate conditions.
    """
    print("% generated by make_master_table.py --pertrial -- do not edit by hand")
    total = 0
    for name, pattern, truth in SOURCES:
        recs, path = load(pattern)
        if recs is None:
            continue
        run_dir = os.path.dirname(path)
        rows = analyse(recs, truth, run_dir)
        for x in sorted(rows, key=lambda r: (r["rec"].get("stratum_index", 0),
                                             r["rec"].get("seed", 0))):
            r = x["rec"]
            label = PERTRIAL_NAME.get(name, name)
            if r.get("stratum") is not None:
                tag = ("K" if r.get("stratum_index", 0) == 0
                       else f"S$_{r['stratum_index']}$")
                label = f"cyclo-octane {tag}"
            m = margin_row(run_dir, r)
            marg = f"{m['n_certified']}/{m['n_overlaps']}" if m else "--"
            verdict = {"orientable": "O", "non-orientable": "N",
                       "undetermined": "U"}[x["verdict"]]
            corr = ("\\cmark" if x["correct"] else
                    "---" if x["correct"] is None else "\\xmark")
            cert = "\\cmark" if x["certified"] else "\\xmark"
            reasons = ("---" if not x["reasons"] else ", ".join(sorted(set(
                ("$\\varepsilon$" if "varep" in t else
                 "$\\eta$" if "eta" in t or "\\eta" in t else "margin")
                for t in x["reasons"]))))
            print(f"{label} & {r['seed']} & {r['varepsilon']:.3f} & "
                  f"{r['eta_pca']:.2f} & {r.get('n_eta_outliers', 0)} & "
                  f"{r['delta']:.3f} & {marg} & {verdict} & {corr} & "
                  f"{cert} & {reasons} \\\\")
            total += 1
    print(f"% rows: {total}")


STRICT = False


def load(pattern, expected=EXPECTED_TRIALS):
    """See _load.  Honours the module-level STRICT flag (--strict)."""
    return _load(pattern, expected, STRICT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--summary", action="store_true",
                    help="LaTeX body of the main-text summary table")
    ap.add_argument("--pertrial", action="store_true",
                    help="LaTeX longtable body: one row per trial")
    ap.add_argument("--strict", action="store_true",
                    help="ignore in-progress runs; use the newest complete one")
    a = ap.parse_args()
    globals()["STRICT"] = a.strict
    if a.pertrial:
        pertrial_table()
    elif a.summary:
        summary_table()
    else:
        main(latex=a.latex)
