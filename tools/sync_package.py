#!/usr/bin/env python3
"""Keep atlasae-package/ in sync with src/atlasae/.

The package is published standalone -- https://github.com/EduPH/atlasae, and
Zenodo -- but is developed here alongside the experiments, with
atlasae-package/ as the staging area for that repository.  Two copies drift
silently: an edit to src/atlasae/ that is not mirrored ships a stale package.
This script is the guard.

    python tools/sync_package.py --check     # report drift, exit 1 if any
    python tools/sync_package.py             # mirror src/ -> package/, report

Only .py files under src/atlasae/ are mirrored.  The package's own metadata
(pyproject.toml, README.md, CITATION.cff, .zenodo.json, LICENSE, examples/)
is maintained by hand and never touched here.

Run --check before tagging a release or uploading to Zenodo.
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "atlasae"
DST = ROOT / "atlasae-package" / "src" / "atlasae"


def compare():
    """Return (only_in_src, only_in_pkg, differing) as sorted name lists."""
    src = {p.name for p in SRC.glob("*.py")}
    dst = {p.name for p in DST.glob("*.py")}
    differing = sorted(
        n for n in src & dst
        if not filecmp.cmp(SRC / n, DST / n, shallow=False)
    )
    return sorted(src - dst), sorted(dst - src), differing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report drift without writing; exit 1 if out of sync")
    a = ap.parse_args()

    if not SRC.is_dir():
        sys.exit(f"source package not found: {SRC}")
    if not DST.is_dir():
        sys.exit(f"standalone package not found: {DST}")

    missing, extra, differing = compare()

    if not (missing or extra or differing):
        print(f"in sync: {len(list(SRC.glob('*.py')))} modules identical")
        return 0

    if a.check:
        print("OUT OF SYNC")
        for n in missing:
            print(f"  missing from package : {n}")
        for n in extra:
            print(f"  only in package      : {n}  (delete by hand if obsolete)")
        for n in differing:
            print(f"  differs              : {n}")
        print("\nrun `python tools/sync_package.py` to mirror src/ -> package/")
        return 1

    for n in missing + differing:
        shutil.copy2(SRC / n, DST / n)
        print(f"  copied {n}")
    for n in extra:
        print(f"  NOTE {n} exists only in the package; delete it by hand "
              f"if it is obsolete")

    missing, extra, differing = compare()
    if missing or differing:
        print("sync FAILED — files still differ")
        return 1
    print(f"synced: {len(missing) + len(differing)} pending, package now matches src/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
