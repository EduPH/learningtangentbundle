#!/usr/bin/env bash
#
# Run every experiment in the paper, in sequence.
#
#   ./run_all.sh                  # everything (~6-9 h)
#   ./run_all.sh --quick          # tiny budgets, ~15 min, to check it all works
#   ./run_all.sh --resume         # skip stages that already completed
#   ./run_all.sh --only s2,klein  # a subset (names below)
#   ./run_all.sh --list           # show stage names and exit
#
# Result directories are timestamped, so this never overwrites an earlier run:
# previous results stay put and the new ones land alongside.  Synthetic-manifold
# runs land in experiments/results/, which is where the table scripts look;
# see experiments/README.md for which run backs which table.  Trained atlases
# are saved under <results_dir>/atlases/.
#
# Each stage logs to logs/<stage>.log.  A stage that fails is reported and the
# run continues, so one broken stage does not cost the whole night.

set -uo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

MAT="$ROOT/experiments/E4_real_data/pointsCycloOctane.mat"
IMAGES="$ROOT/experiments/E4_real_data/vanhateren_iml"
SEEDS=5
QUICK=0
RESUME=0
ONLY=""
EXTRA=""

STAGES="s2 mobius klein rp2 rp2raw mobius_s1 t3 cyclooctane patches_model patches_real epsfloor_real epsfloor_model gauge signconstancy figures tables"

while [ $# -gt 0 ]; do
  case "$1" in
    --quick)  QUICK=1; SEEDS=1; shift ;;
    --resume) RESUME=1; shift ;;
    --only)   ONLY="$2"; shift 2 ;;
    --seeds)  SEEDS="$2"; shift 2 ;;
    --mat)    MAT="$2"; shift 2 ;;
    --images) IMAGES="$2"; shift 2 ;;
    --list)   echo "stages: $STAGES"; exit 0 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p logs .run_markers
[ $QUICK -eq 1 ] && EXTRA="--epochs 60"

wanted () {   # stage name -> 0 if it should run
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

STARTED=$(date +%s)
OK_LIST=""; FAIL_LIST=""; SKIP_LIST=""

run () {      # run <stage> <workdir> <command...>
  local name="$1" dir="$2"; shift 2
  wanted "$name" || return 0
  if [ $RESUME -eq 1 ] && [ -f ".run_markers/$name" ]; then
    echo "  [skip] $name (already done; delete .run_markers/$name to force)"
    SKIP_LIST="$SKIP_LIST $name"; return 0
  fi
  echo
  echo "=============================================================="
  echo "  $name        $(date '+%H:%M:%S')"
  echo "=============================================================="
  local t0 t1
  t0=$(date +%s)
  ( cd "$dir" && "$@" ) 2>&1 | tee "logs/$name.log" | grep -viE \
      "^\s*$|tensorflow|oneDNN|cuda|absl|tf-trt|WARNING|retracing" | tail -25
  local rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  if [ $rc -eq 0 ]; then
    touch ".run_markers/$name"
    printf "  -> %s OK (%dm %ds)\n" "$name" $(( (t1-t0)/60 )) $(( (t1-t0)%60 ))
    OK_LIST="$OK_LIST $name"
  else
    printf "  -> %s FAILED (rc=%d), see logs/%s.log\n" "$name" "$rc" "$name"
    FAIL_LIST="$FAIL_LIST $name"
  fi
}

echo "Fiber-bundle autoencoder: full experiment run"
echo "  seeds=$SEEDS  quick=$QUICK  resume=$RESUME  only=${ONLY:-all}"
echo "  cyclo-octane data: $MAT"
echo "  van Hateren images: $IMAGES"
[ -f "$MAT" ]     || echo "  !! missing $MAT -- cyclooctane will fail"
[ -d "$IMAGES" ]  || echo "  !! missing $IMAGES -- real patches will fail"

# ---- 1. synthetic manifolds ------------------------------------------------
# S2 uses 4000 points: ample for the coboundary test at 1000, but the
# sign-constancy margin scales as sqrt(n) and needs the denser sample
# (27% of overlaps certified at n=1000, 100% at n=4000).
run s2     experiments python paper_experiments.py --manifold S2 --n-points 4000 --seeds "$SEEDS" $EXTRA
run mobius experiments python paper_experiments.py --manifold Mobius --seeds "$SEEDS" $EXTRA
# Klein at 4000 too, but for the opposite reason: its shortfall is
# conditioning, not density, and showing that it still fails at 4x the data
# is a stronger statement than showing it fails at 1000 points.
run klein  experiments python paper_experiments.py --manifold Klein --n-points 4000 --seeds "$SEEDS" $EXTRA
run rp2    experiments python paper_experiments.py --manifold RP2    --seeds "$SEEDS" $EXTRA
run rp2raw experiments python paper_experiments.py --manifold RP2    --seeds "$SEEDS" --raw-patches $EXTRA
# d=3 pair: Mobius x S1 (non-orientable; one certified odd cycle suffices)
# and the flat 3-torus (orientable control; needs full margin coverage, so
# it may end correct-but-uncertified at this density -- see the paper).
run mobius_s1 experiments python paper_experiments.py --manifold MobiusS1 --seeds "$SEEDS" $EXTRA
run t3        experiments python paper_experiments.py --manifold T3       --seeds "$SEEDS" $EXTRA

# ---- 2. real data ----------------------------------------------------------
run cyclooctane   experiments/E4_real_data python cyclooctane.py --mat "$MAT" --seeds "$SEEDS" $EXTRA
run patches_model experiments/E4_real_data python natural_patches.py --synthetic --seeds "$SEEDS" $EXTRA
run patches_real  experiments/E4_real_data python natural_patches.py --images "$IMAGES" --seeds "$SEEDS" $EXTRA

# ---- 3. the epsilon floor (supports the "thickness" claim) -----------------
if [ $QUICK -eq 1 ]; then
  run epsfloor_model experiments/E4_real_data python eps_floor_sweep.py --synthetic --quick
else
  run epsfloor_real  experiments/E4_real_data python eps_floor_sweep.py --images "$IMAGES" --seeds 3
  run epsfloor_model experiments/E4_real_data python eps_floor_sweep.py --synthetic --seeds 3
fi

# ---- 4. delta: gauge fixing, and sign constancy ----------------------------
run gauge experiments python E6_gauge/run_gauge.py \
    --manifolds S2 Mobius RP2 --seeds 42 43 44 45 46 $EXTRA
if [ $QUICK -eq 1 ]; then
  run signconstancy experiments python E7_signconstancy/run_signconstancy.py \
      --manifold S2 --n 500 --epochs 60
else
  run signconstancy experiments python E7_signconstancy/run_signconstancy.py \
      --manifold S2 Mobius Klein --n 1000 2000 4000 --epochs 4000
fi

# ---- 5. figures and tables -------------------------------------------------
figures () {
  set -e
  cd "$ROOT/experiments"
  python E1_eta_codim/eta_validation.py \
      --fig ../Fcom__Atlas_Autoencoders/eta_fig/eta_validation.png
  # atlas figures (data | latent charts + signed nerve | reconstruction),
  # regenerated from the newest saved atlas of each experiment
  AF=../Fcom__Atlas_Autoencoders/atlas_fig
  # seeds are pinned to the ones named in the figure captions
  for spec in "S2_seed42 panel atlas_S2" "Mobius_seed42 panel atlas_Mobius" \
              "Klein_seed43 panel atlas_Klein" "RP2_seed42 panel atlas_RP2" \
              "MobiusS1_seed42 panel atlas_MobiusS1" "T3_seed42 panel atlas_T3" \
              "Mobius_seed42 pipeline pipeline_mobius"; do
    set -- $spec
    atlas=$(ls -dt results/results_paper_*/atlases/$1 2>/dev/null | head -1)
    [ -n "$atlas" ] && python make_atlas_figs.py "$atlas" --$2 --out "$AF/$3.png"
  done
  python make_rp2_embedding_fig.py \
      --out ../Fcom__Atlas_Autoencoders/cyclo_fig/rp2_embedding.png
  cd E4_real_data
  python make_teaser.py
  python make_transition_fig.py --epochs $([ $QUICK -eq 1 ] && echo 60 || echo 4000)
  # these write into experiments/E4_real_data/cyclo_fig/, not into the paper
  cp cyclo_fig/teaser.png cyclo_fig/transitions.png \
     "$ROOT/Fcom__Atlas_Autoencoders/cyclo_fig/"
  # Figure 2 likewise reads from the paper folder: take the van Hateren run
  # (synthetic=false), which is the newest patch run without --synthetic
  vh=$(for d in $(ls -dt results_patches_*/ 2>/dev/null); do
         python3 -c "import json,sys;r=json.load(open('$d/results.json'));\
                     sys.exit(0 if not r[0].get('synthetic',False) else 1)" \
           2>/dev/null && { echo "$d"; break; }
       done)
  [ -n "$vh" ] && cp "$vh"/core_patches.png "$vh"/nerve_signs.png \
                     "$ROOT/Fcom__Atlas_Autoencoders/patches_fig/"
}
export -f figures; export ROOT QUICK
run figures "$ROOT" bash -c figures

tables () {
  set -e
  cd "$ROOT/experiments"
  echo "----- tab:summary_all (main.tex) -----"
  python make_master_table.py --summary 2>/dev/null
  echo; echo "----- tab:master (supplement.tex) -----"
  python make_master_table.py --latex 2>/dev/null
  echo; echo "----- tab:theta (supplement.tex) -----"
  python E5_theta/theta_certificate.py --latex 2>/dev/null
  echo; echo "----- console summary -----"
  python make_master_table.py
}
export -f tables
run tables "$ROOT" bash -c tables

# ---- summary ---------------------------------------------------------------
ELAPSED=$(( $(date +%s) - STARTED ))
echo
echo "=============================================================="
printf "  done in %dh %dm\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60))
echo "=============================================================="
echo "  ok:     ${OK_LIST:- none}"
echo "  skipped:${SKIP_LIST:- none}"
echo "  FAILED: ${FAIL_LIST:- none}"
echo
echo "  table bodies are in logs/tables.log -- paste over the blocks marked"
echo "  '% generated by ... -- do not edit by hand'"
echo
echo "  previous results were not overwritten; compare before adopting:"
echo "     cd experiments && python make_master_table.py --strict"
[ -z "$FAIL_LIST" ] || exit 1
