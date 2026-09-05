#!/usr/bin/env bash
# Experiment queue, run once the encoder cache exists.
#
#   ./run_experiments.sh cache_large runs_large google/siglip-large-patch16-256
#
# Order matters: each stage is only worth running if the one before it paid off,
# and the cheap diagnostic picks hard_weight before we spend a full tune on it.
set -u
CACHE=${1:-cache}
RUNS=${2:-runs}
ENC=${3:-google/siglip2-base-patch16-224}
export AHC_CACHE=$CACHE AHC_RUNS=$RUNS AHC_ENCODER=$ENC
LOG=$RUNS/experiments.log
mkdir -p "$RUNS"
say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }

say "config: cache=$CACHE runs=$RUNS encoder=$ENC"

say "1. visual centroids"
python -m ahc.centroids 2>&1 | grep -v "it/s" | tee -a "$LOG"

say "2. clip head (what)"
python -m ahc.train_clip_head --epochs 40 --seeds 3 2>&1 \
  | grep -E "seed|saved|samples|counts" | tee -a "$LOG"

say "3. window head, hard-negative sweep"
BEST_W=1.0; BEST_AUC=-1
for W in 1.0 3.0 10.0; do
  python -m ahc.train_head --epochs 30 --seeds 3 --hard-weight "$W" 2>&1 \
    | grep -E "saved|in-scene negatives" | tee -a "$LOG"
  A=$(python -m ahc.diagnose 2>/dev/null | grep "mean AUC" | awk '{print $3}')
  S=$(python -m ahc.diagnose 2>/dev/null | grep "^saturated" | awk '{print $6}')
  echo "  hard_weight=$W  mean_AUC=$A  saturated=$S" | tee -a "$LOG"
  if awk "BEGIN{exit !($A > $BEST_AUC)}"; then BEST_AUC=$A; BEST_W=$W; fi
done
echo "  best hard_weight=$BEST_W (mean AUC $BEST_AUC)" | tee -a "$LOG"
python -m ahc.train_head --epochs 30 --seeds 5 --hard-weight "$BEST_W" 2>&1 \
  | grep -E "saved" | tee -a "$LOG"

say "4. pairwise discriminators"
python -m ahc.pairwise --epochs 60 2>&1 | grep -vE "it/s" | tee -a "$LOG"

say "5. description bank"
python -m ahc.build_desc_bank 2>&1 | tail -2 | tee -a "$LOG"

say "6. tune decoder"
python -m ahc.tune 2>&1 | grep -E "^L|best|wrote|projected" | tee -a "$LOG"

say "7. score (cache mode, level-1 blend on)"
python - <<'PY'
import json, os
from pathlib import Path
r = Path(os.environ["AHC_RUNS"])
p = json.load(open(r / "params.json"))
p["level1"].update({"w_window": 0.2, "w_zeroshot": 0.4, "topk_frac": 0.15})
json.dump(p, open(r / "params_final.json", "w"), indent=2)
PY
python -m ahc.infer --params "$RUNS/params_final.json" --explain \
  --manifest manifest.json --out "$RUNS/sub_cached.json" 2>&1 | tail -3 | tee -a "$LOG"
python -m ahc.score "$RUNS/sub_cached.json" 2>&1 | tee -a "$LOG"

say "8. signal diagnostic"
python -m ahc.diagnose 2>&1 | tee -a "$LOG"

say "done - results in $LOG"
