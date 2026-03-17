#!/bin/sh

# Paths
SCRIPT_DIR=$(dirname "$0")
PARAM_DIR="EXP_DIR/HW2_EXP3"
OUT_DIR="OUTPUT_DIR/HW2-EXP3"
EXP1_DIR="OUTPUT_DIR/HW2-EXP1"

QREL="INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed"
TREC="INPUT_DIR/trec_eval-9.0.4"

# 1. Prepare Output Directory and Baseline
mkdir -p "$OUT_DIR"

# Copy Exp-1.2b per_query file to use as Baseline for W/T/L
BASELINE_SRC="$EXP1_DIR/HW2-Exp-1.2b.per_query"
if [ -f "$BASELINE_SRC" ]; then
    cp "$BASELINE_SRC" "$OUT_DIR/"
    echo "Baseline (Exp-1.2b) copied successfully."
else
    echo "Error: Baseline $BASELINE_SRC not found. Run Experiment 1 first."
    exit 1
fi

echo "=============================================="
echo "Starting HW2 Experiment 3 (Custom)"
echo "=============================================="

# 2. Run Loop (3.1b -> 3.1f)
for param in "$PARAM_DIR"/HW2-Exp-3.1*.param
do
  exp=$(basename "$param" .param)
  echo "----------------------------------------------"
  echo "Running: $exp"

  # QryEval
  python QryEval.py "$param" > "$OUT_DIR/$exp.log"

  # trec_eval (Summary)
  "$TREC" -m recip_rank -m map -m P.10,20,30 -m ndcg_cut.10,20,30 -m recall.100,1000 \
  "$QREL" "$OUT_DIR/$exp.teIn" > "$OUT_DIR/$exp.teOut"

  # trec_eval (Per Query)
  "$TREC" -q -m map "$QREL" "$OUT_DIR/$exp.teIn" > "$OUT_DIR/$exp.per_query"

  # Runtime
  runtime=$(grep '^Time:' "$OUT_DIR/$exp.log" | awk '{print $2}')
  echo "RunTime all ${runtime:-0:00:00}" >> "$OUT_DIR/$exp.teOut"
  
  echo "Finished: $exp"
done

echo "=============================================="
echo "Generating Results..."

# Generate CSV
python te2csv.py "$OUT_DIR/HW2-Exp-3"
mv te2csv.csv "$OUT_DIR/HW2_Exp3_Metrics.csv"

# Calculate Win/Tie/Loss using the new Exp3 script
if [ -f "$SCRIPT_DIR/calc_wtl_exp3.py" ]; then
    python "$SCRIPT_DIR/calc_wtl_exp3.py" "$OUT_DIR"
else
    echo "Warning: calc_wtl_exp3.py not found. Stats skipped."
fi

echo "Done."