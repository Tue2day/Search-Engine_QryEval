#!/bin/sh

# ==============================================================================
# Configuration
# ==============================================================================

# Directory where this script is stored (used to find calc_wtl.py)
SCRIPT_DIR=$(dirname "$0")

# Input/Output Directories (Relative to Project Root)
PARAM_DIR="EXP_DIR/HW2_EXP1"
OUT_DIR="OUTPUT_DIR/HW2-EXP1"

# Paths to Tools and Data (Matches your HW1 setup)
# Note: Ensure trec_eval is executable
QREL="INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed"
TREC="INPUT_DIR/trec_eval-9.0.4"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Starting HW2 Experiment 1"
echo "Params: $PARAM_DIR"
echo "Output: $OUT_DIR"
echo "Script Location: $SCRIPT_DIR"
echo "=============================================="

# ==============================================================================
# 1. Run Experiments Loop
# ==============================================================================

# Iterate over all param files starting with HW2-Exp-1
for param in "$PARAM_DIR"/HW2-Exp-1.*.param
do
  # Extract experiment name (e.g., HW2-Exp-1.1a)
  exp=$(basename "$param" .param)

  echo "----------------------------------------------"
  echo "Running experiment: $exp"

  # A. Run QryEval (Generates .teIn, redirects stdout to .log)
  python QryEval.py "$param" > "$OUT_DIR/$exp.log"

  # B. Run trec_eval (Generates summary .teOut)
  # Calculates standard metrics: MRR, MAP, P@k, NDCG@k, Recall
  "$TREC" \
  -m recip_rank \
  -m map \
  -m P.10,20,30 \
  -m ndcg_cut.10,20,30 \
  -m recall.100,1000 \
  "$QREL" \
  "$OUT_DIR/$exp.teIn" \
  > "$OUT_DIR/$exp.teOut"

  # C. Run trec_eval per-query (Generates .per_query)
  # Required for calculating Win/Tie/Loss statistics later
  "$TREC" \
  -q \
  -m map \
  "$QREL" \
  "$OUT_DIR/$exp.teIn" \
  > "$OUT_DIR/$exp.per_query"

  # D. Extract Runtime and append to .teOut (Matches HW1 logic)
  # Looks for "Time: xx.xx" in the QryEval log
  runtime=$(grep '^Time:' "$OUT_DIR/$exp.log" | awk '{print $2}')
  if [ -z "$runtime" ]; then
    runtime="0:00:00"
  fi
  echo "RunTime all $runtime" >> "$OUT_DIR/$exp.teOut"

  echo "Finished: $exp (Runtime: $runtime)"
done

echo "=============================================="
echo "All experiments finished."

# ==============================================================================
# 2. Generate Results CSV
# ==============================================================================

# A. Use the official te2csv.py (Located in Root) to generate metrics CSV
# This aggregates all .teOut files into one CSV
echo "Generating Metrics CSV using te2csv.py..."
python te2csv.py "$OUT_DIR/HW2-Exp-1"
mv te2csv.csv "$OUT_DIR/HW2_Exp1_Metrics.csv"

# B. Use calc_wtl_exp1.py (Located in script/HW2) to calculate Win/Tie/Loss
# This reads the .per_query files generated in step 1.C
if [ -f "$SCRIPT_DIR/calc_wtl_exp1.py" ]; then
    echo "Calculating Win/Tie/Loss statistics..."
    python "$SCRIPT_DIR/calc_wtl.py" "$OUT_DIR"
else
    echo "Warning: calc_wtl.py not found in $SCRIPT_DIR. Win/Tie/Loss stats skipped."
fi

echo "----------------------------------------------"
echo "Results generation complete."
echo "Metrics CSV: $OUT_DIR/HW2_Exp1_Metrics.csv"
echo "Check console output above for Win/Tie/Loss table."