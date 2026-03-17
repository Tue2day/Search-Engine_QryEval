#!/bin/sh

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR=$(dirname "$0")

# Paths (Relative to Root)
PARAM_DIR="EXP_DIR/HW2_EXP2"
OUT_DIR="OUTPUT_DIR/HW2-EXP2"

# Baseline Info (from Experiment 1)
BASELINE_DIR="OUTPUT_DIR/HW2-EXP1"
BASELINE_PER_QUERY="$BASELINE_DIR/HW2-Exp-1.1a.per_query"

# Tools
QREL="INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed"
TREC="INPUT_DIR/trec_eval-9.0.4"

# Check Prerequisites
if [ ! -f "$BASELINE_PER_QUERY" ]; then
    echo "Error: Baseline file $BASELINE_PER_QUERY not found."
    echo "Please run Experiment 1 first!"
    exit 1
fi

mkdir -p "$OUT_DIR"
# Copy baseline per_query to current dir so calc_wtl can find it easily
cp "$BASELINE_PER_QUERY" "$OUT_DIR/"

echo "=============================================="
echo "Starting HW2 Experiment 2 (Different Fields)"
echo "Params: $PARAM_DIR"
echo "Output: $OUT_DIR"
echo "=============================================="

# ==============================================================================
# Run Loop
# ==============================================================================

for param in "$PARAM_DIR"/HW2-Exp-2.*.param
do
  exp=$(basename "$param" .param)
  echo "----------------------------------------------"
  echo "Running: $exp"

  # 1. QryEval
  python QryEval.py "$param" > "$OUT_DIR/$exp.log"

  # 2. trec_eval (Summary Stats)
  "$TREC" \
  -m recip_rank \
  -m P.10,20,30 \
  -m map \
  -m ndcg_cut.10,20,30 \
  -m recall.100,1000 \
  "$QREL" \
  "$OUT_DIR/$exp.teIn" \
  > "$OUT_DIR/$exp.teOut"

  # 3. trec_eval (Per Query for W/T/L)
  "$TREC" -q -m map "$QREL" "$OUT_DIR/$exp.teIn" > "$OUT_DIR/$exp.per_query"

  # 4. Runtime
  runtime=$(grep '^Time:' "$OUT_DIR/$exp.log" | awk '{print $2}')
  if [ -z "$runtime" ]; then runtime="0:00:00"; fi
  echo "RunTime all $runtime" >> "$OUT_DIR/$exp.teOut"

  echo "Finished: $exp"
done

echo "=============================================="
echo "Experiments finished."

# ==============================================================================
# Generate Results
# ==============================================================================

# 1. Generate Metrics CSV
echo "Generating Metrics CSV..."
python te2csv.py "$OUT_DIR/HW2-Exp-2"
mv te2csv.csv "$OUT_DIR/HW2_Exp2_Metrics.csv"

# 2. Calculate Win/Tie/Loss
if [ -f "$SCRIPT_DIR/calc_wtl_exp2.py" ]; then
    echo "Calculating Win/Tie/Loss statistics (vs 1.1b and 1.2b)..."
    python "$SCRIPT_DIR/calc_wtl_exp2.py" "$OUT_DIR"
else
    echo "Warning: calc_wtl_exp2.py not found. Skipping stats."
fi

echo "----------------------------------------------"
echo "Done."