#!/bin/sh

SCRIPT_DIR=$(dirname "$0")

PARAM_DIR="EXP_DIR/HW3_EXP1"
OUT_DIR="OUTPUT_DIR/HW3-EXP1"

QREL="INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed"
TREC="INPUT_DIR/trec_eval-9.0.4"

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Starting HW3 Experiment 1"
echo "Params: $PARAM_DIR"
echo "Output: $OUT_DIR"
echo "=============================================="

for param in "$PARAM_DIR"/HW3-Exp-1.*.param
do
  exp=$(basename "$param" .param)

  echo "----------------------------------------------"
  echo "Running experiment: $exp"

  python QryEval.py "$param" > "$OUT_DIR/$exp.log"

  "$TREC" \
  -m recip_rank \
  -m P.10,20,30 \
  -m map \
  -m ndcg_cut.10,20,30 \
  "$QREL" \
  "$OUT_DIR/$exp.teIn" \
  > "$OUT_DIR/$exp.teOut"

  runtime=$(grep '^Time:' "$OUT_DIR/$exp.log" | awk '{print $2}')
  if [ -z "$runtime" ]; then
    runtime="0:00:00"
  fi
  echo "RunTime all $runtime" >> "$OUT_DIR/$exp.teOut"

  echo "Finished: $exp"
done

echo "=============================================="
echo "All HW3 Experiment 1 runs finished."

python te2csv.py "$OUT_DIR/HW3-Exp-1"
mv te2csv.csv "$OUT_DIR/HW3_Exp1_Metrics.csv"

echo "----------------------------------------------"
echo "Metrics CSV: $OUT_DIR/HW3_Exp1_Metrics.csv"
