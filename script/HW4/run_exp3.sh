#!/bin/sh

PARAM_DIR="EXP_DIR/HW4_EXP3"
OUT_DIR="OUTPUT_DIR/HW4-EXP3"

QREL="INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed"
TREC="INPUT_DIR/trec_eval-9.0.4"

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Starting HW4 Experiment 3"
echo "Params: $PARAM_DIR"
echo "Output: $OUT_DIR"
echo "=============================================="

for param in "$PARAM_DIR"/HW4-Exp-3.1*.param
do
  exp=$(basename "$param" .param)
  tein="$OUT_DIR/$exp.teIn"

  echo "----------------------------------------------"
  echo "Running experiment: $exp"

  python QryEval.py "$param" > "$OUT_DIR/$exp.log"

  "$TREC" \
    -m recip_rank \
    -m P.10,20,30 \
    -m map_cut.1000 \
    -m ndcg_cut.10,20,30 \
    -m recall.100,500,1000 \
    "$QREL" \
    "$tein" \
    > "$OUT_DIR/$exp.teOut"

  runtime=$(grep '^Time:' "$OUT_DIR/$exp.log" | awk '{print $2}')
  if [ -z "$runtime" ]; then
    runtime="0:00:00"
  fi
  echo "RunTime all $runtime" >> "$OUT_DIR/$exp.teOut"

  echo "Finished: $exp"
done

echo "=============================================="
echo "All HW4 Experiment 3 runs finished."

python te2csv.py "$OUT_DIR/HW4-Exp-3.1"
mv te2csv.csv "$OUT_DIR/HW4_Exp3_Metrics.csv"

echo "----------------------------------------------"
echo "Metrics CSV: $OUT_DIR/HW4_Exp3_Metrics.csv"
