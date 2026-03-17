#!/bin/sh


PARAM_DIR="EXP_DIR/HW1_EXP2"
OUT_DIR="OUTPUT_DIR/HW1-EXP2"

QREL="INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed"
TREC="INPUT_DIR/trec_eval-9.0.4"

mkdir -p "$OUT_DIR"


# Run all Experiment 2 query-set experiments
for param in "$PARAM_DIR"/HW1-Exp-*.param
do
  exp=$(basename "$param" .param)

  echo "=============================================="
  echo "Running experiment: $exp"

  # 1. Run QryEval (generates .teIn, stdout -> log)
  python QryEval.py "$param" > "$OUT_DIR/$exp.log"

  # 2. Run trec_eval (generates .teOut)
  "$TREC" \
    -m recip_rank \
    -m map \
    -m P.10,20,30 \
    -m ndcg_cut.10,20,30 \
    -m recall.100,1000 \
    "$QREL" \
    "$OUT_DIR/$exp.teIn" \
    > "$OUT_DIR/$exp.teOut"

  # 3. Extract runtime from log and append to teOut
  runtime=$(grep '^Time:' "$OUT_DIR/$exp.log" | awk '{print $2}')
  if [ -z "$runtime" ]; then
    runtime="0:00:00"
  fi
  echo "RunTime all $runtime" >> "$OUT_DIR/$exp.teOut"

  echo "Finished: $exp"
done

echo "=============================================="
echo "All Experiment 2 runs finished."


# Generate CSV
python te2csv.py "$OUT_DIR/HW1-Exp"
mv te2csv.csv "$OUT_DIR/HW1_Results.csv"

echo "CSV saved to $OUT_DIR/HW1_Results.csv"
