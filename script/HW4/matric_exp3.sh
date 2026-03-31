#!/bin/sh

OUT_DIR="OUTPUT_DIR/HW4-EXP3"
QREL="INPUT_DIR/cw09a.adhoc.1-200.qrel.indexed"
TREC="INPUT_DIR/trec_eval-9.0.4"

echo "=============================================="
echo "Regenerating teOut and CSV from existing runs"
echo "Output: $OUT_DIR"
echo "=============================================="

for tein in "$OUT_DIR"/HW4-Exp-3.*.teIn
do
  exp=$(basename "$tein" .teIn)

  echo "----------------------------------------------"
  echo "Evaluating existing ranking: $exp"

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
echo "All teOut files regenerated."

python te2csv.py "$OUT_DIR/HW4-Exp-3."
mv te2csv.csv "$OUT_DIR/HW4_Exp3_Metrics.csv"

echo "----------------------------------------------"
echo "Metrics CSV: $OUT_DIR/HW4_Exp3_Metrics.csv"