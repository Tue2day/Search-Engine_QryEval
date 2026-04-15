#!/bin/sh

PARAM_DIR="EXP_DIR/HW5_EXP3"
OUT_DIR="OUTPUT_DIR/HW5-EXP3"

QREL="INPUT_DIR/verified-wikipedia-dev.qrel"
TREC="INPUT_DIR/trec_eval-9.0.4"
QA_EVAL="INPUT_DIR/triviaqa_evaluation/triviaqa_evaluation.py"
QA_DATASET="INPUT_DIR/triviaqa_evaluation/verified-wikipedia-dev.json"
BM25_INRANK="EXP_DIR/HW5_EXP3/HW5-Exp-BM25.inRank"
DENSE_INRANK="EXP_DIR/HW5_EXP3/HW5-Exp-Dense.inRank"

mkdir -p "$OUT_DIR"

if [ ! -f "$BM25_INRANK" ]; then
  echo "Missing BM25 inRank file: $BM25_INRANK"
  exit 1
fi

if [ ! -f "$DENSE_INRANK" ]; then
  echo "Missing dense inRank file: $DENSE_INRANK"
  exit 1
fi

echo "=============================================="
echo "Starting HW5 Experiment 3"
echo "Params: $PARAM_DIR"
echo "Output: $OUT_DIR"
echo "=============================================="

for exp in \
  HW5-Exp-3.1b \
  HW5-Exp-3.1c \
  HW5-Exp-3.1d \
  HW5-Exp-3.1e \
  HW5-Exp-3.1f \
  HW5-Exp-3.1g
do
  param="$PARAM_DIR/$exp.param"
  tein="$OUT_DIR/$exp.teIn"
  qain="$OUT_DIR/$exp.qaIn"

  echo "----------------------------------------------"
  echo "Running experiment: $exp"

  python3 QryEval.py "$param" > "$OUT_DIR/$exp.log"

  "$TREC" \
    -m recip_rank \
    -m P.1,5 \
    "$QREL" \
    "$tein" \
    > "$OUT_DIR/$exp.teOut"

  python3 "$QA_EVAL" \
    --dataset_file "$QA_DATASET" \
    --prediction_file "$qain" \
    > "$OUT_DIR/$exp.qaOut"

  echo "Finished: $exp"
done

echo "=============================================="
echo "All HW5 Experiment 3 runs finished."

python3 script/HW5/exp3_to_csv.py

echo "----------------------------------------------"
echo "Metrics CSV: $OUT_DIR/HW5_Exp3_Metrics.csv"
