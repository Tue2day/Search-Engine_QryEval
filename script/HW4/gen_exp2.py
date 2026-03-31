import json
import os
import shutil


INDEX_PATH = "INPUT_DIR/index-cw09"
SOURCE_DIR = "EXP_DIR/HW3_EXP1"

EXP_DIR = "EXP_DIR/HW4_EXP2"
OUTPUT_DIR = "OUTPUT_DIR/HW4-EXP2"

TEST_QUERY_NAME = "HW3-Exp-Bow.qry"

BM25_K1 = "1.2"
BM25_B = "0.75"
BM25_DEPTH = "1000"
RERANK_DEPTH = "250"

# Fix the BERT model and title handling so the experiment isolates
# passage formation and aggregation effects.
BERT_MODEL_PATH = "INPUT_DIR/ms-marco-MiniLM-L-6-v2"
BERT_MAX_TITLE_LENGTH = "16"


def ensure_dirs():
    for path in [EXP_DIR, OUTPUT_DIR]:
        os.makedirs(path, exist_ok=True)


def clear_generated_files():
    if not os.path.exists(EXP_DIR):
        return

    for filename in os.listdir(EXP_DIR):
        if filename.startswith("HW4-Exp-2.") and (
            filename.endswith(".param") or filename.endswith(".qry")):
            os.remove(os.path.join(EXP_DIR, filename))


def copy_query_file(target_path):
    shutil.copyfile(f"{SOURCE_DIR}/{TEST_QUERY_NAME}", target_path)


def baseline_param(query_path, output_path):
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": {
            "type": "BM25",
            "BM25:k_1": BM25_K1,
            "BM25:b": BM25_B,
            "outputLength": BM25_DEPTH
        },
        "task_2:output": {
            "type": "trec_eval",
            "outputPath": output_path,
            "outputLength": BM25_DEPTH
        }
    })


def bert_task(psg_len, psg_stride, psg_cnt, aggregation):
    return({
        "type": "bertrr",
        "rerankDepth": RERANK_DEPTH,
        "bertrr:modelPath": BERT_MODEL_PATH,
        "bertrr:psgLen": str(psg_len),
        "bertrr:psgStride": str(psg_stride),
        "bertrr:psgCnt": str(psg_cnt),
        "bertrr:maxTitleLength": BERT_MAX_TITLE_LENGTH,
        "bertrr:scoreAggregation": aggregation
    })


def rerank_param(query_path, exp_id, reranker_task):
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": {
            "type": "inRankFile",
            "inRankFile:Path": f"{OUTPUT_DIR}/HW4-Exp-2.0a.teIn"
        },
        "task_2:reranker": reranker_task,
        "task_3:output": {
            "type": "trec_eval",
            "outputPath": f"{OUTPUT_DIR}/{exp_id}.teIn",
            "outputLength": BM25_DEPTH
        }
    })


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def build_experiments():
    experiments = [
        ("HW4-Exp-2.0a", None),

        # FirstP: psgCnt does not affect the score, so only the 3-passage
        # variants are needed for the report tables.
        ("HW4-Exp-2.1a", bert_task(50, 25, 3, "firstp")),
        ("HW4-Exp-2.1c", bert_task(100, 75, 3, "firstp")),
        ("HW4-Exp-2.1e", bert_task(200, 175, 3, "firstp")),

        ("HW4-Exp-2.2a", bert_task(50, 25, 3, "maxp")),
        ("HW4-Exp-2.2b", bert_task(50, 25, 6, "maxp")),
        ("HW4-Exp-2.2c", bert_task(100, 75, 3, "maxp")),
        ("HW4-Exp-2.2d", bert_task(100, 75, 6, "maxp")),
        ("HW4-Exp-2.2e", bert_task(200, 175, 3, "maxp")),
        ("HW4-Exp-2.2f", bert_task(200, 175, 6, "maxp")),

        ("HW4-Exp-2.3a", bert_task(50, 25, 3, "avgp")),
        ("HW4-Exp-2.3b", bert_task(50, 25, 6, "avgp")),
        ("HW4-Exp-2.3c", bert_task(100, 75, 3, "avgp")),
        ("HW4-Exp-2.3d", bert_task(100, 75, 6, "avgp")),
        ("HW4-Exp-2.3e", bert_task(200, 175, 3, "avgp")),
        ("HW4-Exp-2.3f", bert_task(200, 175, 6, "avgp")),
    ]

    for exp_id, reranker_task in experiments:
        query_path = f"{EXP_DIR}/{exp_id}.qry"
        copy_query_file(query_path)
        param_path = f"{EXP_DIR}/{exp_id}.param"

        if reranker_task is None:
            write_json(param_path, baseline_param(
                query_path, f"{OUTPUT_DIR}/{exp_id}.teIn"))
        else:
            write_json(param_path, rerank_param(
                query_path, exp_id, reranker_task))


def main():
    ensure_dirs()
    clear_generated_files()
    build_experiments()


if __name__ == "__main__":
    main()
