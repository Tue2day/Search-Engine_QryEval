import json
import os
import shutil


INDEX_PATH = "INPUT_DIR/index-cw22b-wp"
QUERY_SOURCE = "QrySet/HW5/questions.qry"
EXP_DIR = "EXP_DIR/HW5_EXP1"
OUTPUT_DIR = "OUTPUT_DIR/HW5-EXP1"

BM25_INRANK_PATH = "EXP_DIR/HW5_EXP1/HW5-Exp-BM25.inRank"
DENSE_INDEX_PATH = "INPUT_DIR/index-cw22b-wp-faiss-b300-Fp"
DENSE_MODEL_PATH = "INPUT_DIR/co-condenser-marco-retriever"
LLM_AUTH_PATH = "INPUT_DIR/llm_auth.json"

PROMPT_ID = "1"
AGENT_DEPTH = "5"
MAX_TITLE_LENGTH = "15"
RANKING_OUTPUT_LENGTH = "1000"
LLM_SERVER = "128.2.204.71/59596"

PASSAGE_LENGTHS = [50, 100, 150, 200]


def ensure_dirs():
    for path in [EXP_DIR, OUTPUT_DIR, "script/HW5"]:
        os.makedirs(path, exist_ok=True)


def clear_generated_files():
    if not os.path.exists(EXP_DIR):
        return

    for filename in os.listdir(EXP_DIR):
        if filename.startswith("HW5-Exp-1.") and (
                filename.endswith(".param") or filename.endswith(".qry")):
            os.remove(os.path.join(EXP_DIR, filename))


def copy_query_file(target_path):
    shutil.copyfile(QUERY_SOURCE, target_path)


def ranker_task(retrieval_type):
    if retrieval_type == "bm25":
        return({
            "type": "inRankFile",
            "inRankFile:Path": BM25_INRANK_PATH,
            "outputLength": RANKING_OUTPUT_LENGTH
        })

    if retrieval_type == "dense":
        return({
            "type": "dense",
            "dense:indexPath": DENSE_INDEX_PATH,
            "dense:modelPath": DENSE_MODEL_PATH,
            "outputLength": RANKING_OUTPUT_LENGTH
        })

    raise ValueError(f"Unknown retrieval type: {retrieval_type}")


def agent_task(psg_len, psg_cnt):
    return({
        "type": "rag",
        "agentDepth": AGENT_DEPTH,
        "rag:modelServer": LLM_SERVER,
        "rag:authPath": LLM_AUTH_PATH,
        "rag:dense:modelPath": DENSE_MODEL_PATH,
        "rag:psgCnt": str(psg_cnt),
        "rag:psgLen": str(psg_len),
        "rag:psgStride": str(psg_len - 10),
        "rag:maxTitleLength": MAX_TITLE_LENGTH,
        "rag:prompt": PROMPT_ID
    })


def experiment_param(query_path, exp_id, retrieval_type, psg_len, psg_cnt):
    output_prefix = f"{OUTPUT_DIR}/{exp_id}"
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": ranker_task(retrieval_type),
        "task_2:output": {
            "type": "trec_eval",
            "outputPath": f"{output_prefix}.teIn",
            "outputLength": RANKING_OUTPUT_LENGTH
        },
        "task_3:agent": {
            **agent_task(psg_len, psg_cnt),
            "rag:promptPath": f"{output_prefix}.promptRag"
        },
        "task_4:output": {
            "type": "triviaqa_evaluation",
            "outputPath": f"{output_prefix}.qaIn",
            "outputLength": "1"
        }
    })


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=4)


def build_experiments():
    experiment_specs = [
        ("HW5-Exp-1.1a", "bm25", 50, 1),
        ("HW5-Exp-1.1b", "bm25", 100, 1),
        ("HW5-Exp-1.1c", "bm25", 150, 1),
        ("HW5-Exp-1.1d", "bm25", 200, 1),
        ("HW5-Exp-1.2a", "bm25", 50, 6),
        ("HW5-Exp-1.2b", "bm25", 100, 6),
        ("HW5-Exp-1.2c", "bm25", 150, 6),
        ("HW5-Exp-1.2d", "bm25", 200, 6),
        ("HW5-Exp-1.3a", "dense", 50, 1),
        ("HW5-Exp-1.3b", "dense", 100, 1),
        ("HW5-Exp-1.3c", "dense", 150, 1),
        ("HW5-Exp-1.3d", "dense", 200, 1),
        ("HW5-Exp-1.4a", "dense", 50, 6),
        ("HW5-Exp-1.4b", "dense", 100, 6),
        ("HW5-Exp-1.4c", "dense", 150, 6),
        ("HW5-Exp-1.4d", "dense", 200, 6),
    ]

    for exp_id, retrieval_type, psg_len, psg_cnt in experiment_specs:
        query_path = f"{EXP_DIR}/{exp_id}.qry"
        copy_query_file(query_path)
        param_path = f"{EXP_DIR}/{exp_id}.param"
        write_json(param_path, experiment_param(
            query_path, exp_id, retrieval_type, psg_len, psg_cnt))


def main():
    ensure_dirs()
    clear_generated_files()
    build_experiments()


if __name__ == "__main__":
    main()
