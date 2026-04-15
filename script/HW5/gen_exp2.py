import json
import os
import shutil


INDEX_PATH = "INPUT_DIR/index-cw22b-wp"
QUERY_SOURCE = "QrySet/HW5/questions.qry"
EXP_DIR = "EXP_DIR/HW5_EXP2"
OUTPUT_DIR = "OUTPUT_DIR/HW5-EXP2"

BM25_INRANK_PATH = "EXP_DIR/HW5_EXP2/HW5-Exp-BM25.inRank"
DENSE_INRANK_PATH = "EXP_DIR/HW5_EXP2/HW5-Exp-Dense.inRank"
DENSE_INDEX_PATH = "INPUT_DIR/index-cw22b-wp-faiss-b300-Fp"
DENSE_MODEL_PATH = "INPUT_DIR/co-condenser-marco-retriever"
LLM_AUTH_PATH = "INPUT_DIR/llm_auth.json"

LLM_SERVER = "128.2.204.71/59596"
RANKING_OUTPUT_LENGTH = "1000"
AGENT_DEPTH = "5"
RAG_MAX_TITLE_LENGTH = "15"

BM25_BEST_CONFIG = {
    "psgCnt": "6",
    "psgLen": "150",
    "psgStride": "140"
}

DENSE_BEST_CONFIG = {
    "psgCnt": "6",
    "psgLen": "100",
    "psgStride": "90"
}


def ensure_dirs():
    for path in [EXP_DIR, OUTPUT_DIR, "script/HW5"]:
        os.makedirs(path, exist_ok=True)


def clear_generated_files():
    if not os.path.exists(EXP_DIR):
        return

    for filename in os.listdir(EXP_DIR):
        if filename.startswith("HW5-Exp-2.") and (
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
            "type": "inRankFile",
            "inRankFile:Path": DENSE_INRANK_PATH,
            "outputLength": RANKING_OUTPUT_LENGTH
        })

    raise ValueError(f"Unknown retrieval type: {retrieval_type}")


def agent_task(retrieval_type, prompt_id):
    if retrieval_type == "bm25":
        config = BM25_BEST_CONFIG
    elif retrieval_type == "dense":
        config = DENSE_BEST_CONFIG
    else:
        raise ValueError(f"Unknown retrieval type: {retrieval_type}")

    return({
        "type": "rag",
        "agentDepth": AGENT_DEPTH,
        "rag:modelServer": LLM_SERVER,
        "rag:authPath": LLM_AUTH_PATH,
        "rag:dense:modelPath": DENSE_MODEL_PATH,
        "rag:psgCnt": config["psgCnt"],
        "rag:psgLen": config["psgLen"],
        "rag:psgStride": config["psgStride"],
        "rag:maxTitleLength": RAG_MAX_TITLE_LENGTH,
        "rag:prompt": str(prompt_id)
    })


def experiment_param(query_path, exp_id, retrieval_type, prompt_id):
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
            **agent_task(retrieval_type, prompt_id),
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
        ("HW5-Exp-2.1a", "bm25", 1),
        ("HW5-Exp-2.1b", "bm25", 2),
        ("HW5-Exp-2.1c", "bm25", 3),
        ("HW5-Exp-2.1d", "bm25", 4),
        ("HW5-Exp-2.1e", "bm25", 5),
        ("HW5-Exp-2.1f", "bm25", 6),
        ("HW5-Exp-2.2a", "dense", 1),
        ("HW5-Exp-2.2b", "dense", 2),
        ("HW5-Exp-2.2c", "dense", 3),
        ("HW5-Exp-2.2d", "dense", 4),
        ("HW5-Exp-2.2e", "dense", 5),
        ("HW5-Exp-2.2f", "dense", 6),
    ]

    for exp_id, retrieval_type, prompt_id in experiment_specs:
        query_path = f"{EXP_DIR}/{exp_id}.qry"
        copy_query_file(query_path)
        param_path = f"{EXP_DIR}/{exp_id}.param"
        write_json(param_path, experiment_param(
            query_path, exp_id, retrieval_type, prompt_id))


def main():
    ensure_dirs()
    clear_generated_files()
    build_experiments()


if __name__ == "__main__":
    main()
