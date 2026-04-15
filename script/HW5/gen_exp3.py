import json
import os
import shutil


INDEX_PATH = "INPUT_DIR/index-cw22b-wp"
QUERY_SOURCE = "QrySet/HW5/questions.qry"
EXP_DIR = "EXP_DIR/HW5_EXP3"
OUTPUT_DIR = "OUTPUT_DIR/HW5-EXP3"

BM25_INRANK_SOURCE_CANDIDATES = [
    "EXP_DIR/HW5_EXP2/HW5-Exp-BM25.inRank",
    "EXP_DIR/HW5_EXP1/HW5-Exp-BM25.inRank",
]
DENSE_INRANK_SOURCE_CANDIDATES = [
    "EXP_DIR/HW5_EXP2/HW5-Exp-Dense.inRank",
    "OUTPUT_DIR/HW5-EXP1/HW5-Exp-1.4b.teIn",
]

BM25_INRANK_PATH = f"{EXP_DIR}/HW5-Exp-BM25.inRank"
DENSE_INRANK_PATH = f"{EXP_DIR}/HW5-Exp-Dense.inRank"

DENSE_MODEL_PATH = "INPUT_DIR/co-condenser-marco-retriever"
BERT_MODEL_PATH = "INPUT_DIR/ms-marco-MiniLM-L-6-v2"
LLM_AUTH_PATH = "INPUT_DIR/llm_auth.json"
LLM_SERVER = "128.2.204.71/59596"
RANKING_OUTPUT_LENGTH = "1000"


def ensure_dirs():
    for path in [EXP_DIR, OUTPUT_DIR, "script/HW5"]:
        os.makedirs(path, exist_ok=True)


def clear_generated_files():
    if not os.path.exists(EXP_DIR):
        return

    for filename in os.listdir(EXP_DIR):
        if filename.startswith("HW5-Exp-3.1") and (
                filename.endswith(".param") or filename.endswith(".qry")):
            os.remove(os.path.join(EXP_DIR, filename))


def copy_first_available(source_candidates, destination):
    for source in source_candidates:
        if os.path.exists(source):
            shutil.copyfile(source, destination)
            return

    raise FileNotFoundError(
        f"Missing source file for {destination}. Checked: "
        f"{', '.join(source_candidates)}")


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


def bertrr_task(rerank_depth="100"):
    return({
        "type": "bertrr",
        "rerankDepth": rerank_depth,
        "bertrr:modelPath": BERT_MODEL_PATH,
        "bertrr:psgLen": "100",
        "bertrr:psgStride": "75",
        "bertrr:psgCnt": "6",
        "bertrr:maxTitleLength": "16",
        "bertrr:scoreAggregation": "maxp"
    })


def rag_task(
        agent_depth="5",
        psg_cnt="6",
        psg_len="150",
        psg_stride="140",
        prompt_id="1"):
    return({
        "type": "rag",
        "agentDepth": agent_depth,
        "rag:modelServer": LLM_SERVER,
        "rag:authPath": LLM_AUTH_PATH,
        "rag:dense:modelPath": DENSE_MODEL_PATH,
        "rag:psgCnt": psg_cnt,
        "rag:psgLen": psg_len,
        "rag:psgStride": psg_stride,
        "rag:maxTitleLength": "15",
        "rag:prompt": prompt_id
    })


def output_trec_task(output_prefix):
    return({
        "type": "trec_eval",
        "outputPath": f"{output_prefix}.teIn",
        "outputLength": RANKING_OUTPUT_LENGTH
    })


def output_qa_task(output_prefix):
    return({
        "type": "triviaqa_evaluation",
        "outputPath": f"{output_prefix}.qaIn",
        "outputLength": "1"
    })


def baseline_param(query_path, exp_id):
    output_prefix = f"{OUTPUT_DIR}/{exp_id}"
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": ranker_task("bm25"),
        "task_2:output": output_trec_task(output_prefix),
        "task_3:agent": {
            **rag_task(),
            "rag:promptPath": f"{output_prefix}.promptRag"
        },
        "task_4:output": output_qa_task(output_prefix)
    })


def custom1_param(query_path, exp_id):
    output_prefix = f"{OUTPUT_DIR}/{exp_id}"
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": ranker_task("bm25"),
        "task_2:output": output_trec_task(output_prefix),
        "task_3:agent": {
            **rag_task(psg_cnt="1", psg_len="150", psg_stride="140"),
            "rag:promptPath": f"{output_prefix}.promptRag"
        },
        "task_4:output": output_qa_task(output_prefix)
    })


def custom2_param(query_path, exp_id):
    output_prefix = f"{OUTPUT_DIR}/{exp_id}"
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": ranker_task("bm25"),
        "task_2:reranker": bertrr_task("100"),
        "task_3:output": output_trec_task(output_prefix),
        "task_4:agent": {
            **rag_task(),
            "rag:promptPath": f"{output_prefix}.promptRag"
        },
        "task_5:output": output_qa_task(output_prefix)
    })


def custom3_param(query_path, exp_id):
    output_prefix = f"{OUTPUT_DIR}/{exp_id}"
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": ranker_task("bm25"),
        "task_2:reranker": bertrr_task("100"),
        "task_3:output": output_trec_task(output_prefix),
        "task_4:agent": {
            **rag_task(prompt_id="3"),
            "rag:promptPath": f"{output_prefix}.promptRag"
        },
        "task_5:output": output_qa_task(output_prefix)
    })


def custom4_param(query_path, exp_id):
    output_prefix = f"{OUTPUT_DIR}/{exp_id}"
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": ranker_task("bm25"),
        "task_2:output": output_trec_task(output_prefix),
        "task_3:agent": {
            **rag_task(psg_cnt="10", psg_len="100", psg_stride="90"),
            "rag:promptPath": f"{output_prefix}.promptRag"
        },
        "task_4:output": output_qa_task(output_prefix)
    })


def custom5_param(query_path, exp_id):
    output_prefix = f"{OUTPUT_DIR}/{exp_id}"
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": ranker_task("dense"),
        "task_2:reranker": bertrr_task("100"),
        "task_3:output": output_trec_task(output_prefix),
        "task_4:agent": {
            **rag_task(psg_cnt="6", psg_len="100", psg_stride="90"),
            "rag:promptPath": f"{output_prefix}.promptRag"
        },
        "task_5:output": output_qa_task(output_prefix)
    })


def custom6_param(query_path, exp_id):
    output_prefix = f"{OUTPUT_DIR}/{exp_id}"
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": ranker_task("bm25"),
        "task_2:reranker": {
            "type": "rankfusion",
            "rerankDepth": "1000",
            "rankfusion:secondaryRankPath": DENSE_INRANK_PATH,
            "rankfusion:method": "interleave",
            "rankfusion:maxInputRank": "1000",
            "rankfusion:maxOutputRank": "100"
        },
        "task_3:output": output_trec_task(output_prefix),
        "task_4:agent": {
            **rag_task(),
            "rag:promptPath": f"{output_prefix}.promptRag"
        },
        "task_5:output": output_qa_task(output_prefix)
    })


EXPERIMENT_BUILDERS = [
    ("HW5-Exp-3.1a", baseline_param),
    ("HW5-Exp-3.1b", custom1_param),
    ("HW5-Exp-3.1c", custom2_param),
    ("HW5-Exp-3.1d", custom3_param),
    ("HW5-Exp-3.1e", custom4_param),
    ("HW5-Exp-3.1f", custom5_param),
    ("HW5-Exp-3.1g", custom6_param),
]


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=4)


def build_experiments():
    for exp_id, builder in EXPERIMENT_BUILDERS:
        query_path = f"{EXP_DIR}/{exp_id}.qry"
        param_path = f"{EXP_DIR}/{exp_id}.param"
        copy_query_file(query_path)
        write_json(param_path, builder(query_path, exp_id))


def main():
    ensure_dirs()
    clear_generated_files()
    copy_first_available(BM25_INRANK_SOURCE_CANDIDATES, BM25_INRANK_PATH)
    copy_first_available(DENSE_INRANK_SOURCE_CANDIDATES, DENSE_INRANK_PATH)
    build_experiments()


if __name__ == "__main__":
    main()
