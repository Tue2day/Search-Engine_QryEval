import json
import os
import shutil


INDEX_PATH = "INPUT_DIR/index-cw09"
SOURCE_DIR = "EXP_DIR/HW3_EXP1"

EXP_DIR = "EXP_DIR/HW4_EXP1"
OUTPUT_DIR = "OUTPUT_DIR/HW4-EXP1"

TEST_QUERY_NAME = "HW3-Exp-Bow.qry"
TRAIN_QUERY_NAME = "HW3-train.qry"
TRAIN_QREL_NAME = "HW3-train.qrel"

TRAIN_QUERY_PATH = f"{EXP_DIR}/{TRAIN_QUERY_NAME}"
TRAIN_QREL_PATH = f"{EXP_DIR}/{TRAIN_QREL_NAME}"
BASE_INRANK_PATH = f"{OUTPUT_DIR}/HW4-Exp-1.1a.teIn"

BM25_K1 = "1.2"
BM25_B = "0.75"
QL_MU = "1500"

LTR_ENABLED_FEATURES = [4, 5, 8, 17]

BERT6_PATH = "INPUT_DIR/ms-marco-MiniLM-L-6-v2"
BERT12_PATH = "INPUT_DIR/ms-marco-MiniLM-L-12-v2"
BERT_PSG_LEN = "150"
BERT_PSG_STRIDE = "125"
BERT_PSG_CNT = "3"
BERT_MAX_TITLE_LENGTH = "16"
BERT_AGGREGATION = "maxp"


def ensure_dirs():
    for path in [EXP_DIR, OUTPUT_DIR]:
        os.makedirs(path, exist_ok=True)


def clear_generated_files():
    if not os.path.exists(EXP_DIR):
        return

    for filename in os.listdir(EXP_DIR):
        if filename.startswith("HW4-Exp-1.") and (
            filename.endswith(".param") or filename.endswith(".qry")):
            os.remove(os.path.join(EXP_DIR, filename))


def copy_shared_inputs():
    for filename in [TEST_QUERY_NAME, TRAIN_QUERY_NAME, TRAIN_QREL_NAME]:
        shutil.copyfile(f"{SOURCE_DIR}/{filename}", f"{EXP_DIR}/{filename}")


def copy_query_file(target_path):
    shutil.copyfile(f"{SOURCE_DIR}/{TEST_QUERY_NAME}", target_path)


def disabled_csv(enabled_features):
    all_features = set(range(1, 21))
    disabled = sorted(all_features - set(enabled_features))
    return(",".join(str(fid) for fid in disabled))


def baseline_param(query_path, output_path):
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": {
            "type": "BM25",
            "BM25:k_1": BM25_K1,
            "BM25:b": BM25_B,
            "outputLength": "1000"
        },
        "task_2:output": {
            "type": "trec_eval",
            "outputPath": output_path,
            "outputLength": "1000"
        }
    })


def ltr_task(exp_id, rerank_depth, ranklib_model):
    task = {
        "type": "ltr",
        "rerankDepth": str(rerank_depth),
        "ltr:BM25:b": BM25_B,
        "ltr:BM25:k_1": BM25_K1,
        "ltr:QL:mu": QL_MU,
        "ltr:trainingQueryFile": TRAIN_QUERY_PATH,
        "ltr:trainingQrelsFile": TRAIN_QREL_PATH,
        "ltr:trainingFeatureVectorsFile": (
            f"{OUTPUT_DIR}/{exp_id}.LtrTrain"),
        "ltr:modelFile": f"{OUTPUT_DIR}/{exp_id}.Model",
        "ltr:testingFeatureVectorsFile": (
            f"{OUTPUT_DIR}/{exp_id}.LtrTest"),
        "ltr:testingDocumentScores": f"{OUTPUT_DIR}/{exp_id}.DocScore",
        "ltr:featureDisable": disabled_csv(LTR_ENABLED_FEATURES),
        "ltr:toolkit": "RankLib",
        "ltr:RankLib:model": str(ranklib_model)
    }

    if str(ranklib_model) == "4":
        task["ltr:RankLib:metric2t"] = "MAP"

    return(task)


def bert_task(rerank_depth, model_path):
    return({
        "type": "bertrr",
        "rerankDepth": str(rerank_depth),
        "bertrr:modelPath": model_path,
        "bertrr:psgLen": BERT_PSG_LEN,
        "bertrr:psgStride": BERT_PSG_STRIDE,
        "bertrr:psgCnt": BERT_PSG_CNT,
        "bertrr:maxTitleLength": BERT_MAX_TITLE_LENGTH,
        "bertrr:scoreAggregation": BERT_AGGREGATION
    })


def rerank_param(query_path, exp_id, reranker_task):
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": {
            "type": "inRankFile",
            "inRankFile:Path": BASE_INRANK_PATH
        },
        "task_2:reranker": reranker_task,
        "task_3:output": {
            "type": "trec_eval",
            "outputPath": f"{OUTPUT_DIR}/{exp_id}.teIn",
            "outputLength": "1000"
        }
    })


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def build_experiments():
    experiments = [
        ("HW4-Exp-1.1a", None),
        ("HW4-Exp-1.1b", ltr_task("HW4-Exp-1.1b", 100, 4)),
        ("HW4-Exp-1.1c", ltr_task("HW4-Exp-1.1c", 250, 4)),
        ("HW4-Exp-1.1d", ltr_task("HW4-Exp-1.1d", 500, 4)),
        ("HW4-Exp-1.2b", ltr_task("HW4-Exp-1.2b", 100, 7)),
        ("HW4-Exp-1.2c", ltr_task("HW4-Exp-1.2c", 250, 7)),
        ("HW4-Exp-1.2d", ltr_task("HW4-Exp-1.2d", 500, 7)),
        ("HW4-Exp-1.3b", bert_task(100, BERT6_PATH)),
        ("HW4-Exp-1.3c", bert_task(250, BERT6_PATH)),
        ("HW4-Exp-1.3d", bert_task(500, BERT6_PATH)),
        ("HW4-Exp-1.4b", bert_task(100, BERT12_PATH)),
        ("HW4-Exp-1.4c", bert_task(250, BERT12_PATH)),
        ("HW4-Exp-1.4d", bert_task(500, BERT12_PATH)),
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
    copy_shared_inputs()
    build_experiments()


if __name__ == "__main__":
    main()
