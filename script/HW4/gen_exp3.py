import json
import os
import shutil


INDEX_PATH = "INPUT_DIR/index-cw09"
SOURCE_DIR = "EXP_DIR/HW3_EXP1"

EXP_DIR = "EXP_DIR/HW4_EXP3"
OUTPUT_DIR = "OUTPUT_DIR/HW4-EXP3"

TEST_QUERY_NAME = "HW3-Exp-Bow.qry"
TRAIN_QUERY_NAME = "HW3-train.qry"
TRAIN_QREL_NAME = "HW3-train.qrel"

TRAIN_QUERY_PATH = f"{EXP_DIR}/{TRAIN_QUERY_NAME}"
TRAIN_QREL_PATH = f"{EXP_DIR}/{TRAIN_QREL_NAME}"

BM25_K1 = "1.2"
BM25_B = "0.75"
BM25_DEPTH = "1000"
QL_MU = "1500"

LTR_FEATURE_DISABLE = "1,2,3,6,7,9,10,11,12,13,14,15,16,18,19,20"
SVMRANK_C = "0.001"

BERT6_PATH = "INPUT_DIR/ms-marco-MiniLM-L-6-v2"
BERT12_PATH = "INPUT_DIR/ms-marco-MiniLM-L-12-v2"


def ensure_dirs():
    for path in [EXP_DIR, OUTPUT_DIR]:
        os.makedirs(path, exist_ok=True)


def clear_generated_files():
    if not os.path.exists(EXP_DIR):
        return

    for filename in os.listdir(EXP_DIR):
        if filename.startswith("HW4-Exp-3.1") and (
            filename.endswith(".param") or filename.endswith(".qry")):
            os.remove(os.path.join(EXP_DIR, filename))


def copy_shared_inputs():
    for filename in [TRAIN_QUERY_NAME, TRAIN_QREL_NAME]:
        shutil.copyfile(f"{SOURCE_DIR}/{filename}", f"{EXP_DIR}/{filename}")


def copy_query_file(target_path):
    shutil.copyfile(f"{SOURCE_DIR}/{TEST_QUERY_NAME}", target_path)


def bm25_ranker_task():
    return({
        "type": "BM25",
        "BM25:k_1": BM25_K1,
        "BM25:b": BM25_B,
        "outputLength": BM25_DEPTH
    })


def ranked_boolean_ranker_task():
    return({
        "type": "RankedBoolean",
        "outputLength": BM25_DEPTH
    })


def prf_task(exp_id, algorithm, num_docs, num_terms, field_in, field_out,
             orig_weight=None):
    task = {
        "type": "prf",
        "prf:algorithm": algorithm,
        "prf:numDocs": str(num_docs),
        "prf:numTerms": str(num_terms),
        "prf:expansionFieldIn": field_in,
        "prf:expansionFieldOut": field_out,
        "prf:expansionQueryFile": f"{OUTPUT_DIR}/{exp_id}.qryOut"
    }

    if orig_weight is not None:
        task["prf:rm3:origWeight"] = str(orig_weight)

    return(task)


def ltr_svmrank_task(exp_id, rerank_depth):
    return({
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
        "ltr:featureDisable": LTR_FEATURE_DISABLE,
        "ltr:toolkit": "SVMRank",
        "ltr:svmRankLearnPath": "INPUT_DIR/svm_rank_learn",
        "ltr:svmRankClassifyPath": "INPUT_DIR/svm_rank_classify",
        "ltr:svmRankParamC": SVMRANK_C
    })


def ltr_ranklib_task(exp_id, rerank_depth, ranklib_model, metric2t=None):
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
        "ltr:featureDisable": LTR_FEATURE_DISABLE,
        "ltr:toolkit": "RankLib",
        "ltr:RankLib:model": str(ranklib_model)
    }

    if metric2t is not None:
        task["ltr:RankLib:metric2t"] = metric2t

    return(task)


def bert_task(rerank_depth, model_path, psg_len, psg_stride, psg_cnt,
              aggregation, max_title_length):
    return({
        "type": "bertrr",
        "rerankDepth": str(rerank_depth),
        "bertrr:modelPath": model_path,
        "bertrr:psgLen": str(psg_len),
        "bertrr:psgStride": str(psg_stride),
        "bertrr:psgCnt": str(psg_cnt),
        "bertrr:maxTitleLength": str(max_title_length),
        "bertrr:scoreAggregation": aggregation
    })


def output_task(exp_id):
    return({
        "type": "trec_eval",
        "outputPath": f"{OUTPUT_DIR}/{exp_id}.teIn",
        "outputLength": BM25_DEPTH
    })


def config_1(exp_id):
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": f"{EXP_DIR}/{exp_id}.qry",
        "task_1:ranker": bm25_ranker_task(),
        "task_2:reranker": ltr_svmrank_task(exp_id, 250),
        "task_3:reranker": bert_task(
            100, BERT12_PATH, 100, 75, 6, "maxp", 16),
        "task_4:output": output_task(exp_id)
    })


def config_2(exp_id):
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": f"{EXP_DIR}/{exp_id}.qry",
        "task_1:ranker": bm25_ranker_task(),
        "task_2:rewriter": prf_task(
            exp_id, "rm3", 10, 10, "keywords", "body", 0.0),
        "task_3:ranker": bm25_ranker_task(),
        "task_4:reranker": ltr_svmrank_task(exp_id, 250),
        "task_5:reranker": bert_task(
            100, BERT12_PATH, 100, 75, 6, "maxp", 16),
        "task_6:output": output_task(exp_id)
    })


def config_3(exp_id):
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": f"{EXP_DIR}/{exp_id}.qry",
        "task_1:ranker": bm25_ranker_task(),
        "task_2:reranker": ltr_ranklib_task(exp_id, 100, 7),
        "task_3:reranker": bert_task(
            100, BERT6_PATH, 100, 75, 6, "maxp", 16),
        "task_4:output": output_task(exp_id)
    })


def config_4(exp_id):
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": f"{EXP_DIR}/{exp_id}.qry",
        "task_1:ranker": ranked_boolean_ranker_task(),
        "task_2:reranker": ltr_svmrank_task(exp_id, 250),
        "task_3:reranker": bert_task(
            100, BERT12_PATH, 100, 75, 6, "maxp", 16),
        "task_4:output": output_task(exp_id)
    })


def config_5(exp_id):
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": f"{EXP_DIR}/{exp_id}.qry",
        "task_1:ranker": bm25_ranker_task(),
        "task_2:reranker": bert_task(
            250, BERT6_PATH, 50, 25, 6, "avgp", 16),
        "task_3:reranker": bert_task(
            100, BERT12_PATH, 100, 75, 6, "maxp", 16),
        "task_4:output": output_task(exp_id)
    })


def config_6(exp_id):
    return({
        "indexPath": INDEX_PATH,
        "queryFilePath": f"{EXP_DIR}/{exp_id}.qry",
        "task_1:ranker": bm25_ranker_task(),
        "task_2:reranker": bert_task(
            250, BERT6_PATH, 100, 75, 3, "maxp", 16),
        "task_3:reranker": ltr_svmrank_task(exp_id, 100),
        "task_4:output": output_task(exp_id)
    })


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def build_experiments():
    experiments = [
        ("HW4-Exp-3.1a", config_1),
        ("HW4-Exp-3.1b", config_2),
        ("HW4-Exp-3.1c", config_3),
        ("HW4-Exp-3.1d", config_4),
        ("HW4-Exp-3.1e", config_5),
        ("HW4-Exp-3.1f", config_6),
    ]

    for exp_id, config_builder in experiments:
        query_path = f"{EXP_DIR}/{exp_id}.qry"
        copy_query_file(query_path)
        param_path = f"{EXP_DIR}/{exp_id}.param"
        write_json(param_path, config_builder(exp_id))


def main():
    ensure_dirs()
    clear_generated_files()
    copy_shared_inputs()
    build_experiments()


if __name__ == "__main__":
    main()
