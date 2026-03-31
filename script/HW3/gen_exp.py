import json
import os
import shutil


INDEX_PATH = "INPUT_DIR/index-cw09"
SOURCE_DIR = "/Users/Caleb/Documents/CMU/11642/HW3"

EXP1_DIR = "EXP_DIR/HW3_EXP1"
EXP2_DIR = "EXP_DIR/HW3_EXP2"
EXP3_DIR = "EXP_DIR/HW3_EXP3"

OUTPUT1_DIR = "OUTPUT_DIR/HW3-EXP1"
OUTPUT2_DIR = "OUTPUT_DIR/HW3-EXP2"
OUTPUT3_DIR = "OUTPUT_DIR/HW3-EXP3"

TEST_QUERY_NAME = "HW3-Exp-Bow.qry"
TRAIN_QUERY_NAME = "HW3-train.qry"
TRAIN_QREL_NAME = "HW3-train.qrel"

TRAIN_QUERY_PATH = f"{EXP1_DIR}/{TRAIN_QUERY_NAME}"
TRAIN_QREL_PATH = f"{EXP1_DIR}/{TRAIN_QREL_NAME}"
BASE_QUERY_PATH = f"{EXP1_DIR}/{TEST_QUERY_NAME}"
BASE_INRANK_PATH = f"{OUTPUT1_DIR}/HW3-Exp-1.1a.teIn"

BM25_K1 = "1.2"
BM25_B = "0.75"
QL_MU = "1500"
RERANK_DEPTH = "100"


def ensure_dirs():
    for path in [EXP1_DIR, EXP2_DIR, EXP3_DIR, OUTPUT1_DIR, OUTPUT2_DIR, OUTPUT3_DIR]:
        os.makedirs(path, exist_ok=True)


def copy_shared_inputs():
    for filename, target_dir in [
        (TEST_QUERY_NAME, EXP1_DIR),
        (TRAIN_QUERY_NAME, EXP1_DIR),
        (TRAIN_QREL_NAME, EXP1_DIR),
    ]:
        shutil.copyfile(f"{SOURCE_DIR}/{filename}", f"{target_dir}/{filename}")


def copy_query_file(target_path):
    shutil.copyfile(f"{SOURCE_DIR}/{TEST_QUERY_NAME}", target_path)


def disabled_csv(enabled_features):
    all_features = set(range(1, 21))
    disabled = sorted(all_features - set(enabled_features))
    return ",".join(str(fid) for fid in disabled)


def ltr_task(exp_id, output_dir, toolkit, enabled_features, ranklib_model=None, use_prf=False):
    task = {
        "type": "ltr",
        "rerankDepth": RERANK_DEPTH,
        "ltr:BM25:b": BM25_B,
        "ltr:BM25:k_1": BM25_K1,
        "ltr:QL:mu": QL_MU,
        "ltr:trainingQueryFile": TRAIN_QUERY_PATH,
        "ltr:trainingQrelsFile": TRAIN_QREL_PATH,
        "ltr:trainingFeatureVectorsFile": f"{output_dir}/{exp_id}.LtrTrain",
        "ltr:modelFile": f"{output_dir}/{exp_id}.Model",
        "ltr:testingFeatureVectorsFile": f"{output_dir}/{exp_id}.LtrTest",
        "ltr:testingDocumentScores": f"{output_dir}/{exp_id}.DocScore",
        "ltr:featureDisable": disabled_csv(enabled_features)
    }

    if toolkit == "SVMRank":
        task.update({
            "ltr:toolkit": "SVMRank",
            "ltr:svmRankLearnPath": "INPUT_DIR/svm_rank_learn",
            "ltr:svmRankClassifyPath": "INPUT_DIR/svm_rank_classify",
            "ltr:svmRankParamC": "0.001"
        })
    else:
        task.update({
            "ltr:toolkit": "RankLib",
            "ltr:RankLib:model": str(ranklib_model)
        })
        if str(ranklib_model) == "4":
            task["ltr:RankLib:metric2t"] = "MAP"

    return task


def baseline_param(query_path, output_path):
    return {
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
    }


def rerank_param(query_path, output_dir, exp_id, toolkit, enabled_features, ranklib_model=None, use_prf=False):
    config = {
        "indexPath": INDEX_PATH,
        "queryFilePath": query_path,
        "task_1:ranker": {
            "type": "inRankFile",
            "inRankFile:Path": BASE_INRANK_PATH
        }
    }

    next_task = 2
    if use_prf:
        config[f"task_{next_task}:rewriter"] = {
            "type": "prf",
            "prf:algorithm": "rm3",
            "prf:numDocs": "50",
            "prf:numTerms": "20",
            "prf:expansionFieldIn": "body",
            "prf:expansionFieldOut": "body",
            "prf:expansionQueryFile": f"{output_dir}/{exp_id}.qryOut",
            "prf:rm3:origWeight": "0.0"
        }
        next_task += 1

    config[f"task_{next_task}:reranker"] = ltr_task(
        exp_id, output_dir, toolkit, enabled_features, ranklib_model, use_prf)
    next_task += 1
    config[f"task_{next_task}:output"] = {
        "type": "trec_eval",
        "outputPath": f"{output_dir}/{exp_id}.teIn",
        "outputLength": "1000"
    }

    return config


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def generate_exp1():
    experiments = [
        ("HW3-Exp-1.1a", EXP1_DIR, OUTPUT1_DIR, None, None, None, False, None),
        ("HW3-Exp-1.1b", EXP1_DIR, OUTPUT1_DIR, "SVMRank", [5, 6, 8, 9, 11, 12, 14, 15], None, False, None),
        ("HW3-Exp-1.1c", EXP1_DIR, OUTPUT1_DIR, "SVMRank", list(range(5, 17)), None, False, None),
        ("HW3-Exp-1.1d", EXP1_DIR, OUTPUT1_DIR, "SVMRank", list(range(1, 17)), None, False, None),
        ("HW3-Exp-1.1e", EXP1_DIR, OUTPUT1_DIR, "SVMRank", list(range(1, 17)), None, True, None),
        ("HW3-Exp-1.2b", EXP1_DIR, OUTPUT1_DIR, "RankLib", [5, 6, 8, 9, 11, 12, 14, 15], 4, False, None),
        ("HW3-Exp-1.2c", EXP1_DIR, OUTPUT1_DIR, "RankLib", list(range(5, 17)), 4, False, None),
        ("HW3-Exp-1.2d", EXP1_DIR, OUTPUT1_DIR, "RankLib", list(range(1, 17)), 4, False, None),
        ("HW3-Exp-1.2e", EXP1_DIR, OUTPUT1_DIR, "RankLib", list(range(1, 17)), 4, True, None),
        ("HW3-Exp-1.3b", EXP1_DIR, OUTPUT1_DIR, "RankLib", [5, 6, 8, 9, 11, 12, 14, 15], 7, False, None),
        ("HW3-Exp-1.3c", EXP1_DIR, OUTPUT1_DIR, "RankLib", list(range(5, 17)), 7, False, None),
        ("HW3-Exp-1.3d", EXP1_DIR, OUTPUT1_DIR, "RankLib", list(range(1, 17)), 7, False, None),
        ("HW3-Exp-1.3e", EXP1_DIR, OUTPUT1_DIR, "RankLib", list(range(1, 17)), 7, True, None),
    ]

    for exp_id, exp_dir, output_dir, toolkit, enabled, ranklib_model, use_prf, _ in experiments:
        query_path = f"{exp_dir}/{exp_id}.qry"
        copy_query_file(query_path)
        param_path = f"{exp_dir}/{exp_id}.param"
        if exp_id == "HW3-Exp-1.1a":
            write_json(param_path, baseline_param(query_path, f"{output_dir}/{exp_id}.teIn"))
        else:
            write_json(param_path, rerank_param(
                query_path, output_dir, exp_id, toolkit, enabled, ranklib_model, use_prf))


def generate_exp2():
    setups = {
        "a": list(range(1, 17)),
        "b": list(range(1, 18)),
        "c": list(range(1, 17)) + [18],
        "d": list(range(1, 17)) + [19],
        "e": list(range(1, 17)) + [20],
        "f": list(range(1, 21)),
    }
    groups = [("1", "SVMRank", None), ("2", "RankLib", 4), ("3", "RankLib", 7)]

    for prefix, toolkit, ranklib_model in groups:
        for suffix, enabled in setups.items():
            exp_id = f"HW3-Exp-2.{prefix}{suffix}"
            query_path = f"{EXP2_DIR}/{exp_id}.qry"
            copy_query_file(query_path)
            param_path = f"{EXP2_DIR}/{exp_id}.param"
            write_json(param_path, rerank_param(
                query_path, OUTPUT2_DIR, exp_id, toolkit, enabled, ranklib_model, False))


def generate_exp3():
    combinations = {
        "HW3-Exp-3.1a": list(range(1, 21)),
        "HW3-Exp-3.1b": [4, 5, 8, 17],
        "HW3-Exp-3.1c": [4, 5, 6, 8, 9, 17],
        "HW3-Exp-3.1d": [4, 5, 6, 17],
        "HW3-Exp-3.1e": [4, 5, 8],
    }

    for exp_id, enabled in combinations.items():
        query_path = f"{EXP3_DIR}/{exp_id}.qry"
        copy_query_file(query_path)
        param_path = f"{EXP3_DIR}/{exp_id}.param"
        write_json(param_path, rerank_param(
            query_path, OUTPUT3_DIR, exp_id, "SVMRank", enabled, None, False))


def main():
    ensure_dirs()
    copy_shared_inputs()
    generate_exp1()
    generate_exp2()
    generate_exp3()
    print("Generated HW3 experiment files in EXP_DIR.")


if __name__ == "__main__":
    main()
