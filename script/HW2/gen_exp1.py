import json
import os

# --- Configuration ---
INDEX_PATH = "INPUT_DIR/index-cw09"
QUERY_FILE = "EXP_DIR/HW2_EXP1/HW2-Exp-Bow.qry" 

# Directory to save the generated .param files 
PARAM_DIR = "EXP_DIR/HW2_EXP1"

# Directory to save the output files (.teIn, .qryOut)
OUTPUT_DIR = "OUTPUT_DIR/HW2-EXP1"

# Experiment Settings
NUM_DOCS_LIST = [10, 20, 30, 50]
SUFFIXES = ['b', 'c', 'd', 'e'] 
NUM_TERMS = 10 
BM25_K1 = "1.2"
BM25_B = "0.75"

# Ensure directories exist
if not os.path.exists(PARAM_DIR):
    os.makedirs(PARAM_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. Generate Baseline: HW2-Exp-1.1a
# ==========================================
baseline_id = "HW2-Exp-1.1a"
# Save .param file to PARAM_DIR
baseline_filename = f"{PARAM_DIR}/{baseline_id}.param"
baseline_tein_path = f"{OUTPUT_DIR}/{baseline_id}.teIn"

baseline_config = {
    "indexPath": INDEX_PATH,
    "queryFilePath": QUERY_FILE,
    "task_1:ranker": {
        "type": "BM25",
        "BM25:k_1": BM25_K1,
        "BM25:b": BM25_B,
        "outputLength": "1000"
    },
    "task_2:output": {
        "type": "trec_eval",
        "outputPath": baseline_tein_path,
        "outputLength": "1000"
    }
}

with open(baseline_filename, 'w') as f:
    json.dump(baseline_config, f, indent=4)
print(f"Generated {baseline_filename}")

# ==========================================
# 2. Generate Okapi Experiment Group: HW2-Exp-1.1[b-e]
# ==========================================
for i, num_docs in enumerate(NUM_DOCS_LIST):
    suffix = SUFFIXES[i]
    exp_id = f"HW2-Exp-1.1{suffix}" 
    # Save .param file to PARAM_DIR
    param_filename = f"{PARAM_DIR}/{exp_id}.param"
    
    config = {
        "indexPath": INDEX_PATH,
        "queryFilePath": QUERY_FILE,
        "task_1:ranker": {
            "type": "inRankFile",
            "inRankFile:Path": baseline_tein_path
        },
        "task_2:rewriter": {
            "type": "prf",
            "prf:algorithm": "okapi",
            "prf:numDocs": str(num_docs),
            "prf:numTerms": str(NUM_TERMS),
            "prf:expansionFieldIn": "body",
            "prf:expansionFieldOut": "body",
            "prf:expansionQueryFile": f"{OUTPUT_DIR}/{exp_id}.qryOut",
            "prf:rm3:origWeight": "0.0"
        },
        "task_3:ranker": {
            "type": "BM25",
            "BM25:k_1": BM25_K1,
            "BM25:b": BM25_B,
            "outputLength": "1000"
        },
        "task_4:output": {
            "type": "trec_eval",
            "outputPath": f"{OUTPUT_DIR}/{exp_id}.teIn",
            "outputLength": "1000"
        }
    }
    with open(param_filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated {param_filename}")

# ==========================================
# 3. Generate RM3 Experiment Group: HW2-Exp-1.2[b-e]
# ==========================================
for i, num_docs in enumerate(NUM_DOCS_LIST):
    suffix = SUFFIXES[i]
    exp_id = f"HW2-Exp-1.2{suffix}" 
    # Save .param file to PARAM_DIR
    param_filename = f"{PARAM_DIR}/{exp_id}.param"
    
    config = {
        "indexPath": INDEX_PATH,
        "queryFilePath": QUERY_FILE,
        "task_1:ranker": {
            "type": "inRankFile",
            "inRankFile:Path": baseline_tein_path
        },
        "task_2:rewriter": {
            "type": "prf",
            "prf:algorithm": "rm3",
            "prf:numDocs": str(num_docs),
            "prf:numTerms": str(NUM_TERMS),
            "prf:expansionFieldIn": "body",
            "prf:expansionFieldOut": "body",
            "prf:expansionQueryFile": f"{OUTPUT_DIR}/{exp_id}.qryOut",
            "prf:rm3:origWeight": "0.0" 
        },
        "task_3:ranker": {
            "type": "BM25",
            "BM25:k_1": BM25_K1,
            "BM25:b": BM25_B,
            "outputLength": "1000"
        },
        "task_4:output": {
            "type": "trec_eval",
            "outputPath": f"{OUTPUT_DIR}/{exp_id}.teIn",
            "outputLength": "1000"
        }
    }
    with open(param_filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated {param_filename}")