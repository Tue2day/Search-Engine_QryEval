import json
import os

# ==============================================================================
# Experiment 3: The "Trust vs. Noise" Hypothesis
# ==============================================================================

# Paths
INDEX_PATH = "INPUT_DIR/index-cw09"
QUERY_FILE = "EXP_DIR/HW2_EXP1/HW2-Exp-Bow.qry"
# 注意：所有 RM3 实验的输入都是最原始的 BM25 排名结果 (1.1a)
BASELINE_INRANK = "OUTPUT_DIR/HW2-EXP1/HW2-Exp-1.1a.teIn"

# Output Directories
PARAM_DIR = "EXP_DIR/HW2_EXP3"
OUTPUT_DIR = "OUTPUT_DIR/HW2-EXP3"

if not os.path.exists(PARAM_DIR): os.makedirs(PARAM_DIR)
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Configurations for Table Columns 2-6
configs = [
    # Config 1: Body Standard (Compare 0.5 vs Baseline 0.0)
    {"id": "HW2-Exp-3.1b", "field": "body",     "weight": "0.5", "desc": "Body, Balanced (0.5)"},
    
    # Config 2: Body Aggressive (Trust Noisy Feedback)
    {"id": "HW2-Exp-3.1c", "field": "body",     "weight": "0.2", "desc": "Body, Aggressive (0.2)"},
    
    # Config 3: Body Conservative (Distrust Noisy Feedback)
    {"id": "HW2-Exp-3.1d", "field": "body",     "weight": "0.8", "desc": "Body, Conservative (0.8)"},
    
    # Config 4: Keywords Aggressive (Trust Clean Feedback) -> HYPOTHESIS WINNER
    {"id": "HW2-Exp-3.1e", "field": "keywords", "weight": "0.2", "desc": "Keywords, Aggressive (0.2)"},
    
    # Config 5: Keywords Conservative (Distrust Clean Feedback)
    {"id": "HW2-Exp-3.1f", "field": "keywords", "weight": "0.8", "desc": "Keywords, Conservative (0.8)"},
]

print(f"Generating parameters for Experiment 3 ({len(configs)} configs)...")

for conf in configs:
    param_filename = f"{PARAM_DIR}/{conf['id']}.param"
    
    config = {
        "indexPath": INDEX_PATH,
        "queryFilePath": QUERY_FILE,
        
        # Task 1: Always start with BM25 Baseline
        "task_1:ranker": {
            "type": "inRankFile",
            "inRankFile:Path": BASELINE_INRANK
        },
        
        # Task 2: Rewriter (RM3)
        "task_2:rewriter": {
            "type": "prf",
            "prf:algorithm": "rm3",
            "prf:numDocs": "10",         # Fixed as per Exp 1
            "prf:numTerms": "10",        # Fixed as per Exp 1
            "prf:expansionFieldIn": conf['field'],
            "prf:expansionFieldOut": "body",
            "prf:expansionQueryFile": f"{OUTPUT_DIR}/{conf['id']}.qryOut",
            "prf:rm3:origWeight": conf['weight']
        },
        
        # Task 3: Ranker
        "task_3:ranker": {
            "type": "BM25",
            "BM25:k_1": "1.2",
            "BM25:b": "0.75",
            "outputLength": "1000"
        },
        
        # Task 4: Output
        "task_4:output": {
            "type": "trec_eval",
            "outputPath": f"{OUTPUT_DIR}/{conf['id']}.teIn",
            "outputLength": "1000"
        }
    }
    
    with open(param_filename, 'w') as f:
        json.dump(config, f, indent=4)
        
print("Done. Parameter files created in", PARAM_DIR)