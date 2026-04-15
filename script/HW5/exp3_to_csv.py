import ast
import csv
import os


OUTPUT_DIR = "OUTPUT_DIR/HW5-EXP3"
BASELINE_PREFIX = "OUTPUT_DIR/HW5-EXP1/HW5-Exp-1.2c"
LONG_CSV_PATH = f"{OUTPUT_DIR}/HW5_Exp3_Metrics.csv"

LONG_EXPERIMENTS = [
    ("HW5-Exp-3.1a", "BM25 -> RAG(bestp)", BASELINE_PREFIX),
    ("HW5-Exp-3.1b", "BM25 -> RAG(firstp)", f"{OUTPUT_DIR}/HW5-Exp-3.1b"),
    ("HW5-Exp-3.1c", "BM25 -> BERT-6L -> RAG", f"{OUTPUT_DIR}/HW5-Exp-3.1c"),
    ("HW5-Exp-3.1d", "BM25 -> BERT-6L -> RAG (Prompt 3)", f"{OUTPUT_DIR}/HW5-Exp-3.1d"),
    ("HW5-Exp-3.1e", "BM25 -> RAG (psgCnt=10)", f"{OUTPUT_DIR}/HW5-Exp-3.1e"),
    ("HW5-Exp-3.1f", "dense -> BERT-6L -> RAG", f"{OUTPUT_DIR}/HW5-Exp-3.1f"),
    ("HW5-Exp-3.1g", "BM25 + dense merge top 100 -> RAG", f"{OUTPUT_DIR}/HW5-Exp-3.1g"),
]

TABLE_SPEC = {
    "filename": "HW5_Exp3_Table_1.csv",
    "title": "Custom Experiments",
    "experiments": [
        ("HW5-Exp-3.1a", "Baseline\n(n/a)", BASELINE_PREFIX),
        ("HW5-Exp-3.1b", "Custom 1\nExp-3.1b", f"{OUTPUT_DIR}/HW5-Exp-3.1b"),
        ("HW5-Exp-3.1c", "Custom 2\nExp-3.1c", f"{OUTPUT_DIR}/HW5-Exp-3.1c"),
        ("HW5-Exp-3.1d", "Custom 3\nExp-3.1d", f"{OUTPUT_DIR}/HW5-Exp-3.1d"),
        ("HW5-Exp-3.1e", "Custom 4\nExp-3.1e", f"{OUTPUT_DIR}/HW5-Exp-3.1e"),
        ("HW5-Exp-3.1f", "Custom 5\nExp-3.1f", f"{OUTPUT_DIR}/HW5-Exp-3.1f"),
        ("HW5-Exp-3.1g", "Custom 6\nExp-3.1g", f"{OUTPUT_DIR}/HW5-Exp-3.1g"),
    ]
}


def read_runtime(prefix):
    log_path = f"{prefix}.log"
    if not os.path.exists(log_path):
        return("")

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Time:"):
                return(line.split("Time:", 1)[1].strip())
    return("")


def read_te_metrics(prefix):
    metrics = {"MRR": "", "P@1": "", "P@5": ""}
    te_path = f"{prefix}.teOut"
    if not os.path.exists(te_path):
        return(metrics)

    with open(te_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            metric_name, _, value = parts
            if metric_name == "recip_rank":
                metrics["MRR"] = value
            elif metric_name == "P_1":
                metrics["P@1"] = value
            elif metric_name == "P_5":
                metrics["P@5"] = value
    return(metrics)


def read_qa_metrics(prefix):
    metrics = {"exact": "", "f1": ""}
    qa_path = f"{prefix}.qaOut"
    if not os.path.exists(qa_path):
        return(metrics)

    last_dict = None
    with open(qa_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                last_dict = line

    if last_dict is None:
        return(metrics)

    data = ast.literal_eval(last_dict)
    metrics["exact"] = data.get("exact_match", "")
    metrics["f1"] = data.get("f1", "")
    return(metrics)


def format_exact_or_f1(value):
    if value == "":
        return("")
    return(f"{float(value):.2f}")


def format_rank_metric(value):
    if value == "":
        return("")
    return(f"{float(value):.4f}")


def collect_metrics(prefix):
    qa = read_qa_metrics(prefix)
    te = read_te_metrics(prefix)
    runtime = read_runtime(prefix)
    return({
        "exact": format_exact_or_f1(qa["exact"]),
        "f1": format_exact_or_f1(qa["f1"]),
        "MRR": format_rank_metric(te["MRR"]),
        "P@1": format_rank_metric(te["P@1"]),
        "P@5": format_rank_metric(te["P@5"]),
        "time": runtime
    })


def write_long_csv():
    with open(LONG_CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "exp_id", "pipeline", "exact", "f1", "MRR", "P@1", "P@5", "time"
        ])
        for exp_id, pipeline, prefix in LONG_EXPERIMENTS:
            metrics = collect_metrics(prefix)
            writer.writerow([
                exp_id,
                pipeline,
                metrics["exact"],
                metrics["f1"],
                metrics["MRR"],
                metrics["P@1"],
                metrics["P@5"],
                metrics["time"]
            ])


def write_report_table():
    path = f'{OUTPUT_DIR}/{TABLE_SPEC["filename"]}'
    metrics_by_exp = {
        exp_id: collect_metrics(prefix)
        for exp_id, _, prefix in TABLE_SPEC["experiments"]
    }

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([TABLE_SPEC["title"]] + [""] * len(TABLE_SPEC["experiments"]))
        writer.writerow(["Metrics"] + [label for _, label, _ in TABLE_SPEC["experiments"]])
        for metric_name in ["exact", "f1", "MRR", "P@1", "P@5", "time"]:
            row = [metric_name]
            for exp_id, _, _ in TABLE_SPEC["experiments"]:
                row.append(metrics_by_exp[exp_id][metric_name])
            writer.writerow(row)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_long_csv()
    write_report_table()


if __name__ == "__main__":
    main()
