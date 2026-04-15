import ast
import csv
import os


OUTPUT_DIR = "OUTPUT_DIR/HW5-EXP2"
LONG_CSV_PATH = f"{OUTPUT_DIR}/HW5_Exp2_Metrics.csv"

LONG_EXPERIMENTS = [
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

TABLES = [
    {
        "filename": "HW5_Exp2_Table_1.csv",
        "title": "BM25 retrieval",
        "experiments": [
            ("HW5-Exp-2.1a", "Prompt 1\nExp-2.1a"),
            ("HW5-Exp-2.1b", "Prompt 2\nExp-2.1b"),
            ("HW5-Exp-2.1c", "Prompt 3\nExp-2.1c"),
            ("HW5-Exp-2.1d", "Prompt 4\nExp-2.1d"),
            ("HW5-Exp-2.1e", "Prompt 5\nExp-2.1e"),
            ("HW5-Exp-2.1f", "Prompt 6\nExp-2.1f"),
        ]
    },
    {
        "filename": "HW5_Exp2_Table_2.csv",
        "title": "dense retrieval",
        "experiments": [
            ("HW5-Exp-2.2a", "Prompt 1\nExp-2.2a"),
            ("HW5-Exp-2.2b", "Prompt 2\nExp-2.2b"),
            ("HW5-Exp-2.2c", "Prompt 3\nExp-2.2c"),
            ("HW5-Exp-2.2d", "Prompt 4\nExp-2.2d"),
            ("HW5-Exp-2.2e", "Prompt 5\nExp-2.2e"),
            ("HW5-Exp-2.2f", "Prompt 6\nExp-2.2f"),
        ]
    }
]


def read_runtime(exp_id):
    log_path = f"{OUTPUT_DIR}/{exp_id}.log"
    if not os.path.exists(log_path):
        return("")

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Time:"):
                return(line.split("Time:", 1)[1].strip())
    return("")


def read_te_metrics(exp_id):
    metrics = {"MRR": "", "P@1": "", "P@5": ""}
    te_path = f"{OUTPUT_DIR}/{exp_id}.teOut"
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


def read_qa_metrics(exp_id):
    metrics = {"exact": "", "f1": ""}
    qa_path = f"{OUTPUT_DIR}/{exp_id}.qaOut"
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


def collect_metrics(exp_id):
    qa = read_qa_metrics(exp_id)
    te = read_te_metrics(exp_id)
    runtime = read_runtime(exp_id)
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
            "exp_id", "retrieval", "prompt",
            "exact", "f1", "MRR", "P@1", "P@5", "time"
        ])

        for exp_id, retrieval, prompt_id in LONG_EXPERIMENTS:
            metrics = collect_metrics(exp_id)
            writer.writerow([
                exp_id,
                retrieval,
                prompt_id,
                metrics["exact"],
                metrics["f1"],
                metrics["MRR"],
                metrics["P@1"],
                metrics["P@5"],
                metrics["time"]
            ])


def write_report_table(table_spec):
    path = f'{OUTPUT_DIR}/{table_spec["filename"]}'
    metrics_by_exp = {
        exp_id: collect_metrics(exp_id)
        for exp_id, _ in table_spec["experiments"]
    }

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([table_spec["title"]] + [""] * len(table_spec["experiments"]))
        writer.writerow(["Metrics"] + [label for _, label in table_spec["experiments"]])
        for metric_name in ["exact", "f1", "MRR", "P@1", "P@5", "time"]:
            row = [metric_name]
            for exp_id, _ in table_spec["experiments"]:
                row.append(metrics_by_exp[exp_id][metric_name])
            writer.writerow(row)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_long_csv()
    for table_spec in TABLES:
        write_report_table(table_spec)


if __name__ == "__main__":
    main()
