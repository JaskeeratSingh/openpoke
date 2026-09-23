"""Turns result rows into the tables in the README: the four metrics plus diagnostics.

run_benchmark.py prints this after a run. You can also re-print an old run:
    python -m evals.report evals/results/<file>.jsonl
"""

import json
import sys
from collections import defaultdict

from .dataset import REUSE_TESTS


def is_reuse_case(row):
    return row["test"] in REUSE_TESTS


def metrics(rows):
    """The four headline metrics (plus a few counts) for a group of rows.

    Rows whose API call failed ("error") and rows where the IA asked the user a question
    ("asked_user") are left out of every rate and counted separately.
    """
    scored = [row for row in rows if row["outcome"] not in ("error", "asked_user")]
    reuse = [row for row in scored if is_reuse_case(row)]
    create = [row for row in scored if not is_reuse_case(row)]

    def share(count, total):
        return None if total == 0 else count / total

    ranks = [row["gold_rank"] for row in reuse if row["strategy"] in ("S2", "S3")]
    return {
        "reuse_accuracy": share(sum(row["outcome"] == "reuse_correct" for row in reuse), len(reuse)),
        "create_accuracy": share(sum(row["outcome"] == "created" for row in create), len(create)),
        "wrong_reuse_rate": share(sum(row["outcome"] == "reuse_wrong" for row in scored), len(scored)),
        "unnecessary_new_rate": share(sum(row["outcome"] == "created" for row in reuse), len(reuse)),
        "no_dispatch": sum(row["outcome"] == "no_dispatch" for row in scored),
        "asked_user": sum(row["outcome"] == "asked_user" for row in rows),
        "errors": sum(row["outcome"] == "error" for row in rows),
        "recall_at_1": share(sum(rank == 1 for rank in ranks), len(ranks)) if ranks else None,
        "recall_at_5": share(sum(rank is not None and rank <= 5 for rank in ranks), len(ranks)) if ranks else None,
        "avg_prompt_tokens": share(sum(row["tokens_in"] for row in scored), sum(row["model_calls"] for row in scored)),
        "cost": sum(row["cost"] for row in rows),
        "cases": len(rows),
    }


def fmt(value, percent=True):
    if value is None:
        return "-"
    if percent:
        return f"{value * 100:.0f}%"
    return f"{value:,.0f}"


def print_table(title, header, lines):
    print(f"\n{title}")
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *lines)]
    print("  ".join(str(cell).ljust(width) for cell, width in zip(header, widths)))
    for line in lines:
        print("  ".join(str(cell).ljust(width) for cell, width in zip(line, widths)))


def metric_cells(m):
    return [
        fmt(m["reuse_accuracy"]), fmt(m["create_accuracy"]), fmt(m["wrong_reuse_rate"]),
        fmt(m["unnecessary_new_rate"]), m["no_dispatch"], m["asked_user"], m["errors"],
        fmt(m["recall_at_1"]), fmt(m["recall_at_5"]), fmt(m["avg_prompt_tokens"], percent=False),
        f"${m['cost']:.3f}", m["cases"],
    ]


METRIC_HEADER = ["reuse acc", "create acc", "wrong reuse", "unnecessary new", "no dispatch", "asked", "errors",
                 "recall@1", "recall@5", "prompt tokens", "cost", "cases"]


def print_report(rows):
    """Print one table per experiment, a per-test breakdown, and the order check."""
    main_rows = [row for row in rows if row.get("shuffle_seed") is None]
    order_rows = [row for row in rows if row.get("shuffle_seed") is not None]

    growth = [row for row in main_rows if row["experiment"] == "growth"]
    if growth:
        groups = defaultdict(list)
        for row in growth:
            groups[(row["strategy"], row["size"])].append(row)
        lines = [[strategy, size] + metric_cells(metrics(group)) for (strategy, size), group in sorted(groups.items())]
        print_table("GROWTH (by strategy and roster size)", ["strategy", "EAs"] + METRIC_HEADER, lines)

    decay = [row for row in main_rows if row["experiment"] == "decay"]
    if decay:
        groups = defaultdict(list)
        for row in decay:
            level = row["level"] if is_reuse_case(row) else "create cases"
            groups[(row["strategy"], level)].append(row)
        lines = [[strategy, level] + metric_cells(metrics(group)) for (strategy, level), group in sorted(groups.items())]
        print_table("DECAY (100 EAs, by strategy and level)", ["strategy", "level"] + METRIC_HEADER, lines)

    if main_rows:
        groups = defaultdict(list)
        for row in main_rows:
            groups[(row["experiment"], row["strategy"], row["test"])].append(row)
        lines = []
        for (experiment, strategy, test), group in sorted(groups.items()):
            wanted = "reuse_correct" if is_reuse_case(group[0]) else "created"
            scored = [row for row in group if row["outcome"] not in ("error", "asked_user")]
            correct = sum(row["outcome"] == wanted for row in scored)
            wrong = sum(row["outcome"] == "reuse_wrong" for row in scored)
            lines.append([experiment, strategy, test, f"{correct}/{len(scored)}", wrong])
        print_table("PER TEST (all sizes pooled)", ["experiment", "strategy", "test", "correct", "wrong reuse"], lines)

    if order_rows:
        by_case = defaultdict(list)
        for row in order_rows:
            by_case[(row["strategy"], row["case_id"])].append(row["outcome"])
        changed = defaultdict(lambda: [0, 0])
        for (strategy, _), outcomes in by_case.items():
            changed[strategy][1] += 1
            if len(set(outcomes)) > 1:
                changed[strategy][0] += 1
        lines = [[strategy, f"{count}/{total}"] for strategy, (count, total) in sorted(changed.items())]
        print_table("ORDER CHECK (cases whose outcome changed across 3 shuffles)", ["strategy", "changed"], lines)

    print(f"\nTotal cost: ${sum(row['cost'] for row in rows):.3f}")


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    print_report(load_rows(sys.argv[1]))
