"""Command line entry point: runs benchmark tests and saves + prints the results.

Examples (run from the repo root):
    python -m evals.run_benchmark                                  # everything
    python -m evals.run_benchmark --experiment growth --sizes 10 100 --strategies S0 S1
    python -m evals.run_benchmark --tests easy_overlap --limit 3    # a quick smoke run
    python -m evals.run_benchmark --order-check                     # only the order check
    python -m evals.run_benchmark --parallel 32                     # more model calls at once

Results go to evals/results/<date-time>.jsonl, one JSON row per case.
"""

import argparse
import asyncio
import json
from datetime import datetime

from . import bench
from .dataset import EVALS_DIR, GROWTH_SIZES
from .report import print_report
from .strategies import STRATEGIES

DEFAULT_PARALLEL_CALLS = 16


def parse_args():
    parser = argparse.ArgumentParser(description="Run the agent routing benchmark.")
    parser.add_argument("--experiment", choices=["growth", "decay", "both"], default="both")
    parser.add_argument("--sizes", type=int, nargs="+", default=GROWTH_SIZES, help="Growth roster sizes")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument("--tests", nargs="+", default=list(bench.ALL_TESTS), choices=list(bench.ALL_TESTS))
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases of each test")
    parser.add_argument("--order-check", action="store_true", help="run only the order check")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL_CALLS,
                        help="how many model calls may run at the same time")
    return parser.parse_args()


async def run(args):
    """Run every requested (test, strategy, world) combination and return all rows."""
    limiter = asyncio.Semaphore(args.parallel)
    jobs = []

    if args.order_check:
        for strategy in args.strategies:
            jobs.append(bench.check_order_stability(strategy, limiter=limiter))
    else:
        worlds = []
        if args.experiment in ("growth", "both"):
            worlds += [("growth", size) for size in args.sizes]
        if args.experiment in ("decay", "both"):
            worlds.append(("decay", None))

        for test_name in args.tests:
            test_function = bench.ALL_TESTS[test_name]
            for strategy in args.strategies:
                for experiment, size in worlds:
                    if test_name == "elliptical_reference" and experiment == "growth":
                        continue  # elliptical cases only exist in Decay
                    jobs.append(test_function(strategy, experiment, size, limit=args.limit, limiter=limiter))

    rows = []
    for finished in asyncio.as_completed(jobs):
        rows.extend(await finished)
        print(f"  {len(rows)} cases done", end="\r", flush=True)
    print()
    return rows


def main():
    args = parse_args()
    rows = asyncio.run(run(args))

    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print_report(rows)
    print(f"Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
