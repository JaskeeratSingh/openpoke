"""The seven benchmark tests, plus the order check.

Each test runs all of its cases through one strategy in one world and returns one graded
row per case. run_benchmark.py calls them for every strategy and world you ask for.

These functions are named test_* to match the README, but this file is not a pytest file:
pytest only collects files named test_*.py inside tests/, so running `pytest` never
spends money on API calls.
"""

import asyncio

from . import harness
from .dataset import load_cases


async def run_test(test_name, strategy, experiment, size=None, limit=None, **options):
    """Shared body of every test: load the test's cases and run them all."""
    cases = [case for case in load_cases(experiment) if case["test"] == test_name]
    if limit is not None:
        cases = cases[:limit]
    jobs = [harness.run_case(case, strategy, size, **options) for case in cases]
    return await asyncio.gather(*jobs)


async def test_easy_overlap(strategy, experiment, size=None, **options):
    """The request names what the agent is about. The control: every strategy should be near 100%."""
    return await run_test("easy_overlap", strategy, experiment, size, **options)


async def test_semantic_disconnect(strategy, experiment, size=None, **options):
    """Same kind of request, but sharing no words with the agent's name ("a present" -> Present Ideas)."""
    return await run_test("semantic_disconnect", strategy, experiment, size, **options)


async def test_related_context(strategy, experiment, size=None, **options):
    """A new task the agent wasn't made for, but it holds a fact the task needs, so reuse it."""
    return await run_test("related_context", strategy, experiment, size, **options)


async def test_misleading_name(strategy, experiment, size=None, **options):
    """The agent's name comes from its first job; the request is about a later, unrelated job it did."""
    return await run_test("misleading_name", strategy, experiment, size, **options)


async def test_elliptical_reference(strategy, experiment, size=None, **options):
    """The request names nothing ("tell him thursday works"); the last history entry says who. Decay only."""
    return await run_test("elliptical_reference", strategy, experiment, size, **options)


async def test_false_friend(strategy, experiment, size=None, **options):
    """An agent looks related (same first name, same keyword) but holds nothing useful: create a new one."""
    return await run_test("false_friend", strategy, experiment, size, **options)


async def test_unrelated(strategy, experiment, size=None, **options):
    """Nothing in the roster is close: create a new agent."""
    return await run_test("unrelated", strategy, experiment, size, **options)


ALL_TESTS = {
    "easy_overlap": test_easy_overlap,
    "semantic_disconnect": test_semantic_disconnect,
    "related_context": test_related_context,
    "misleading_name": test_misleading_name,
    "elliptical_reference": test_elliptical_reference,
    "false_friend": test_false_friend,
    "unrelated": test_unrelated,
}

ORDER_CHECK_SIZE = 100
ORDER_CHECK_CASES_PER_TEST = 5
ORDER_CHECK_SEEDS = [1, 2, 3]


async def check_order_stability(strategy, **options):
    """Does the decision change when only the order of the agent list changes?

    Takes the first 5 Growth cases of each test (30 cases) at 100 agents and runs each one
    3 times with the list shuffled differently. report.py counts how many cases changed outcome.
    S2 has no list, so it is skipped.
    """
    if strategy == "S2":
        return []
    cases = []
    for test_name in ALL_TESTS:
        test_cases = [case for case in load_cases("growth") if case["test"] == test_name]
        cases.extend(test_cases[:ORDER_CHECK_CASES_PER_TEST])
    jobs = []
    for case in cases:
        for seed in ORDER_CHECK_SEEDS:
            jobs.append(harness.run_case(case, strategy, ORDER_CHECK_SIZE, shuffle_seed=seed, **options))
    return await asyncio.gather(*jobs)
