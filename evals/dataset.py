"""Loads the benchmark dataset and puts together the roster and history for each case.

Everything else in evals/ reads data through this file: build_dataset.py, harness.py,
bench.py and tests/test_eval.py.
"""

import json
import random
import re
import tempfile
from functools import lru_cache
from html import escape
from pathlib import Path

from server.services.conversation.summarization.working_memory_log import WorkingMemoryLog

EVALS_DIR = Path(__file__).parent
WRITTEN_DIR = EVALS_DIR / "data" / "written"
BUILT_DIR = EVALS_DIR / "data" / "built"

GROWTH_SIZES = [10, 50, 100, 250, 500, 1000]
DECAY_SIZE = 100
DEV_SIZE = 100

REUSE_TESTS = ["easy_overlap", "semantic_disconnect", "related_context", "misleading_name", "elliptical_reference"]
CREATE_TESTS = ["false_friend", "unrelated"]


def load_json(path):
    """Read a JSON file. Used for every data file in evals/data/."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    """Write a JSON file, creating its folder if needed. Used by build_dataset.py."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# lru_cache means each file is read from disk only once, however many cases use it.
@lru_cache(maxsize=None)
def load_main_agents():
    """The hand-written agents (targets, false friends and anchors) from agents.json."""
    return load_json(WRITTEN_DIR / "agents.json")


@lru_cache(maxsize=None)
def load_fillers():
    """The generated filler agents that pad rosters up to their size (built by build_dataset.py)."""
    return load_json(BUILT_DIR / "fillers.json")


@lru_cache(maxsize=None)
def agents_by_name():
    """Every agent, main and filler, looked up by its name."""
    return {agent["name"]: agent for agent in load_main_agents() + load_fillers()}


@lru_cache(maxsize=None)
def load_decay_levels():
    """Which level (visible / summarized / gone) each agent ended up at in the Decay history."""
    return load_json(BUILT_DIR / "decay_levels.json")


def load_cases(experiment):
    """Return the cases for one experiment: "growth", "decay" or "dev".

    cases.json stores each case once, with a list of the experiments it belongs to.
    This copies it once per experiment, gives it an id like "growth-easy_overlap-01",
    and, for Decay, adds the level of the agent it is about.
    """
    if experiment == "dev":
        cases = []
        for written in load_json(WRITTEN_DIR / "dev_cases.json"):
            case = dict(written)
            case["experiment"] = "dev"
            case["level"] = None
            cases.append(case)
        return cases

    cases = []
    for written in load_json(WRITTEN_DIR / "cases.json"):
        if experiment not in written["experiments"]:
            continue
        case = dict(written)
        case["experiment"] = experiment
        case["id"] = f"{experiment}-{written['id']}"
        case["level"] = None
        if experiment == "decay" and case["answer"] is not None:
            if case["test"] == "elliptical_reference":
                case["level"] = "visible"
            else:
                case["level"] = load_decay_levels()[case["answer"]]
        cases.append(case)
    return cases


def roster_for(case, size=None):
    """The list of agents the interaction agent can choose from for this case.

    Decay: always the same 100 agents (all hand-written agents + fillers).
    Growth and dev: the 3 anchors (the agents the Growth chat talks about), the case's own
    agents, and fillers up to `size`. The roster is shuffled the same way for every size, so
    a bigger roster is always the smaller one plus more fillers.
    """
    main_agents = load_main_agents()
    fillers = load_fillers()

    if case["experiment"] == "decay":
        roster = main_agents + fillers[: DECAY_SIZE - len(main_agents)]
        random.Random("decay").shuffle(roster)
        return roster

    anchors = [agent for agent in main_agents if agent["role"] == "anchor"]
    case_agents = [agents_by_name()[name] for name in case["roster_agents"]]
    fixed = anchors + case_agents

    largest = fixed + fillers[: max(GROWTH_SIZES) - len(fixed)]
    random.Random(case["id"]).shuffle(largest)

    keep = {agent["name"] for agent in fixed + fillers[: size - len(fixed)]}
    return [agent for agent in largest if agent["name"] in keep]


def render_entries(entries):
    """Turn [{"tag", "timestamp", "text"}, ...] into history text, exactly as production renders it.

    It writes the entries into a throwaway WorkingMemoryLog (the production class) and asks
    it for the transcript, so the formatting is the real one.
    """
    with tempfile.TemporaryDirectory() as folder:
        memory = WorkingMemoryLog(Path(folder) / "working_memory.log")
        for entry in entries:
            memory.append_entry(entry["tag"], entry["text"], entry["timestamp"])
        return memory.render_transcript()


@lru_cache(maxsize=None)
def _read_text(path):
    return Path(path).read_text(encoding="utf-8")


def history_for(case):
    """The chat history the interaction agent sees for this case.

    Growth and dev: the same short chat about the 3 anchors. It never mentions a case's
    agents, so everything the IA knows about them comes from the agent list itself.
    Decay: the long shared history, plus the case's extra lines if it has any.
    """
    if case["experiment"] in ("growth", "dev"):
        return _read_text(BUILT_DIR / "growth_history.txt")

    history = _read_text(BUILT_DIR / "decay_history.txt")
    if case.get("history_suffix"):
        history = history + "\n" + render_entries(case["history_suffix"])
    return history


def measure_level(agent, history_text):
    """How much the history still says about an agent: "visible", "summarized" or "gone".

    visible:    its result message is still in the history, word for word
    summarized: only the summary mentions it (one of its key terms appears there)
    gone:       the history says nothing about it
    Used by build_dataset.py to label Decay cases, and by the checks to confirm the labels.
    """
    summary_match = re.search(r"<conversation_summary>(.*?)</conversation_summary>", history_text, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1)
        tail = history_text[summary_match.end():]
    else:
        summary = ""
        tail = history_text

    result_prefix = escape(f"[SUCCESS] {agent['name']}:", quote=False)
    if result_prefix in tail:
        return "visible"
    for term in agent["key_terms"]:
        if escape(term, quote=False).lower() in summary.lower():
            return "summarized"
    return "gone"


def contains_phrase(text, phrase):
    """True if `phrase` appears in `text` as whole words, ignoring case ("ink" does not match "think")."""
    pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def card_text(agent):
    """The agent's card as plain text lines. Shown to the IA by S1/S3 and embedded by S2/S3."""
    card = agent["card"]
    return (
        f"purpose: {card['purpose']}\n"
        f"people: {card['people']}\n"
        f"details: {card['details']}\n"
        f"state: {card['state']}"
    )
