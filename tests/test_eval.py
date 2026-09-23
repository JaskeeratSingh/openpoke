"""Six checks for the routing benchmark in evals/. No network, no API key.

Checks 1-3 look at the dataset: is every case's right answer really the right answer, and
are the Growth and Decay levels what the summarizer actually produced?
Checks 4-6 look at the harness: does it measure production, grade correctly, and touch nothing?

They need the built dataset (python -m evals.build_dataset).
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path

from server.agents.interaction_agent import agent as interaction_agent

from evals import harness, strategies
from evals.dataset import (
    BUILT_DIR,
    GROWTH_SIZES,
    agents_by_name,
    card_text,
    contains_phrase,
    history_for,
    load_cases,
    load_decay_levels,
    load_fillers,
    load_main_agents,
    measure_level,
    roster_for,
)

REPO = Path(__file__).resolve().parent.parent
NAME_STOPWORDS = {"s", "to", "a", "the", "of", "and", "for", "with", "up", "in", "on"}


def all_cases():
    return load_cases("growth") + load_cases("decay") + load_cases("dev")


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# ---------------------------------------------------------------- 1

def test_answers_are_unambiguous():
    """Reuse cases: the answer is in the roster, and no other agent in the roster shares a
    person, company or ID with the request. Create cases: no agent in the roster does, and
    no agent mentions the request's topic. Also: filler first names never clash with names
    used by the hand-written agents.

    Why: a case with two defensible answers scores noise. (The deeper judgement, "is this
    context actually useful", is checked by reading the cases; see the README.)
    """
    problems = []
    for case in all_cases():
        size = max(GROWTH_SIZES) if case["experiment"] != "decay" else None
        roster = roster_for(case, size)
        names = {agent["name"] for agent in roster}
        allowed = {word.lower() for word in case.get("surface_words", [])}

        if case["answer"] is not None and case["answer"] not in names:
            problems.append(f"{case['id']}: answer not in roster")

        for agent in roster:
            if agent["name"] == case["answer"]:
                continue
            for term in agent["key_terms"]:
                if term.lower() not in allowed and contains_phrase(case["message"], term):
                    problems.append(f"{case['id']}: mentions '{term}' from {agent['name']}")
            if case["answer"] is None:
                agent_text = agent["name"] + " " + card_text(agent)
                for word in case.get("avoid_words", []):
                    if contains_phrase(agent_text, word):
                        problems.append(f"{case['id']}: {agent['name']} mentions '{word}'")

    # Fillers must never share a first name with a hand-written agent or a request, or a
    # "which Alex?" case could quietly get a second right answer.
    main_text = " ".join(agent["name"] + " " + card_text(agent) for agent in load_main_agents())
    main_text += " " + " ".join(case["message"] for case in all_cases())
    for filler in load_fillers():
        if filler["card"]["people"] == "None.":
            continue
        first_name = filler["card"]["people"].replace("Dr. ", "").split()[0]
        if first_name.lower() in words(main_text):
            problems.append(f"filler {filler['name']} reuses the first name {first_name}")

    assert problems == []


# ---------------------------------------------------------------- 2

def test_requests_follow_word_rules():
    """Every request contains its must_mention words. semantic_disconnect and misleading_name
    requests share no word with the answer's name. semantic_disconnect requests also avoid
    its key terms. elliptical_reference requests contain no key term of any agent.

    Why: if a "disconnect" request quietly repeats the agent's name, it becomes an easy case
    and flatters the embedding strategies.
    """
    problems = []
    agents = agents_by_name()
    for case in all_cases():
        message = case["message"].lower()
        for phrase in case.get("must_mention", []):
            if phrase.lower() not in message:
                problems.append(f"{case['id']}: missing '{phrase}'")

        if case["test"] in ("semantic_disconnect", "misleading_name"):
            target = agents[case["answer"]]
            shared = (words(target["name"]) - NAME_STOPWORDS) & words(message)
            if shared:
                problems.append(f"{case['id']}: shares {shared} with the agent name")
            if case["test"] == "semantic_disconnect":
                for term in target["key_terms"]:
                    if contains_phrase(message, term):
                        problems.append(f"{case['id']}: contains key term '{term}'")

        if case["test"] == "elliptical_reference":
            for agent in load_main_agents():
                for term in agent["key_terms"]:
                    if contains_phrase(message, term):
                        problems.append(f"{case['id']}: contains key term '{term}'")

    assert problems == []


# ---------------------------------------------------------------- 3

def test_decay_levels_match_history():
    """Each agent's Decay level, recomputed from the frozen history, matches the stored
    label, and visible and summarized each have enough reuse cases to compare. Also: the
    Growth chat never mentions a case's agents (Growth measures the roster alone).

    Why: the summarizer decides what survives, so labels must follow what it actually did.
    """
    growth_history = (BUILT_DIR / "growth_history.txt").read_text(encoding="utf-8").lower()
    for agent in load_main_agents():
        if agent["role"] != "anchor":
            assert f"] {agent['name'].lower()}:" not in growth_history, agent["name"]
            for term in agent["key_terms"]:
                assert not contains_phrase(growth_history, term), (agent["name"], term)

    history = (BUILT_DIR / "decay_history.txt").read_text(encoding="utf-8")
    levels = load_decay_levels()
    for agent in load_main_agents():
        assert measure_level(agent, history) == levels[agent["name"]], agent["name"]

    # "gone" is allowed to be empty: the summarizer keeps pending, future-dated work (see README).
    counts = {"visible": 0, "summarized": 0, "gone": 0}
    for case in load_cases("decay"):
        if case["answer"] is not None and case["test"] != "elliptical_reference":
            counts[case["level"]] += 1
    for level in ["visible", "summarized"]:
        assert counts[level] >= 12, f"only {counts[level]} reuse cases at level {level}: {counts}"


# ---------------------------------------------------------------- 4

class _FakeRoster:
    def __init__(self, names):
        self._names = names

    def load(self):
        pass

    def get_agents(self):
        return list(self._names)


def _outside_active_agents(messages):
    content = messages[0]["content"]
    start = content.index("<active_agents>")
    end = content.index("</active_agents>")
    return content[:start] + content[end:]


def test_s0_prompt_matches_production(monkeypatch):
    """The prompt the harness builds for S0 is byte-for-byte what production builds, and
    S1/S3 differ from it only inside <active_agents>.

    Why: S0 must be today's code, or the baseline is wrong; and a strategy may only change
    the agent list, or other differences get credited to it.
    """
    case = load_cases("growth")[0]
    roster = roster_for(case, 10)
    history = history_for(case)

    monkeypatch.setattr(interaction_agent, "get_agent_roster", lambda: _FakeRoster([a["name"] for a in roster]))
    production = interaction_agent.prepare_message_with_history(case["message"], history, message_type="user")

    s0 = harness.build_messages(strategies.render_names(roster), history, case["message"])
    assert s0 == production

    s1 = harness.build_messages(strategies.render_cards(roster), history, case["message"])
    ranked = strategies.rank_agents(roster, case["message"])
    s3 = harness.build_messages(strategies.render_cards(strategies.shortlist(ranked)), history, case["message"])
    assert _outside_active_agents(s1) == _outside_active_agents(s0)
    assert _outside_active_agents(s3) == _outside_active_agents(s0)


# ---------------------------------------------------------------- 5

def test_grader_labels():
    """Canned dispatches get the right outcome label, using production's exact-name rule.

    Why: the grader decides every number in the report.
    """
    roster = {"Present Ideas", "Gift Receipts"}
    assert harness.grade(["Present Ideas"], "Present Ideas", roster) == "reuse_correct"
    assert harness.grade(["Gift Receipts"], "Present Ideas", roster) == "reuse_wrong"
    assert harness.grade(["Present Finder"], "Present Ideas", roster) == "created"
    assert harness.grade(["present ideas"], "Present Ideas", roster) == "created"
    assert harness.grade([], "Present Ideas", roster) == "no_dispatch"
    assert harness.grade([], "Present Ideas", roster, "what's your budget?") == "asked_user"
    assert harness.grade(["Present Ideas", "Present Finder"], "Present Ideas", roster) == "reuse_correct"
    assert harness.grade(["Present Ideas", "Gift Receipts"], "Present Ideas", roster) == "reuse_wrong"
    assert harness.grade(["Present Finder"], None, roster) == "created"
    assert harness.grade(["Present Ideas"], None, roster) == "reuse_wrong"


# ---------------------------------------------------------------- 6

def _tool_call_response(name, arguments):
    return {
        "choices": [{"message": {"content": "", "tool_calls": [
            {"id": f"call_{name}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


def _hash_folder(folder):
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(folder)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_run_has_no_side_effects():
    """With a fake model that announces first and then dispatches, the harness makes exactly
    2 calls, grades the dispatch, never runs a tool, and leaves server/data/ unchanged.

    Why: a benchmark run must never create agents, write logs or start an execution agent.
    """
    case = load_cases("growth")[0]
    calls = []

    async def fake_model(**request):
        calls.append(request)
        if len(calls) == 1:
            return _tool_call_response("send_message_to_user", {"message": "on it"})
        return _tool_call_response("send_message_to_agent", {"agent_name": case["answer"], "instructions": "go"})

    before = _hash_folder(REPO / "server" / "data")
    row = asyncio.run(harness.run_case(case, "S1", 10, call_model=fake_model))
    after = _hash_folder(REPO / "server" / "data")

    assert len(calls) == 2
    assert row["outcome"] == "reuse_correct"
    assert before == after
