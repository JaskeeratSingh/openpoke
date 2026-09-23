"""Builds everything in evals/data/built/ from the hand-written files in evals/data/written/.

    python -m evals.build_dataset                   # build everything (~10 summarizer calls)
    python -m evals.build_dataset --skip-summarizer # keep the existing Decay history

Steps:
  1. fillers.json          1,000 generated filler agents that compete with the real ones
  2. growth_history.txt    the fixed Growth chat (about the 3 anchors only), rendered like production
  3. decay_conversation.json + decay_history.txt
                           a ~1,090-entry conversation replayed through the production summarizer
  4. decay_levels.json     visible / summarized / gone for every hand-written agent
  5. s2_threshold.json     S2's similarity threshold, tuned on the dev cases
  6. manifest.json         what was built, with which seed and model
"""

import argparse
import asyncio
import random
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from server.config import get_settings
from server.openrouter_client import request_chat_completion
from server.services.conversation.log import ConversationLog, _default_formatter
from server.services.conversation.summarization import summarizer
from server.services.conversation.summarization.working_memory_log import WorkingMemoryLog

from . import strategies
from .dataset import (
    BUILT_DIR,
    DECAY_SIZE,
    DEV_SIZE,
    WRITTEN_DIR,
    load_cases,
    load_json,
    load_main_agents,
    measure_level,
    render_entries,
    roster_for,
    save_json,
)
from .harness import EVAL_MODEL, EVAL_PROVIDER, grade

SEED = 2026
NUMBER_OF_FILLERS = 1000

# How many conversation entries each part of the Decay conversation has.
# The production summarizer folds the oldest 100 entries into the summary each time
# 110 unsummarized entries pile up. With 1,090 entries in total, it runs 10 times and
# leaves exactly the last 90 entries as the verbatim tail. So "late" agents stay visible.
SEGMENT_SIZES = {"early": 500, "middle": 500, "late": 90}
SUMMARIZER_TIMEOUT = 300
SUMMARIZER_REASONING = {"effort": "low"}


# ---------------------------------------------------------------- 1. fillers

def filler_values(words, rng):
    """Random values for one filler's {placeholders}. IDs are made up, so they never match a real agent's."""
    invoice = f"INV-{rng.randint(1000, 9999)}"
    while invoice in ("INV-8291", "INV-7312"):
        invoice = f"INV-{rng.randint(1000, 9999)}"
    item = rng.choice(words["items"])
    return {
        "first": rng.choice(words["first_names"]),
        "last": rng.choice(words["last_names"]),
        "relation": rng.choice(words["relations"]),
        "topic": rng.choice(words["topics"]),
        "company": rng.choice(words["companies"]),
        "invoice": invoice,
        "amount": f"{rng.randint(80, 9000):,}",
        "city": rng.choice(words["cities"]),
        "airline": rng.choice(words["airlines"]),
        "date": f"{rng.choice(['Oct', 'Nov', 'Dec'])} {rng.randint(1, 28)}",
        "code": "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6)),
        "street": rng.choice(words["streets"]),
        "rent": f"{rng.randint(1500, 3200):,}",
        "time": rng.choice(["9am", "10:30am", "1pm", "3pm", "4:30pm"]),
        "course": rng.choice(words["courses"]),
        "venue": rng.choice(words["venues"]),
        "guests": rng.randint(4, 20),
        "clinic": rng.choice(words["clinics"]),
        "store": rng.choice(words["stores"]),
        "item": item,
        "item_title": item.title(),
        "order": f"#{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
        "pet": rng.choice(words["pets"]),
        "animal": rng.choice(words["animals"]),
        "bank": rng.choice(words["banks"]),
        "digits": str(rng.randint(1000, 9999)),
        "artist": rng.choice(words["artists"]),
        "row": rng.choice("ABCDEFGHJKL"),
        "policy": rng.choice(words["policies"]),
        "utility": rng.choice(words["utilities"]),
        "month": rng.choice(words["months"]),
    }


def make_fillers():
    """Generate filler agents that compete with the real ones.

    Each filler comes from one of the categories in filler_words.json (emails, invoices,
    travel, housing, ...), named the way the real agents are ("Email to Ingrid",
    "Invoice Chase Quarry Media"). But each is about different people, companies, places and
    IDs, so it never holds a fact a case needs. tests/test_eval.py checks that.
    """
    words = load_json(WRITTEN_DIR / "filler_words.json")
    rng = random.Random(SEED)
    used_names = {agent["name"] for agent in load_main_agents()}
    fillers = []
    attempts = 0
    while len(fillers) < NUMBER_OF_FILLERS:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("could not make enough distinct filler names; add words to filler_words.json")
        category = rng.choice(words["categories"])
        values = filler_values(words, rng)
        name = rng.choice(category["names"]).format(**values)
        if name in used_names:
            continue
        used_names.add(name)
        fillers.append({
            "name": name,
            "role": "filler",
            "category": category["category"],
            "key_terms": [term.format(**values) for term in category["key_terms"]],
            "episodes": [{"user": category["user"].format(**values), "result": category["result"].format(**values)}],
            "card": {field: text.format(**values) for field, text in category["card"].items()},
        })
    return fillers


# ---------------------------------------------------------------- building conversations

def episode_entries(agent, episode, rng):
    """One episode as 4 conversation entries: the user asks, Poke acknowledges,
    the agent reports back, Poke passes the result on."""
    acknowledgements = ["on it", "checking", "give me a sec", "sure, doing that now", "looking into it"]
    result = episode["result"]
    return [
        {"tag": "user_message", "text": episode["user"]},
        {"tag": "poke_reply", "text": rng.choice(acknowledgements)},
        {"tag": "agent_message", "text": f"[SUCCESS] {agent['name']}: {result}"},
        {"tag": "poke_reply", "text": result[0].lower() + result[1:]},
    ]


def build_segment(agent_blocks, target_size, small_talk, rng):
    """Mix small talk in between agent episodes until the segment has `target_size` entries.

    agent_blocks is a list of (episode_number, entries). All first episodes come before
    all second episodes, shuffled within each group, so an agent's episodes stay in order.
    Small talk is inserted at random positions, which never reorders the agent blocks.
    """
    by_number = {}
    for number, entries in agent_blocks:
        by_number.setdefault(number, []).append(entries)
    blocks = []
    for number in sorted(by_number):
        group = by_number[number]
        rng.shuffle(group)
        blocks.extend(group)

    size = sum(len(block) for block in blocks)
    while size < target_size:
        user_text, reply_text = rng.choice(small_talk)
        talk = [{"tag": "user_message", "text": user_text}, {"tag": "poke_reply", "text": reply_text}]
        blocks.insert(rng.randint(0, len(blocks)), talk)
        size += 2
    return blocks


def build_decay_conversation(fillers):
    """The full Decay conversation as a list of {"tag", "timestamp", "text"} entries.

    Each hand-written agent's episodes go in its "decay_slot" segment (early / middle / late).
    The fillers in the Decay roster each get their one episode in the early segment.
    """
    rng = random.Random(SEED)
    small_talk = load_json(WRITTEN_DIR / "small_talk.json")
    main_agents = load_main_agents()
    decay_fillers = fillers[: DECAY_SIZE - len(main_agents)]

    agent_blocks = {"early": [], "middle": [], "late": []}
    for agent in main_agents:
        for number, episode in enumerate(agent["episodes"]):
            agent_blocks[agent["decay_slot"]].append((number, episode_entries(agent, episode, rng)))
    for filler in decay_fillers:
        agent_blocks["early"].append((0, episode_entries(filler, filler["episodes"][0], rng)))

    blocks = []
    for segment in ["early", "middle", "late"]:
        blocks += build_segment(agent_blocks[segment], SEGMENT_SIZES[segment], small_talk, rng)
    return add_timestamps(blocks, datetime(2026, 6, 1, 9, 0, 0), rng)


def add_timestamps(blocks, start, rng):
    """Flatten blocks into entries with timestamps: blocks a few hours apart,
    entries inside a block a minute or so apart."""
    clock = start
    entries = []
    for block in blocks:
        clock += timedelta(minutes=rng.randint(60, 520))
        for entry in block:
            clock += timedelta(seconds=rng.randint(3, 90))
            entries.append({"tag": entry["tag"], "timestamp": clock.strftime("%Y-%m-%d %H:%M:%S"), "text": entry["text"]})
    return entries


async def replay_through_summarizer(entries):
    """Feed the conversation into the production summarizer one entry at a time, exactly as
    production would, and return the rendered history the IA would see at the end.

    The summarizer normally works on the app's real log files. Here we point it at
    temporary files instead (by replacing three lookups inside the summarizer module),
    and tell it to use EVAL_MODEL, so the real logs are never touched.
    """
    with tempfile.TemporaryDirectory() as folder:
        conversation_path = Path(folder) / "conversation.log"
        conversation = ConversationLog(conversation_path)
        memory = WorkingMemoryLog(Path(folder) / "working_memory.log")
        settings = get_settings().model_copy(update={"summarizer_model": EVAL_MODEL})

        # Left at its default, DeepSeek "thinks" for ~12k tokens before each summary,
        # which takes minutes. Low reasoning gives the same kind of summary much faster,
        # and EVAL_PROVIDER keeps it on a fast host (see harness.py).
        async def summarizer_call(**request):
            return await request_chat_completion(**request, reasoning=SUMMARIZER_REASONING, provider=EVAL_PROVIDER)

        original = (
            summarizer._resolve_conversation_log,
            summarizer.get_working_memory_log,
            summarizer.get_settings,
            summarizer.request_chat_completion,
        )
        summarizer._resolve_conversation_log = lambda: conversation
        summarizer.get_working_memory_log = lambda: memory
        summarizer.get_settings = lambda: settings
        summarizer.request_chat_completion = summarizer_call
        try:
            runs = 0
            for number, entry in enumerate(entries, start=1):
                with conversation_path.open("a", encoding="utf-8") as handle:
                    handle.write(_default_formatter(entry["tag"], entry["timestamp"], entry["text"]))
                memory.append_entry(entry["tag"], entry["text"], entry["timestamp"])
                try:
                    # A call can hang for many minutes (the API keeps the connection alive),
                    # so give up after SUMMARIZER_TIMEOUT seconds and retry on the next entry.
                    if await asyncio.wait_for(summarizer.summarize_conversation(), SUMMARIZER_TIMEOUT):
                        runs += 1
                        print(f"  summarizer run {runs} done (after entry {number})", flush=True)
                except (Exception, asyncio.TimeoutError) as error:
                    # Production's background worker also swallows the error and tries again
                    # after the next message, so we do the same.
                    print(f"  summarizer failed after entry {number}, will retry: {error}", flush=True)
            return memory.render_transcript()
        finally:
            (
                summarizer._resolve_conversation_log,
                summarizer.get_working_memory_log,
                summarizer.get_settings,
                summarizer.request_chat_completion,
            ) = original


# ---------------------------------------------------------------- 5. S2 threshold

def tune_s2_threshold():
    """Pick the similarity threshold S2 uses, using only the 20 dev cases.

    For each threshold from 0.00 to 1.00, count how many dev cases S2 would get right,
    and keep the middle of the best range.
    """
    cases = load_cases("dev")
    rankings = []
    for case in cases:
        roster = roster_for(case, DEV_SIZE)
        ranked = strategies.rank_agents(roster, case["message"])
        rankings.append((case, ranked, {agent["name"] for agent in roster}))

    scores = {}
    for step in range(101):
        threshold = step / 100
        correct = 0
        for case, ranked, roster_names in rankings:
            picked = strategies.pick_nearest(ranked, threshold)
            outcome = grade([picked] if picked else ["(new agent)"], case["answer"], roster_names)
            wanted = "reuse_correct" if case["answer"] else "created"
            correct += outcome == wanted
        scores[threshold] = correct

    best = max(scores.values())
    best_thresholds = [threshold for threshold, score in scores.items() if score == best]
    chosen = best_thresholds[len(best_thresholds) // 2]
    return {"threshold": chosen, "dev_correct": best, "dev_cases": len(cases)}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Build the routing benchmark dataset.")
    parser.add_argument("--skip-summarizer", action="store_true", help="keep the existing Decay history")
    args = parser.parse_args()

    print("1. fillers")
    fillers = make_fillers()
    save_json(BUILT_DIR / "fillers.json", fillers)

    print("2. growth history")
    growth_history = render_entries(load_json(WRITTEN_DIR / "growth_history.json"))
    (BUILT_DIR / "growth_history.txt").write_text(growth_history, encoding="utf-8")

    print("3. decay conversation")
    conversation = build_decay_conversation(fillers)
    save_json(BUILT_DIR / "decay_conversation.json", conversation)
    history_path = BUILT_DIR / "decay_history.txt"
    if args.skip_summarizer and history_path.exists():
        print("  keeping the existing decay_history.txt")
    else:
        print(f"  replaying {len(conversation)} entries through the summarizer ({EVAL_MODEL})")
        history_path.write_text(asyncio.run(replay_through_summarizer(conversation)), encoding="utf-8")

    print("4. decay levels")
    decay_history = history_path.read_text(encoding="utf-8")
    levels = {agent["name"]: measure_level(agent, decay_history) for agent in load_main_agents()}
    save_json(BUILT_DIR / "decay_levels.json", levels)
    target_levels = Counter(levels[agent["name"]] for agent in load_main_agents() if agent["role"] == "target")
    print(f"  target agents per level: {dict(target_levels)}")

    print("5. S2 threshold")
    threshold = tune_s2_threshold()
    save_json(BUILT_DIR / "s2_threshold.json", threshold)
    print(f"  {threshold}")

    save_json(BUILT_DIR / "manifest.json", {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "summarizer_model": EVAL_MODEL,
        "embedding_model": strategies.EMBEDDING_MODEL,
        "main_agents": len(load_main_agents()),
        "fillers": len(fillers),
        "decay_conversation_entries": len(conversation),
        "decay_target_levels": dict(target_levels),
        "growth_cases": len(load_cases("growth")),
        "decay_cases": len(load_cases("decay")),
        "dev_cases": len(load_cases("dev")),
        "s2_threshold": threshold["threshold"],
    })
    print("done")


if __name__ == "__main__":
    main()
