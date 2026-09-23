"""Runs one case through one strategy and grades the answer.

For S0, S1 and S3 it asks the real interaction agent (IA): the production system prompt,
the production tool list and the production prompt layout, with only the agent list
swapped. Tool calls are never executed. We fake their results in the same shape
production returns, and stop at the first send_message_to_agent, because that call
already contains the decision we are grading. The execution agent never runs.

Used by bench.py (every test) and by build_dataset.py (S2's threshold).
"""

import asyncio
import json
import random

from server.agents.interaction_agent.agent import (
    _render_conversation_history,
    _render_current_turn,
    build_system_prompt,
)
from server.agents.interaction_agent.tools import TOOL_SCHEMAS
from server.openrouter_client import request_chat_completion

from . import strategies
from .dataset import BUILT_DIR, history_for, load_json, roster_for

EVAL_MODEL = "deepseek/deepseek-v4.1-flash"

# OpenRouter serves this model from ~25 hosts. By default it picks cheap ones, which can be
# slow (~60 tokens/s) and some run a compressed fp4 version of the model. Pinning CoreWeave
# (fast, fp8) keeps every call on the same version of the model. Fireworks is the backup.
EVAL_PROVIDER = {"order": ["coreweave", "fireworks"], "allow_fallbacks": True}
INPUT_PRICE_PER_MILLION = 0.20   # CoreWeave's price
OUTPUT_PRICE_PER_MILLION = 0.65

MAX_ROUNDS = 8  # same limit as InteractionAgentRuntime.MAX_TOOL_ITERATIONS
API_ATTEMPTS = 4


def build_messages(agents_block, history_text, message):
    """The IA's user message, laid out exactly like production's prepare_message_with_history(),
    but with our own agent list in <active_agents>."""
    sections = [
        _render_conversation_history(history_text),
        f"<active_agents>\n{agents_block}\n</active_agents>",
        _render_current_turn(message, "user"),
    ]
    return [{"role": "user", "content": "\n\n".join(sections)}]


def fake_tool_result(name, arguments):
    """What production would send back after a tool call, in the same JSON shape as
    InteractionAgentRuntime._format_tool_result(). Nothing is actually done."""
    if name == "send_message_to_user":
        result = {"status": "delivered"}
    elif name == "send_draft":
        result = {"status": "draft_recorded", "to": arguments.get("to"), "subject": arguments.get("subject")}
    elif name == "wait":
        result = {"status": "waiting", "reason": arguments.get("reason")}
    else:
        return json.dumps({"tool": name, "status": "error", "arguments": arguments, "error": f"Unknown tool: {name}"})
    return json.dumps({"tool": name, "status": "success", "arguments": arguments, "result": result})


def grade(dispatched, answer, roster_names, reply_text=""):
    """Turn the agent names the IA dispatched to into one outcome label.

    reuse_correct  dispatched to the right existing agent (extra brand-new agents are allowed)
    reuse_wrong    dispatched to an existing agent that isn't the answer
    created        every dispatched name is new to the roster
    asked_user     didn't dispatch, and asked the user a question instead (left out of the rates:
                   asking before acting is often right, e.g. "what's your GST number?")
    no_dispatch    didn't dispatch and didn't ask anything

    Names match exactly, like production (tools.py: `agent_name not in existing_agents`),
    so "present ideas" is a new agent, not "Present Ideas".
    """
    if not dispatched:
        return "asked_user" if "?" in reply_text else "no_dispatch"
    existing = [name for name in dispatched if name in roster_names]
    wrong = [name for name in existing if name != answer]
    if wrong:
        return "reuse_wrong"
    if answer is not None and answer in existing:
        return "reuse_correct"
    return "created"


async def call_with_retries(call_model, **request):
    """Call the model, retrying rate limits and network errors with a growing wait.
    Returns (response, error_text); response is None if every attempt failed."""
    error_text = None
    for attempt in range(API_ATTEMPTS):
        try:
            return await call_model(**request), None
        except Exception as error:
            error_text = str(error)[:300]
            await asyncio.sleep(5 * 2 ** attempt + random.random())
    return None, error_text


async def ask_interaction_agent(agents_block, history_text, message, call_model):
    """Run the IA's tool loop until it dispatches to an agent, stops calling tools, or hits MAX_ROUNDS.

    Returns a dict with the dispatched agent names, token counts, number of calls and any API error.
    """
    messages = build_messages(agents_block, history_text, message)
    result = {"dispatched": [], "model_calls": 0, "tokens_in": 0, "tokens_out": 0, "providers": [],
              "reply_text": "", "error": None}

    for _ in range(MAX_ROUNDS):
        response, error = await call_with_retries(
            call_model,
            model=EVAL_MODEL,
            messages=messages,
            system=build_system_prompt(),
            tools=TOOL_SCHEMAS,
            provider=EVAL_PROVIDER,
        )
        if response is None:
            result["error"] = error
            return result

        result["model_calls"] += 1
        result["providers"].append(response.get("provider"))
        usage = response.get("usage") or {}
        result["tokens_in"] += usage.get("prompt_tokens", 0)
        result["tokens_out"] += usage.get("completion_tokens", 0)

        choice = (response.get("choices") or [{}])[0]
        reply = choice.get("message") or {}
        tool_calls = reply.get("tool_calls") or []

        assistant_entry = {"role": "assistant", "content": reply.get("content") or ""}
        if tool_calls:
            assistant_entry["tool_calls"] = tool_calls
        messages.append(assistant_entry)

        if not tool_calls:
            result["reply_text"] = result["reply_text"] or assistant_entry["content"]
            return result

        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if name == "send_message_to_agent":
                result["dispatched"].append(str(arguments.get("agent_name", "")))
            if name == "send_message_to_user":
                # Kept so a no_dispatch case can be read later: did it ask a question? answer directly?
                result["reply_text"] = str(arguments.get("message", ""))
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or name,
                "content": fake_tool_result(name, arguments),
            })

        if result["dispatched"]:
            return result

    return result


def s2_threshold():
    """The similarity S2 needs before it reuses an agent (tuned on the dev cases by build_dataset.py)."""
    return load_json(BUILT_DIR / "s2_threshold.json")["threshold"]


async def run_case(case, strategy, size=None, call_model=request_chat_completion, limiter=None, shuffle_seed=None):
    """Run one case through one strategy and return one result row (a dict).

    size:          roster size for Growth cases (ignored for Decay)
    call_model:    the function that calls the LLM; tests pass a fake one
    limiter:       an asyncio.Semaphore that caps how many cases call the API at once
    shuffle_seed:  if set, shuffle the agent list first (used by the order check)
    """
    roster = list(roster_for(case, size))
    if shuffle_seed is not None and strategy in ("S0", "S1"):
        random.Random(shuffle_seed).shuffle(roster)
    history = history_for(case)
    roster_names = {agent["name"] for agent in roster}

    row = {
        "experiment": case["experiment"],
        "case_id": case["id"],
        "test": case["test"],
        "strategy": strategy,
        "size": len(roster),
        "level": case["level"],
        "answer": case["answer"],
        "shuffle_seed": shuffle_seed,
        "dispatched": [],
        "gold_rank": None,
        "model_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "providers": [],
        "reply_text": "",
        "cost": 0.0,
        "error": None,
    }

    ranked = None
    if strategy in ("S2", "S3"):
        ranked = strategies.rank_agents(roster, case["message"])
        row["gold_rank"] = strategies.gold_rank(ranked, case["answer"])

    if strategy == "S2":
        picked = strategies.pick_nearest(ranked, s2_threshold())
        row["dispatched"] = [picked] if picked else ["(new agent)"]
        row["outcome"] = grade(row["dispatched"], case["answer"], roster_names)
        return row

    if strategy == "S0":
        agents_block = strategies.render_names(roster)
    elif strategy == "S1":
        agents_block = strategies.render_cards(roster)
    else:
        picked_agents = strategies.shortlist(ranked)
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(picked_agents)
        agents_block = strategies.render_cards(picked_agents)

    if limiter is None:
        limiter = asyncio.Semaphore(1)
    async with limiter:
        answer = await ask_interaction_agent(agents_block, history, case["message"], call_model)

    row.update(answer)
    row["cost"] = (row["tokens_in"] * INPUT_PRICE_PER_MILLION + row["tokens_out"] * OUTPUT_PRICE_PER_MILLION) / 1_000_000
    if row["error"] and not row["dispatched"]:
        row["outcome"] = "error"
    else:
        row["outcome"] = grade(row["dispatched"], case["answer"], roster_names, row["reply_text"])
    return row
