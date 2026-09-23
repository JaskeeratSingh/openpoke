# Agent routing in OpenPoke: the problem, the fix, and how it was measured

## Contents

1. [What OpenPoke is](#1-what-openpoke-is)
2. [Small bugs fixed along the way](#2-small-bugs-fixed-along-the-way)
3. [The actual problem: routing](#3-the-actual-problem-routing)
4. [Four designs](#4-four-designs)
5. [How the designs were tested](#5-how-the-designs-were-tested)
6. [How the test data was built](#6-how-the-test-data-was-built)
7. [How one test case runs and gets graded](#7-how-one-test-case-runs-and-gets-graded)
8. [Results](#8-results)
9. [Checks that keep the dataset honest](#9-checks-that-keep-the-dataset-honest)
10. [Running it yourself](#10-running-it-yourself)
11. [Caveats and open questions](#11-caveats-and-open-questions)
12. [File map](#12-file-map)

---

## 1. What OpenPoke is

OpenPoke is an open-source assistant with two kinds of agent:

- **The interaction agent (IA)** — the one you text. It never does the work itself; it reads
  your message and decides *who* should handle it.
- **Execution agents (EAs)** — the ones that actually do the work (send an email, chase an
  invoice, book a flight). Each EA has a name and its own memory of everything it has done.

An important mental-model correction: **an execution agent is not a running program.** There
is no process called "Invoice Chase." It is a name in `roster.json` plus a log file on disk
(`server/agents/interaction_agent/tools.py:112-150`). "Reusing an agent" means "read the same
log file again." "Creating a new agent" means a brand-new, empty log file appears. This is
why *which* agent gets picked matters so much: pick wrong, and a real memory either goes
unread or gets contaminated.

When a message arrives, the IA calls a tool, `send_message_to_agent(agent_name, instructions)`.
If `agent_name` matches an existing EA **exactly**, that EA keeps its memory. If it doesn't
match, a brand-new EA starts from nothing. That one string-equality check is the entire
routing decision this document is about.

---

## 2. Small bugs fixed along the way

While getting familiar with the codebase (before touching the routing problem), four small,
unrelated bugs surfaced and were fixed:

1. **Outdated Composio call** (`server/services/gmail/client.py`) — Composio retired
   `initiate()`; replaced with `client.connected_accounts.link(...)` per their current docs.
2. **Gmail importance watcher fetched a 10-*month* window and got HTTP 413 on every poll**
   (`server/services/gmail/importance_watcher.py`) — the query used `newer_than:{n}m`, where
   `m` means *months*, not minutes, blowing past Gmail's payload limit. Fixed by bounding the
   window with an explicit epoch (`after:{epoch}`) instead.
3. **The same watcher silently dropped emails delivered ~9 seconds outside its notification
   window** (`server/services/gmail/importance_watcher.py`) — it had two windows (a 10-minute
   fetch window and a 60-second notify window), and Gmail's own indexing lag meant a message
   could be fetched one poll late, land just outside the notify cutoff, and get marked "seen"
   and discarded without ever reaching the classifier. Fixed by anchoring the cutoff to the
   *previous poll* instead of wall-clock time, with a 30-second grace period
   (`_compute_cutoff()`). Regression tests: `tests/test_importance_watcher_window.py`.
4. **The email-search prompt didn't say Gmail's `before:` filter is exclusive**
   (`server/agents/execution_agent/tasks/search_email/system_prompt.py`) — the model searched
   `after:2026/09/13 before:2026/09/20` expecting an inclusive range and missed emails from the
   20th. Fixed by documenting the asymmetry in the prompt and telling the model to omit
   `before:` when the range should run up to the present.

None of these are related to routing; they're recorded here because they were found and fixed
in this repo, and because the routing benchmark deliberately does not depend on them.

---

## 3. The actual problem: routing

### What the IA sees today

The IA's prompt has four parts (`server/agents/interaction_agent/agent.py:20`,
`prepare_message_with_history`):

1. `<conversation_history>` — the chat so far (old parts get summarized; see below).
2. `<active_agents>` — **only the names** of existing EAs, one per line
   (`agent.py:45`, `_render_active_agents`). No description, no history, nothing else.
3. `<new_user_message>`.

So the IA must pick an agent from its bare **name** plus whatever the chat history still says
about it. Production's own instructions tell it to reuse an agent even when its name looks
unrelated, as long as it holds useful context
(`server/agents/interaction_agent/system_prompt.md:19`):

> "Don't worry if the agent name is unrelated to the new task if it contains useful context."

But the IA has no way to check that — it never reads any EA's log. Its only source of
"useful context" is the conversation history, and that fades.

### Why this gets harder over time

- **Growth.** Agents are never deleted — `roster.add_agent()` only appends. After months
  there could be hundreds or thousands of names, all still in the prompt.
- **Decay.** The IA learns what an agent knows only from the conversation history, and old
  history gets **summarized** once it passes a threshold (production: 100 messages, plus a
  10-message tail — `server/config.py:71-72`,
  `server/services/conversation/summarization/summarizer.py:73`). The agent's *name* survives
  in the roster forever; what it *did* fades out of the prompt as soon as it's summarized away.

These interact badly: summarization compresses away the very messages that gave a name its
meaning, while the roster keeps every name forever. Over time you get a growing list of names
whose context has quietly evaporated — and the system prompt still tells the model to route
by context it can no longer see.

### An existing attempt: PR #11, "Smart Roster"

Before designing anything new, the repo's open draft
[PR #11](https://github.com/shlokkhemani/openpoke/pull/11) was read in full — 12 hill-climbing
experiments (make one change, measure, keep it if a number improved). Three of its ideas are
directly relevant to routing:

- **LRU-20 eviction** — keep only the 20 most recently used agents in the roster, delete the
  rest. Problem: hiding a name is the same failure mode the PR's own earlier experiments hit
  and rejected (hidden agents get duplicated), just deferred to the 21st agent instead of the
  1st. Nothing tests for it.
- **Relevance-by-name scoring** — score each agent's *name* by keyword overlap with the
  request. Problem: this directly contradicts the system prompt's own rule above — it assumes
  names are a reliable relevance signal, when production explicitly says they aren't. What's
  worth keeping from this idea isn't the name-scoring, it's the underlying shape: **a roster
  that changes with the request**, rather than a static list.
- **Capping each EA's own log** — parked; a separate concern from routing.

Digging into the PR also surfaced that 84% of its *measured* token savings came from three
plain integer caps (history char budget, per-agent conversation limit, batch response cap) —
not from the "smart roster" machinery it's named after, and none of its roster changes were
tested for whether they still pick the *right* agent, only for token count. That gap —
measuring cost but never correctness — is what this benchmark exists to fill.

---

## 4. Four designs

Two ideas, and two ways to combine them:

**Idea 1 — give the roster more context.** Instead of `<agent name="Present Ideas" />`, show
the IA a short **card** per agent: purpose / people / details / state. It survives
summarization because it's stored with the agent, not derived from the chat.

```
Present Ideas
purpose: Choosing a birthday present for the user's girlfriend, Maya.
people:  Maya, the user's girlfriend. Birthday Oct 14. Loves ceramics and hiking. Vegetarian, so no leather.
details: Budget about $100. Options: Claymates 4-week wheel-throwing voucher ($85) or Osprey trail daypack ($95).
state:   Waiting for the user to choose.
```

Catch: the prompt now grows with every agent's *card*, not just its name — at 1,000 agents
that's roughly 3x the names-only prompt.

**Idea 2 — subcontract the choice.** Hand the decision to something outside the IA's own
prompt entirely. This could be as complex as a dedicated router agent (its own LLM call, whose
only job is picking an agent), or as simple as an embedding similarity search with no model
call at all. This benchmark tests the simple end of that spectrum: embed the request, embed
every agent's name + card, reuse the most similar one if it clears a similarity threshold,
otherwise create a new agent. Catch: similarity isn't usefulness (a shared first name looks
similar but may be the wrong person), and it can't see the conversation, so it can't resolve
"tell him thursday works."

**The four strategies tested** (`evals/strategies.py`) — every one uses the exact same
production prompt, tools, and system prompt; only the `<active_agents>` block changes:

| | What the IA sees | Who decides | Model call? |
|---|---|---|---|
| **S0** | every agent's **name** | the IA | yes — *today's production behaviour* |
| **S1** | every agent's **name + card** | the IA | yes |
| **S2** | nothing (no roster in the prompt at all) | embedding search: nearest agent, or "new" if the best match is below a threshold | no |
| **S3** | the **5** agents whose cards best match the request | the IA | yes |

S0 is the baseline (what OpenPoke does right now). S1 is Idea 1 in full. S2 is Idea 2 in its
simplest form. S3 combines both — search narrows the field, the IA still makes the final call
with the full chat history in view (the same shape as retrieval-augmented tool calling: don't
put every tool in the prompt, retrieve the few relevant ones and let the model choose).

A check (`tests/test_eval.py::test_s0_prompt_matches_production`) verifies S0's prompt is
byte-for-byte identical to what production actually sends, and that S1/S3 differ from it only
inside `<active_agents>` — so no strategy gets credit or blame for anything except its agent
list.

---

## 5. How the designs were tested

Two separate experiments, each changing exactly one thing, so a change in the results has one
possible cause:

### Growth — does a bigger roster hurt?

- **Roster size:** 10, 50, 100, 250, 500, 1,000 agents (`GROWTH_SIZES`, `dataset.py:21`).
- **History: held fixed, and deliberately uninformative.** A short, 20-entry chat about three
  unrelated "anchor" agents (gym, groceries, dentist) that **never mentions** the case's real
  target agent, at any size. This means the IA has *only* the agent list to go on — which is
  exactly the point: it isolates the one thing Growth is testing, roster size, from the other
  variable (how much the history still says). An earlier version of this design let history
  hint at the target, and S0 scored ~90% — an artifact, since a chat that talks about the
  answer flatters name-only routing. That was fixed specifically because it was *unrealistic*,
  not because it was unflattering to any one strategy.
- **Same 120 requests at every size.** A bigger roster is always the smaller one plus more
  filler agents — never a different random draw — so any change across sizes is attributable
  to the extra agents alone (`roster_for()`, `dataset.py:99`).

*Anticipated pushback: doesn't fixing the history and only growing the roster bias the result
against S0, since S0 depends most on history?* No — every strategy sees the identical roster
and identical history for a given case; only the `<active_agents>` block differs (checked
byte-for-byte, above). Growth fixes history at its hardest, most realistic point (nothing to
go on — the case for any agent that's a few months old and never came up again) and varies
only size; Decay does the reverse. All four strategies face that same information-starved
history; S1, S2 and S3 just have something else (a card, or a search) to fall back on, and S0
doesn't. That asymmetry in outcomes is the finding, not a symptom of an unfair setup.

### Decay — does routing survive forgetting?

- **Roster size: held fixed at 100** (a realistic "few months of use" size; Growth already
  covers size as a variable).
- **History: a real ~1,090-entry conversation, replayed through OpenPoke's actual production
  summarizer** (not reimplemented — the real `summarize_conversation()` function, pointed at
  temporary files so nothing in `server/data/` is touched), then frozen. Each of the 51 named
  agents' past work sits early, middle, or late in that conversation
  (its `decay_slot`), so by the end each one has been summarized a different amount.
- Every reuse case is labelled by what the frozen history still says about its target agent:

  | Level | What survives |
  |---|---|
  | **visible** | its result message, word for word (never summarized yet) |
  | **summarized** | only a mention in the summary text, no result line |
  | **gone** | nothing — the name is in the roster and that's it |

  This build measured 8 targets visible, 16 summarized, 0 gone (production's summarizer keeps
  almost anything still pending, and every target's story is still open), so Decay effectively
  compares **visible vs. summarized**.

---

## 6. How the test data was built

No real user traffic exists, so the dataset is synthetic — but built to be checked, not
guessed at. Two kinds of files:

- **`data/written/`** — authored directly (as static content, not generated by a script);
  the source of truth. Edit these, then rebuild.
- **`data/built/`** — generated by a script from the written files. Never edit these directly;
  just rerun the build.

### Written directly, in order

1. **`agents.json` — 51 agents.** 24 **targets** (what test cases are about — each with 1-3
   past jobs full of specific facts, plus a card summarizing them, plus `key_terms` used to
   measure decay). 24 **false friends**, one per target, sharing a first name or keyword but
   holding nothing useful (e.g. "Invoice Chase" tracks Alex *Chen*'s payment; its false friend
   "Tutoring with Alex" is a completely different Alex). 3 **anchors** (gym/groceries/dentist)
   that the Growth chat talks about.
2. **`cases.json` — 142 test-case records, all in one file.** Each case is tagged with which
   of the 7 tests it belongs to (`"test"`) and which experiments it runs in
   (`"experiments": ["growth", "decay"]`, or just one). Example:
   ```json
   {"id": "easy_overlap-01", "test": "easy_overlap", "experiments": ["growth", "decay"],
    "message": "check with alex chen whether the INV-8291 wire actually went out",
    "answer": "Invoice Chase", "roster_agents": ["Invoice Chase", "Tutoring with Alex"],
    "must_mention": ["alex"]}
   ```
   - **`answer`** — the right agent's name, or `null` for a "should create new" case.
   - **`roster_agents`** — which of the 51 named agents must be in the roster *at every Growth
     size*, not just by chance among the fillers. For a reuse case this is the answer plus its
     false friend, so the two always compete, even at 10 agents. For a create case, it's the
     look-alikes that should tempt the model into a wrong reuse. `roster_for()` keeps these
     unconditionally regardless of requested size — this is what actually guarantees the
     answer is reachable in the roster, not luck.
   - **`must_mention`** — words the request text is required to contain, checked by
     `tests/test_eval.py`. This is a pure data-integrity guard, not read by the harness: it
     makes sure a case's premise (e.g. "shares a word with the agent") is actually true of the
     written text, not just true in intent.
3. **`dev_cases.json` — 20 held-out cases**, used *only* to tune S2's similarity threshold,
   never scored.
4. **`growth_history.json`, `small_talk.json`, `filler_words.json`** — the fixed Growth chat,
   chit-chat padding for the Decay conversation, and templates + word lists for filler agents.

### Built by `python -m evals.build_dataset` (one-time; produces `data/built/`)

1. **1,000 filler agents** (`fillers.json`) — generated from 14 realistic categories (email,
   invoice, travel, housing, …), named the way real agents are, but built from a disjoint word
   list so none can ever hold a fact a real case needs.
2. **`growth_history.txt`** — the 20 entries written directly in `growth_history.json`, rendered through production's own
   `WorkingMemoryLog` class, so the formatting is byte-identical to what the real app produces.
   This is the only history any Growth case ever sees, at any size.
3. **`decay_history.txt`** — ~1,090 entries assembled from every agent's episodes (placed
   early/middle/late by `decay_slot`, padded with small talk), fed **one entry at a time**
   through the real production summarizer, then frozen. This is the only step in the whole
   build that costs API calls (~10 summarizer calls, a few minutes, a few cents).
4. **`decay_levels.json`** — each agent's visible/summarized/gone label, measured directly
   from the frozen history above.
5. **`s2_threshold.json`** — every similarity cutoff from 0.00 to 1.00 tried against the 20 dev
   cases; the middle of the best-scoring range is kept (this build: **0.40**, 18/20 correct).

### Turning a case into a runnable scenario, at run time

`cases.json` only has a request and an answer — no roster, no history. Both get assembled live
by `dataset.py`, every time a case runs:

```
cases.json entry ──► load_cases("growth")     copies it per experiment, assigns an id
       │
       ├──► roster_for(case, size)   Growth: 3 anchors + roster_agents + fillers up to size
       │                             (nested: size 50's roster ⊃ size 10's roster)
       │                             Decay: always the fixed 100 (51 main + 49 fillers)
       │
       ├──► history_for(case)        Growth: growth_history.txt, identical for every case/size
       │                             Decay: decay_history.txt (+ extra lines for elliptical cases)
       │
       └──► harness.run_case(case, strategy, size)
```

### Why some cases only run in Growth, or only in Decay

- **`elliptical_reference` is Decay-only, structurally.** Its premise ("tell him thursday
  works") requires the history to actually contain a recent clue about the target — which
  directly violates Growth's core invariant that history must *never* mention a case's target.
  A Growth version of this test would confound the two things Growth and Decay each try to
  isolate.
- **A few reuse-test cases exist only in Decay** (e.g. `easy_overlap-21..24`) — added because
  the checks require at least 12 reuse cases land at "visible" and 12 at "summarized" (pooled),
  and the cases shared with Growth weren't enough on their own. Growth has no concept of
  "level" (it's always "gone" by design), so these extra agents add nothing there.
- **A few create-test cases (`false_friend`, `unrelated`) exist only in Growth** — more
  decoy scenarios matter more as the roster scales to 1,000; Decay's roster is always fixed at
  100 regardless, so it needs fewer.

---

## 7. How one test case runs and gets graded

```
roster + history + request
      │  the strategy builds <active_agents>; everything else is production's real prompt
      ▼
call DeepSeek V4.1 Flash with the production system prompt and tool list
      │  the model may call tools; every result is faked in production's exact JSON shape —
      │  nothing is actually sent, created, or executed
      ▼
stop at the first send_message_to_agent  ← this call is the decision being measured
      │
      ▼
grade the agent_name it chose against the roster and the answer
```

**Grading** (`harness.grade()`), by exact name match (like production — a misspelling counts
as a new agent):

| Label | Meaning | Counted in the rates? |
|---|---|---|
| `reuse_correct` | the right existing agent | yes |
| `reuse_wrong` | a *different* existing agent — the worst mistake, since it writes into someone else's memory | yes |
| `created` | every name dispatched was new | yes |
| `asked_user` | didn't route; asked the user something first | **no** — asking first is often correct and says nothing about which agent it would pick |
| `no_dispatch` / `error` | didn't route and didn't ask / the API failed after 4 retries | no |

**What S2's embedding search actually searches over:** for a given case, the request message
(only the new message, never the chat history — an earlier version also embedded recent
history lines, but those are usually small talk or another agent's news and pulled the search
off course) is embedded and compared against every agent **currently in that case's roster**
(name + card text), using `all-MiniLM-L6-v2` via `fastembed` — free, local, deterministic.
That's a different, freshly-ranked set for every case and every roster size; it is never a
search over some fixed global index.

---

## 8. Results

Full run: `results/20260923-063434.jsonl` (3,320 rows, $6.48). Order check:
`results/20260923-063559.jsonl` ($0.39).

### Headline numbers

**Growth — reuse accuracy** (the right agent, when one should be reused):

| | 10 agents | 100 | 1,000 | Wrong reuse @1,000 | Prompt tokens @1,000 | $ / 1,000 requests @1,000 |
|---|---|---|---|---|---|---|
| S0 · names | 50% | 43% | 33% | 4% | 14.5k | $3.21 |
| S1 · cards | 99% | 96% | 95% | 0% | 55k | $11.68 |
| S2 · search only | 62% | 61% | 54% | 19% | – | free |
| S3 · shortlist + IA | 95% | 90% | 81% | 2% | 3.9k | $1.02 |

**Decay** (100 agents), reuse accuracy, visible → summarized: S0 97%→82%, S1 97%→96%,
S3 89%→89%. (S2's "levels" here aren't directly comparable — it never reads history, so
visible vs. summarized for S2 just reflects which requests happen to land in each bucket.)

### Which kind of request breaks first

Per-test accuracy (pooled across sizes) shows `misleading_name` and `related_context` are
S0's floor:

- `misleading_name`: S0 **9%** (10/114) — the name is set by the agent's *first* job and never
  updates, so a request about a later, unrelated job it did gives S0 nothing to match on.
- `related_context`: the weakest test for **every** strategy (S0 33%, and even S1/S3 dip) —
  the IA tends to invent a new agent named after the task in front of it, instead of reusing
  the one that already holds the needed fact. This is an instruction-following gap in the IA's
  own prompt, not something a better agent list alone fixes.
- `false_friend` / `unrelated` (should create new): near 100% for S0/S1/S3. The exception is
  S2, which wrong-reuses 35 of 120 `false_friend` cases as the roster grows — a shared first
  name looks similar even when it isn't the same person.

### Accuracy against cost

S0 is worst on *both* axes at scale: 3.7k→14.5k tokens per call, but only 55-64% accurate.
S3 clusters tight around ~3.9k tokens across 87-95% accuracy. S1 spans 4k→55k tokens at
94-99% accuracy — most accurate, priciest. There's no accuracy reason left to keep S0 as-is.

### Other findings

- **S0 gets more cautious, not more reckless, as the roster grows.** Wrong-reuse actually
  *falls* from 8% to 4% (10→1,000 agents), but unnecessary-new *rises* from 41% to 62% — with
  more names it can't confidently read, it increasingly plays it safe and creates a new agent
  instead of guessing.
- **Order sensitivity.** Shuffling the same 100-agent roster three different ways flips S0's
  outcome on 10 of 30 sampled cases (S1: 4, S3: 3) — one request came back `reuse_correct`,
  `reuse_wrong`, and `created` across the three shuffles, with nothing but list order
  different. A decision that depends on list order isn't really based on content.
- **The summary keeps facts but drops names.** Only 5 of 16 summarized targets kept their
  agent's *name* next to their facts in the summary — exactly the link S0 depends on and S1's
  cards don't need.
- **The names it invents give away why it fails**: "Email to Prof. Ruiz" instead of reusing
  Lab Application, "Tokyo Trip" instead of Flight Booking — S0 names after the task in front of
  it, not the thread that already holds the context.

### Bottom line

**S1** is the most accurate strategy and never wrong-reuses, at the cost of a prompt that
grows without bound. **S3** gets close to S1's accuracy at a fraction of the tokens — cheaper
than today's S0 even at 1,000 agents — and its remaining misses are mostly search misses (at
1,000 agents, 12 of S3's 15 misses had the right agent outside the top-5 shortlist), which a
better search would directly address. **S2** alone is too willing to reuse the wrong agent as
the roster grows. **S0**, today's production behavior, is worst on accuracy, worst on the
Decay drop, and its answer depends on list order — that is the actual risk in production
today, not the size of the prompt.

---

## 9. Checks that keep the dataset honest

`python -m pytest tests/test_eval.py` — six checks, no network, no API key:

| Check | Verifies |
|---|---|
| `test_answers_are_unambiguous` | the answer is in the roster (reuse cases); no other agent in the roster shares a fact with the request; create cases: no agent covers the topic; fillers never reuse a real agent's first name |
| `test_requests_follow_word_rules` | `must_mention` words are present; `semantic_disconnect`/`misleading_name` requests share no word with the agent's name; `elliptical_reference` requests name no agent |
| `test_decay_levels_match_history` | the Growth chat never mentions a case's target; Decay levels recomputed from the frozen history match the stored labels; visible and summarized each have ≥12 reuse cases |
| `test_s0_prompt_matches_production` | S0's prompt is byte-for-byte production's; S1/S3 differ only inside `<active_agents>` |
| `test_grader_labels` | canned dispatches get the right outcome label |
| `test_run_has_no_side_effects` | a run makes exactly the expected model calls, executes no tool, and leaves `server/data/` untouched |

---

## 10. Running it yourself

```bash
pip install -r evals/requirements.txt          # once — installs fastembed

python -m evals.build_dataset                  # only needed after editing data/written/
python -m pytest tests/test_eval.py            # the six checks above, free

python -m evals.run_benchmark --tests easy_overlap --limit 3 --sizes 10   # smoke run, ~1 cent
python -m evals.run_benchmark                  # everything: ~3,300 rows, ~$6.50, ~12 min
python -m evals.run_benchmark --order-check    # the order-sensitivity check, ~$0.40

python -m evals.report evals/results/20260923-063434.jsonl   # re-print a saved run
```

---

## 11. Caveats and open questions

- **The cards were written directly (by an LLM, with time to consider each one), before any
  test case existed** — so they couldn't be tailored to the cases, but they may still be a bit
  more careful than what a cheap, fast model would produce generating a card on the fly in
  production. If S1 keeps winning, re-run with cards generated the way production actually
  would before building this into the app.
- **The dataset is synthetic** — no real production traffic was available. It shows *where*
  each strategy breaks, not how often real users would hit each case.
- **One run per case.** With 20 cases per test, small gaps are noise; only the large,
  consistent gaps (like S0's) should be trusted.
- **Model mismatch:** the benchmark's IA runs DeepSeek V4.1 Flash (about as good as Sonnet 4 on
  a quick comparison, at a fraction of the cost); `server/config.py` still points production at
  a different model. The app should run what the benchmark measured before trusting these
  numbers for it.
- **Not done yet, only if a next version needs it:** confidence intervals / repeated runs, an
  actual router-agent variant of Idea 2, a pull-style `find_agent` tool, fixing the
  `related_context` instruction-following gap in the IA's own prompt, prompt caching (S3's
  shortlist changes every message, which defeats a cache; S1's card list is stable but large).

---

## 12. File map

```
evals/
  README.md                 this file
  requirements.txt          fastembed — the only extra package
  data/
    written/                authored directly; edit these, then rebuild
      agents.json             51 agents: episodes, card, key terms, decay slot, false friend
      cases.json               142 case records → 120 Growth + 110 Decay cases
      dev_cases.json           20 cases, only for S2's threshold
      growth_history.json      the fixed Growth chat
      small_talk.json          chit-chat that pads the Decay conversation
      filler_words.json        14 filler categories + word lists
    built/                  generated by build_dataset.py; committed so runs are repeatable
      fillers.json             1,000 filler agents
      growth_history.txt       the Growth chat, exactly as the IA sees it
      decay_conversation.json  the 1,090 Decay entries before summarization
      decay_history.txt        the Decay history after the real summarizer
      decay_levels.json        visible / summarized / gone per agent
      s2_threshold.json        S2's threshold and its dev score
      manifest.json            seed, models, and counts for this build
  results/                  one .jsonl per run, one row per case
  dataset.py                loads the data; builds each case's roster and history
  strategies.py             S0-S3: agent-list rendering, embedding search
  harness.py                runs one case through the real IA and grades it
  bench.py                  the seven tests + the order-sensitivity check
  run_benchmark.py          command line: runs tests, saves results, prints the report
  report.py                 turns result rows into the metric tables
  build_dataset.py          builds data/built/ from data/written/
tests/
  test_eval.py              the six checks above (no network, no API key)

Production code this benchmark reuses or checks against (not modified by it):
  server/agents/interaction_agent/agent.py:20   prepare_message_with_history (the prompt layout)
  server/agents/interaction_agent/agent.py:45   _render_active_agents (S0's rendering, today)
  server/agents/interaction_agent/tools.py:117  the exact-name reuse check
  server/agents/interaction_agent/system_prompt.md:19   "reuse if useful context" rule
  server/services/conversation/summarization/summarizer.py:73   summarize_conversation
  server/config.py:71-72                        summary threshold (100) and tail size (10)
  server/openrouter_client/client.py:57-78      reasoning/provider options (added for this benchmark)
```
