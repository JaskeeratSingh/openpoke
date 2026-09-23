"""The four routing strategies, S0-S3, and the embedding search that S2 and S3 share.

S0  the IA sees every agent's name              (today's production behaviour)
S1  the IA sees every agent's card
S2  no IA call: the agent whose card is most similar to the request, or NEW
S3  the IA sees only the 5 most similar cards

harness.py calls these to build the agent list for one case.
"""

from html import escape

import numpy as np

from .dataset import card_text

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SHORTLIST_SIZE = 5

STRATEGIES = ["S0", "S1", "S2", "S3"]


def render_names(agents):
    """S0's agent list: one line per agent name, in the same format as production's
    _render_active_agents() (server/agents/interaction_agent/agent.py)."""
    if not agents:
        return "None"
    return "\n".join(f'<agent name="{escape(agent["name"], quote=True)}" />' for agent in agents)


def render_cards(agents):
    """S1's and S3's agent list: each agent's name plus its card."""
    if not agents:
        return "None"
    parts = []
    for agent in agents:
        name = escape(agent["name"], quote=True)
        parts.append(f'<agent name="{name}">\n{escape(card_text(agent), quote=False)}\n</agent>')
    return "\n".join(parts)


# ---------------------------------------------------------------- embedding search

_embedder = None
_vector_cache = {}


def _get_embedder():
    """Load the embedding model the first time it is needed (it takes a few seconds)."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(EMBEDDING_MODEL)
    return _embedder


def embed(texts):
    """Turn texts into unit-length vectors, so a dot product is the cosine similarity.

    Results are remembered, because the same agent cards are embedded for many cases.
    """
    missing = [text for text in texts if text not in _vector_cache]
    if missing:
        for text, vector in zip(missing, _get_embedder().embed(missing)):
            _vector_cache[text] = vector / np.linalg.norm(vector)
    return [_vector_cache[text] for text in texts]


def agent_search_text(agent):
    """What gets embedded for an agent: its name and its card."""
    return agent["name"] + "\n" + card_text(agent)


def rank_agents(agents, message):
    """All agents sorted from most to least similar to the request, as (agent, similarity) pairs.

    Only the new message is embedded, not the chat history: the last few history lines are
    usually small talk or some other agent's news, and they pulled the search off course.
    """
    request_vector = embed([message])[0]
    agent_vectors = embed([agent_search_text(agent) for agent in agents])
    scored = []
    for agent, vector in zip(agents, agent_vectors):
        scored.append((agent, float(np.dot(request_vector, vector))))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def shortlist(ranked):
    """S3: the top agents from rank_agents()."""
    return [agent for agent, _ in ranked[:SHORTLIST_SIZE]]


def pick_nearest(ranked, threshold):
    """S2: the most similar agent's name, or None (meaning "create a new agent")
    if even the best match is below the threshold."""
    best_agent, best_similarity = ranked[0]
    if best_similarity >= threshold:
        return best_agent["name"]
    return None


def gold_rank(ranked, answer):
    """Where the right agent sits in the ranking (1 = top), or None for create-new cases."""
    if answer is None:
        return None
    for position, (agent, _) in enumerate(ranked, start=1):
        if agent["name"] == answer:
            return position
    return None
