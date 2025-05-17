# agents_smol.py
"""SmolAgents registry with OpenAI as the default backend.

This file **replaces** the LangGraph Pregel graph registry.  It exposes the same
public interface: `get_agent`, `get_all_agent_info`, and `DEFAULT_AGENT`.

Changes in **v2**
-----------------
* Switched the model factory to **OpenAIServerModel** when `OPENAI_API_KEY` is
  available.
* Clean fallback to `InferenceClientModel` (Hugging Face Inference API) if no
  OpenAI credentials are found.
* Added ``OPENAI_MODEL_ID`` env var (defaults to ``gpt-4o-mini``).

Set `OPENAI_API_KEY` in your environment and the agents will automatically route
requests to OpenAI instead of the default public Hugging Face Qwen endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from smolagents import CodeAgent, InferenceClientModel, OpenAIServerModel

from schema import AgentInfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_AGENT = "research-assistant"

# ---------------------------------------------------------------------------
# Dataclass – holds a ready‑to‑use CodeAgent and its description
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    description: str
    agent: CodeAgent


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


_AGENTS: dict[str, Agent] = {}


def _create_model():
    """Return the appropriate model object based on environment variables."""

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        # Use OpenAI – model ID can be overridden via env
        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
        return OpenAIServerModel(api_key=openai_key, model_id=model_id)

    # Otherwise default to the free HF Inference API (rate‑limited!)
    hf_model_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen1.5-7B-Chat")
    return InferenceClientModel(model_id=hf_model_id)


def _create_agent(description: str) -> CodeAgent:
    model = _create_model()
    tools = []  # plug your domain‑specific tools here
    return CodeAgent(
        model=model,
        tools=tools,
        verbosity_level=1,
        max_steps=5,
    )


def _register(key: str, description: str):
    if key not in _AGENTS:
        _AGENTS[key] = Agent(description=description, agent=_create_agent(description))


# ---------------------------------------------------------------------------
# Populate the registry – mirrors the original agent IDs one‑to‑one
# ---------------------------------------------------------------------------

_register("chatbot", "A simple chatbot.")
_register("research-assistant", "A research assistant with web search and calculator.")
_register("rag-assistant", "A RAG assistant with database knowledge.")
_register("command-agent", "A command agent.")
_register("bg-task-agent", "A background task agent.")
_register("langgraph-supervisor-agent", "A langgraph supervisor agent.")
_register("interrupt-agent", "An agent that uses interrupts.")
_register(
    "knowledge-base-agent",
    "A retrieval‑augmented generation agent using a Knowledge Base.",
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def get_agent(agent_id: str) -> CodeAgent:
    if agent_id not in _AGENTS:
        raise KeyError(
            f"Unknown agent id '{agent_id}'. Available: {list(_AGENTS)[:5]}…"
        )
    return _AGENTS[agent_id].agent


def get_all_agent_info() -> list[AgentInfo]:
    return [AgentInfo(key=k, description=a.description) for k, a in _AGENTS.items()]
