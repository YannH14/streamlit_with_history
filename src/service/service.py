# fastapi_smolagents_app.py
"""
FastAPI service powered by **smolagents** – v3.1
================================================
Adds **persistent chat history** (conversations + messages) and feedback tracking via PostgreSQL **using UUID primary keys** and
**AI‑generated conversation titles** with a graceful heuristic fallback.

Environment variables:
* `DATABASE_URL` – Postgres connection string (optional; disables persistence if absent).
* `OPENAI_API_KEY`, `OPENAI_MODEL_ID` – control LLM routing.
* `AUTH_SECRET` – bearer token for authenticated routes.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from smolagents import CodeAgent

# Local imports --------------------------------------------------------
from agents.smol_agents import DEFAULT_AGENT, get_agent, get_all_agent_info
from core import settings
from memory import initialize_database, initialize_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)

# Optional AI summariser (non‑critical)
try:
    from agents.summarizer import summarize_title  # noqa: WPS433 – optional runtime import
except Exception:  # pragma: no cover
    summarize_title = None  # type: ignore  # noqa: WPS410

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Database pool – single pool for conversations, messages, feedback
# ---------------------------------------------------------------------

_DB_POOL: asyncpg.Pool | None = None

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS conversations (
    thread_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_convos_user ON conversations(user_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   UUID REFERENCES conversations(thread_id) ON DELETE CASCADE,
    sender_type TEXT CHECK (sender_type IN ('human', 'ai')) NOT NULL,
    content     TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_msgs_thread ON chat_messages(thread_id);

CREATE TABLE IF NOT EXISTS feedback (
    id       BIGSERIAL PRIMARY KEY,
    run_id   TEXT,
    key      TEXT,
    score    DOUBLE PRECISION,
    meta     JSONB,
    ts       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def _init_db_pool() -> None:
    """Initialise asyncpg pool and ensure schema is present."""
    global _DB_POOL  # noqa: WPS420 – single shared pool
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.info("DATABASE_URL not set – persistence disabled")
        return

    try:
        _DB_POOL = await asyncpg.create_pool(database_url, min_size=1, max_size=10)
        async with _DB_POOL.acquire() as conn:
            await conn.execute(_SCHEMA_SQL)
        logger.info("Database schema ready (%s)", database_url)
    except Exception as exc:  # pragma: no cover – startup failure is non‑fatal
        logger.warning("Cannot initialise database (%s) – persistence disabled", exc)
        _DB_POOL = None


async def _close_db_pool() -> None:  # noqa: WPS430 – symmetrical teardown
    if _DB_POOL:
        await _DB_POOL.close()


# ---------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------

_MAX_TITLE_WORDS = 10


def _heuristic_title(msg: str) -> str:
    """Simple first‑N‑words fallback."""
    words = msg.split()
    base = " ".join(words[:_MAX_TITLE_WORDS])
    return base + (" …" if len(words) > _MAX_TITLE_WORDS else "")


async def _generate_title(msg: str) -> str:  # noqa: WPS231 – small helper
    """Try AI summarisation, fall back to heuristic."""
    if summarize_title:
        try:
            if asyncio.iscoroutinefunction(summarize_title):
                title = await summarize_title(msg)
            else:  # type: ignore[func-returns-value]
                loop = asyncio.get_running_loop()
                title = await loop.run_in_executor(None, summarize_title, msg)
            if title:
                return title.strip()[:120]
        except Exception as exc:  # pragma: no cover
            logger.info("summarize_title failed, falling back (%s)", exc)
    return _heuristic_title(msg)


async def _conversation_exists(thread_id: UUID) -> bool:
    if not _DB_POOL:
        return False
    async with _DB_POOL.acquire() as conn:
        return (
            await conn.fetchval(
                "SELECT 1 FROM conversations WHERE thread_id=$1", thread_id
            )
            is not None
        )


async def _upsert_conversation(thread_id: UUID, user_id: str, first_user_msg: str) -> None:  # noqa: E501
    if not _DB_POOL:
        return
    title = await _generate_title(first_user_msg)
    async with _DB_POOL.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO conversations(thread_id, user_id, title)
            VALUES($1, $2, $3)
            ON CONFLICT(thread_id)
            DO UPDATE SET updated_at = NOW()
            """,
            thread_id,
            user_id,
            title,
        )


async def _insert_message(thread_id: UUID, sender: str, content: str) -> None:
    if not _DB_POOL:
        return
    async with _DB_POOL.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_messages(thread_id, sender_type, content) VALUES($1,$2,$3)",
            thread_id,
            sender,
            content,
        )
        await conn.execute(
            "UPDATE conversations SET updated_at = NOW() WHERE thread_id=$1",
            thread_id,
        )


# ---------------------------------------------------------------------
# Helpers: run smolagents in thread, token streaming
# ---------------------------------------------------------------------

async def _run_agent(agent: CodeAgent, task: str) -> str:  # noqa: WPS110 – third‑party name
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, agent.run, task)


def _yield_tokens(text: str):  # noqa: WPS110 – generator helper
    for tok in text.split():
        yield tok + " "


async def _stream_tokens(answer: str):
    for tkn in _yield_tokens(answer):
        yield tkn
        await asyncio.sleep(0)  # allow event loop switch


# ---------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------

def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(
            HTTPBearer(
                description="Provide AUTH_SECRET bearer token.", auto_error=False
            )
        ),
    ],
) -> None:
    """Simple constant secret check."""
    if settings.AUTH_SECRET:
        if not http_auth or http_auth.credentials != settings.AUTH_SECRET.get_secret_value():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


# ---------------------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: WPS430 – FP style
    await _init_db_pool()
    try:
        async with initialize_database() as saver, initialize_store() as store:  # noqa: WPS440 – contextlib nesting
            if hasattr(saver, "setup"):
                await saver.setup()  # type: ignore[attr-defined]
            if hasattr(store, "setup"):
                await store.setup()  # type: ignore[attr-defined]
            for info in get_all_agent_info():
                ag = get_agent(info.key)
                ag.checkpointer = saver  # type: ignore[attr-defined]
                ag.store = store  # type: ignore[attr-defined]
            yield
    except Exception as exc:  # pragma: no cover – fallback to memory‑only
        logger.warning("Memory DB unavailable – in‑memory only (%s)", exc)
        yield
    await _close_db_pool()


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])

# ---------------------------------------------------------------------
# Routes: info, conversations
# ---------------------------------------------------------------------

@router.get("/info", response_model=ServiceMetadata)
async def info() -> ServiceMetadata:  # noqa: D401 – FastAPI returns models
    models = sorted(settings.AVAILABLE_MODELS)
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


@router.get("/conversations")
async def list_conversations(user_id: str = Query(...)) -> list[Dict[str, Any]]:  # noqa: WPS211 – simple SQL
    if not _DB_POOL:
        return []
    async with _DB_POOL.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT thread_id, title, updated_at
              FROM conversations
             WHERE user_id=$1
         ORDER BY updated_at DESC
            """,
            user_id,
        )
    return [
        {
            "thread_id": str(r["thread_id"]),
            "title": r["title"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


@router.get("/conversations/{thread_id}")
async def get_conversation(
    thread_id: UUID, user_id: str = Query(...)
) -> list[Dict[str, Any]]:
    if not _DB_POOL:
        return []
    async with _DB_POOL.acquire() as conn:
        owner = await conn.fetchval(
            "SELECT user_id FROM conversations WHERE thread_id=$1", thread_id
        )
        if owner != user_id:
            raise HTTPException(status_code=403, detail="Not your conversation")
        msgs = await conn.fetch(
            """
            SELECT sender_type, content, ts
              FROM chat_messages
             WHERE thread_id=$1
         ORDER BY ts
            """,
            thread_id,
        )
    return [
        {"type": r["sender_type"], "content": r["content"], "timestamp": r["ts"]}
        for r in msgs
    ]


# ---------------------------------------------------------------------
# Input helper
# ---------------------------------------------------------------------

async def _handle_input(user_input: UserInput) -> tuple[str, UUID]:  # noqa: WPS110 – domain term
    return user_input.message, uuid4()


# ---------------------------------------------------------------------
# Invoke & stream (persisted)
# ---------------------------------------------------------------------

@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:  # noqa: WPS211 – sequential flow
    message, run_id = await _handle_input(user_input)
    thread_id = UUID(user_input.thread_id) if user_input.thread_id else uuid4()
    user_id = user_input.user_id or ""

    # Ensure conversation row exists, then persist user message
    if not await _conversation_exists(thread_id):
        await _upsert_conversation(thread_id, user_id, message)
    await _insert_message(thread_id, "human", message)

    # Call agent
    agent = get_agent(agent_id)
    try:
        answer = await _run_agent(agent, message)
    except Exception as exc:  # pragma: no cover – propagate nicely
        logger.exception("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail="Agent execution error")

    # Persist assistant message
    await _insert_message(thread_id, "ai", answer)
    return ChatMessage(type="ai", content=answer, run_id=str(run_id))


@router.post("/{agent_id}/stream", response_class=StreamingResponse)
@router.post("/stream", response_class=StreamingResponse)
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:  # noqa: WPS211 – generator route
    message = user_input.message
    thread_id = UUID(user_input.thread_id) if user_input.thread_id else uuid4()
    user_id = user_input.user_id or ""

    if not await _conversation_exists(thread_id):
        await _upsert_conversation(thread_id, user_id, message)
    await _insert_message(thread_id, "human", message)

    async def event_gen():  # noqa: WPS430 – nested async generator
        agent = get_agent(agent_id)
        try:
            raw_answer = await _run_agent(agent, message)
            answer = str(raw_answer)
        except Exception as exc:  # pragma: no cover
            logger.exception("Stream error: %s", exc)
            payload = {"type": "error", "content": "Internal server error"}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Stream individual tokens
        async for tok in _stream_tokens(answer):
            payload = {"type": "token", "content": tok}
            yield f"data: {json.dumps(payload)}\n\n"

        # Persist and send final
        await _insert_message(thread_id, "ai", answer)
        final = ChatMessage(type="ai", content=answer, run_id=str(uuid4()))
        payload = {"type": "message", "content": final.model_dump()}
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------
# Feedback endpoint (unchanged logic, uses same pool)
# ---------------------------------------------------------------------

@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:  # noqa: WPS110 – domain term
    if not _DB_POOL:
        logger.info("/feedback ignored – no DB configured")
        return FeedbackResponse()
    meta_json = json.dumps(feedback.kwargs or {})
    try:
        await _DB_POOL.execute(
            """
            INSERT INTO feedback(run_id, key, score, meta, ts)
                 VALUES($1, $2, $3, $4, $5)
            """,
            feedback.run_id,
            feedback.key,
            feedback.score,
            meta_json,
            _dt.datetime.utcnow(),
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Feedback write failed: %s", exc)
    return FeedbackResponse()


# ---------------------------------------------------------------------
# History route (deprecated – kept for compatibility)
# ---------------------------------------------------------------------

@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:  # noqa: D401 – FastAPI returns model
    agent = get_agent(DEFAULT_AGENT)
    steps = agent.memory.get_full_steps()  # type: ignore[attr-defined]
    return ChatHistory(messages=[ChatMessage(type="ai", content=str(s)) for s in steps])


@app.get("/health")
async def health_check():  # noqa: D401 – simple health route
    return {"status": "ok"}

# Register router ------------------------------------------------------
app.include_router(router)
