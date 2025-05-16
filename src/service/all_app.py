from fastapi import FastAPI
from core.db import register_events
from .routes import conversations, users
from service import service
from memory import initialize_database, initialize_store
from collections.abc import AsyncGenerator
from agents.smol_agents import DEFAULT_AGENT, get_agent, get_all_agent_info
import logging
import warnings
from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Chat Assistant API",
    version="1.0.0",
)
register_events(app)

@app.get("/health",include_in_schema=False)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
async def root():
    return {"status": "online", "info": "/info", "health": "/health"}

app.include_router(users.router)
app.include_router(conversations.router)
app.include_router(service.router)  # this is the router for the agent service

# existing routes (agent invoke, feedback …) stay unchanged
