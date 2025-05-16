from fastapi import FastAPI
from core.db import register_events
from .routes import conversations, users

app = FastAPI(title="Agent Service")
register_events(app)


app.include_router(users.router)
app.include_router(conversations.router)

# existing routes (agent invoke, feedback …) stay unchanged
