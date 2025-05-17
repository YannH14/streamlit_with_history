from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import UUID4, BaseModel

from auth import User, get_current_user

router = APIRouter(prefix="/conversations", tags=["chat"])


class ConversationOut(BaseModel):
    thread_id: UUID4
    title: str
    updated_at: str


class ChatMessageOut(BaseModel):
    sender: str
    content: str
    timestamp: str


async def _ensure_owner(pool, thread_id: UUID4, user_id: str):
    owner = await pool.fetchval(
        "SELECT user_id FROM conversations WHERE thread_id=$1", thread_id
    )
    if owner != user_id:
        raise HTTPException(status_code=403, detail="Not your conversation")


@router.get("", response_model=list[ConversationOut])
async def list_conversations(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
):
    rows = await request.app.state.pool.fetch(
        """
        SELECT thread_id, title, updated_at
          FROM conversations
         WHERE user_id=$1
      ORDER BY updated_at DESC
        """,
        current_user.id,
    )
    return [dict(r) for r in rows]


@router.get("/{thread_id}", response_model=list[ChatMessageOut])
async def get_conversation(
    thread_id: UUID4,
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
):
    await _ensure_owner(request.app.state.pool, thread_id, current_user.id)
    rows = await request.app.state.pool.fetch(
        """
        SELECT sender_type AS sender, content, timestamp
          FROM chat_messages
         WHERE thread_id=$1
      ORDER BY timestamp
        """,
        thread_id,
    )
    return [dict(r) for r in rows]
