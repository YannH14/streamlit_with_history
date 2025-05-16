from functools import cache
from typing import TypeAlias

from langchain_openai import ChatOpenAI
from core.settings import settings
from schema.models import AllModelEnum, OpenAICompatibleName

ModelT: TypeAlias = ChatOpenAI

@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OpenAICompatible base url and endpoint must be configured")
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.5,
        streaming=True,
        openai_api_key=settings.OPENAI_API_KEY,
    )