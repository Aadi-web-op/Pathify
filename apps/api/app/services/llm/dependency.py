from typing import Annotated

from fastapi import Depends, Request

from app.services.llm.base import LLMProvider


def get_llm(request: Request) -> LLMProvider:
    return request.app.state.llm


LLMDep = Annotated[LLMProvider, Depends(get_llm)]
