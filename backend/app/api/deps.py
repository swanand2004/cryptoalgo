from __future__ import annotations

from fastapi import Request

from app.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime
