"""The judges' shared OpenAI call: bounded concurrency, and a failed call reported as None.

Each judge passes its upstream's request parameters through unchanged; only
the plumbing around the call lives here.
"""

from __future__ import annotations

import asyncio
from typing import Any


class BoundedChat:
    """Chat completions, at most ``max_concurrent`` in flight; a call that raises yields None.

    A failed call is unjudged, not wrong: counting it as a miss would report
    an outage as a score.
    """

    def __init__(self, *, client: Any = None, max_concurrent: int = 8):
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(max_retries=5)
        self._client: Any = client
        self._limiter = asyncio.Semaphore(max_concurrent)

    async def complete(self, **request: Any) -> str | None:
        """The first choice's content ("" when empty) for ``request``, or None when the call raised."""
        async with self._limiter:
            try:
                completion = await self._client.chat.completions.create(**request)
            except Exception:
                return None
        return completion.choices[0].message.content or ""
