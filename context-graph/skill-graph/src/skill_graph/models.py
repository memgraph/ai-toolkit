from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timezone


@dataclass
class Skill:
    """Represents a skill that can be stored in and retrieved from Memgraph."""

    name: str
    description: str
    content: str
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
