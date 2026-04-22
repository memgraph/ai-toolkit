import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone

# Spec: lowercase alphanumeric and hyphens, no leading/trailing/consecutive hyphens, 1-64 chars
_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_MAX_NAME_LEN = 64
_MAX_DESCRIPTION_LEN = 1024
_MAX_COMPATIBILITY_LEN = 500


class SkillValidationError(ValueError):
    """Raised when a Skill field violates the Agent Skills specification."""


def validate_name(name: str) -> str:
    """Validate and return a spec-compliant skill name."""
    if not name or len(name) > _MAX_NAME_LEN:
        raise SkillValidationError(
            f"name must be 1-{_MAX_NAME_LEN} characters, got {len(name)}"
        )
    if "--" in name:
        raise SkillValidationError("name must not contain consecutive hyphens")
    if not _NAME_RE.match(name):
        raise SkillValidationError(
            "name must contain only lowercase letters, digits, and hyphens, "
            "and must not start or end with a hyphen"
        )
    return name


def validate_description(description: str) -> str:
    """Validate and return a spec-compliant description."""
    if not description or len(description) > _MAX_DESCRIPTION_LEN:
        raise SkillValidationError(
            f"description must be 1-{_MAX_DESCRIPTION_LEN} characters, got {len(description)}"
        )
    return description


def validate_compatibility(compatibility: Optional[str]) -> Optional[str]:
    """Validate the optional compatibility field."""
    if compatibility is not None:
        if not compatibility or len(compatibility) > _MAX_COMPATIBILITY_LEN:
            raise SkillValidationError(
                f"compatibility must be 1-{_MAX_COMPATIBILITY_LEN} characters, got {len(compatibility)}"
            )
    return compatibility


@dataclass
class Skill:
    """Represents a skill conforming to the Agent Skills specification.

    See https://agentskills.io/specification
    """

    # --- Required (spec) ---
    name: str
    description: str
    content: str

    # --- Optional (spec) ---
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    allowed_tools: List[str] = field(default_factory=list)

    # --- Graph extensions ---
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        validate_name(self.name)
        validate_description(self.description)
        validate_compatibility(self.compatibility)

        now = datetime.now(timezone.utc).isoformat()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
