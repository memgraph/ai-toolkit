"""The error every model reader raises, carrying the contract's error codes (docs/contract.md, section 9)."""

from collections.abc import Sequence


class ModelError(ValueError):
    """A model or mapping file that breaks the contract.

    A ValueError, so callers that caught ValueError before error codes existed
    keep working.

    Attributes:
        codes: Every contract error code found, at least one. A reader in any
            language reports codes from the same table; messages are its own.
    """

    def __init__(self, codes: str | Sequence[str], message: str) -> None:
        super().__init__(message)
        self.codes: tuple[str, ...] = (codes,) if isinstance(codes, str) else tuple(codes)

    @property
    def code(self) -> str:
        """The first code found."""
        return self.codes[0]
