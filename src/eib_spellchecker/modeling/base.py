# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TokenCorrection:
    original: str
    corrected: str
    changed: bool
    confidence: float | None = None
    reason: str | None = None


class SpellCheckerBackend(Protocol):
    language: str

    def correct_word(self, word: str) -> str: ...
    def correct_text(self, text: str) -> tuple[str, list[TokenCorrection]]: ...
    def suggest(self, word: str, limit: int = 5) -> list[str]: ...
