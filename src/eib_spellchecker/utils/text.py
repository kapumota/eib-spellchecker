# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

import re
import unicodedata
from typing import List

WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", flags=re.UNICODE)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str, lowercase: bool = True, strip_accents_flag: bool = False) -> str:
    if strip_accents_flag:
        text = strip_accents(text)
    if lowercase:
        text = text.lower()
    return text


def clean_token(token: str) -> str:
    token = token.strip()
    cleaned = []
    for char in token:
        if char.isalpha() or char in {"-", "'", "’"}:
            cleaned.append(char)
    token = "".join(cleaned)
    return token.strip("-'’")


def tokenize_words(text: str) -> List[str]:
    return [token for raw in WORD_RE.findall(text) if (token := clean_token(raw))]


def preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    return replacement
