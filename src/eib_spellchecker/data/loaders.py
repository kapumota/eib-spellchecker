from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from eib_spellchecker.config import AppConfig
from eib_spellchecker.utils.text import normalize_text, tokenize_words


def iter_corpus_texts(paths: Iterable[Path]):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo de corpus: {path}")
        yield path.read_text(encoding="utf-8", errors="ignore")


def iter_corpus_tokens(config: AppConfig):
    for text in iter_corpus_texts(config.data.corpus_files):
        normalized = normalize_text(
            text,
            lowercase=config.normalize.lowercase,
            strip_accents_flag=config.normalize.strip_accents,
        )
        for token in tokenize_words(normalized):
            yield token


def build_frequency_table(config: AppConfig) -> Counter[str]:
    counter: Counter[str] = Counter()
    counter.update(iter_corpus_tokens(config))
    return counter
