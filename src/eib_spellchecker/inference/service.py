# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

from pathlib import Path

from eib_spellchecker.modeling.base import TokenCorrection
from eib_spellchecker.modeling.legacy_seq2seq import LegacySeq2SeqSpellChecker
from eib_spellchecker.modeling.lexical import LexicalSpellChecker
from eib_spellchecker.modeling.manifest import load_manifest
from eib_spellchecker.modeling.subword import SubwordSpellChecker
from eib_spellchecker.modeling.torch_reranker import TorchHybridSpellChecker


class ArtifactSpellChecker:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.language = backend.language

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "ArtifactSpellChecker":
        artifact_dir = Path(artifact_dir)
        manifest = load_manifest(artifact_dir)
        backend_name = manifest["backend"]
        if backend_name == "lexical":
            return cls(LexicalSpellChecker.from_artifact_dir(artifact_dir))
        if backend_name == "legacy-seq2seq":
            return cls(LegacySeq2SeqSpellChecker.from_artifact_dir(artifact_dir))
        if backend_name == "torch-hybrid-reranker":
            return cls(TorchHybridSpellChecker.from_artifact_dir(artifact_dir))
        if backend_name == "subword":
            return cls(SubwordSpellChecker.from_artifact_dir(artifact_dir))
        raise ValueError(f"Backend no soportado: {backend_name}")

    def suggest(self, word: str, limit: int = 5) -> list[str]:
        return self.backend.suggest(word, limit=limit)

    def correct_token(self, word: str) -> TokenCorrection:
        if hasattr(self.backend, "correct_token"):
            return self.backend.correct_token(word)
        corrected = self.backend.correct_word(word)
        return TokenCorrection(original=word, corrected=corrected, changed=corrected != word)

    def correct_word(self, word: str) -> str:
        return self.correct_token(word).corrected

    def correct_text(self, text: str) -> tuple[str, list[TokenCorrection]]:
        return self.backend.correct_text(text)

    def describe(self) -> dict:
        return {
            "language": self.language,
            "backend": self.backend.__class__.__name__,
        }
