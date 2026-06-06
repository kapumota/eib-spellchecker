# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable

from eib_spellchecker.modeling.base import TokenCorrection
from eib_spellchecker.modeling.manifest import ArtifactManifest
from eib_spellchecker.modeling.policy import (
    DecisionContext,
    LanguageProfile,
    SafetyPolicy,
    analyze_token,
    decide_action,
    soft_frequency_score,
)
from eib_spellchecker.utils.text import clean_token, normalize_text, preserve_case

TOKEN_OR_SEPARATOR_RE = re.compile(r"(\w+|\W+)", flags=re.UNICODE)


@dataclass
class LexicalArtifact:
    model_type: str
    language: str
    min_correction_length: int
    similarity_threshold: float
    vocabulary: list[str]
    frequencies: dict[str, int]
    safety_policy: dict = field(default_factory=dict)


class LexicalSpellChecker:
    def __init__(
        self,
        vocabulary: Iterable[str],
        frequencies: dict[str, int] | None = None,
        language: str = "unknown",
        min_correction_length: int = 3,
        similarity_threshold: float = 0.72,
        safety_policy: SafetyPolicy | None = None,
    ) -> None:
        self.language = language
        self.min_correction_length = min_correction_length
        self.similarity_threshold = similarity_threshold
        self.vocabulary = sorted(set(vocabulary))
        self.vocabulary_set = set(self.vocabulary)
        self.frequencies = frequencies or {}
        self.safety_policy = safety_policy or SafetyPolicy()
        self.language_profile = LanguageProfile.from_vocabulary(self.vocabulary)
        self._by_initial: dict[str, list[str]] = {}
        for token in self.vocabulary:
            self._by_initial.setdefault(token[:1], []).append(token)

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "LexicalSpellChecker":
        artifact_dir = Path(artifact_dir)
        payload = json.loads((artifact_dir / "model.json").read_text(encoding="utf-8"))
        return cls(
            vocabulary=payload["vocabulary"],
            frequencies=payload.get("frequencies", {}),
            language=payload.get("language", "unknown"),
            min_correction_length=payload.get("min_correction_length", 3),
            similarity_threshold=payload.get("similarity_threshold", 0.72),
            safety_policy=SafetyPolicy.from_mapping(payload.get("safety_policy")),
        )

    @staticmethod
    def write_artifact(
        artifact_dir: str | Path,
        *,
        language: str,
        min_correction_length: int,
        similarity_threshold: float,
        vocabulary: list[str],
        frequencies: dict[str, int],
        safety_policy: SafetyPolicy | None = None,
    ) -> Path:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = LexicalArtifact(
            model_type="lexical",
            language=language,
            min_correction_length=min_correction_length,
            similarity_threshold=similarity_threshold,
            vocabulary=vocabulary,
            frequencies=frequencies,
            safety_policy=(safety_policy or SafetyPolicy()).to_dict(),
        )
        (artifact_dir / "model.json").write_text(json.dumps(asdict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        ArtifactManifest(
            artifact_version="2",
            backend="lexical",
            language=language,
            entrypoint="eib_spellchecker.modeling.lexical:LexicalSpellChecker",
            payload_file="model.json",
        ).write(artifact_dir)
        return artifact_dir

    def suggest(self, word: str, limit: int = 5) -> list[str]:
        normalized = normalize_text(clean_token(word), lowercase=True, strip_accents_flag=False)
        if not normalized:
            return []
        pool = self._candidate_pool(normalized)
        candidates = get_close_matches(
            normalized,
            pool,
            n=max(limit * 2, 10),
            cutoff=max(0.5, self.similarity_threshold - 0.15),
        )
        if not candidates:
            candidates = pool[: max(limit * 2, 10)]
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                self._levenshtein(normalized, candidate),
                -self.frequencies.get(candidate, 0),
                abs(len(candidate) - len(normalized)),
            ),
        )
        return ranked[:limit]

    def _score_candidate(self, normalized: str, candidate: str) -> float:
        similarity = 1.0 - (self._levenshtein(normalized, candidate) / max(len(normalized), len(candidate), 1))
        return max(0.0, min(1.0, 0.84 * similarity + 0.16 * soft_frequency_score(self.frequencies.get(candidate, 0))))

    def correct_token(self, word: str) -> TokenCorrection:
        cleaned = clean_token(word)
        if not cleaned:
            return TokenCorrection(original=word, corrected=word, changed=False, confidence=1.0, reason="empty-token")

        normalized = normalize_text(cleaned, lowercase=True, strip_accents_flag=False)
        if len(normalized) < self.min_correction_length:
            return TokenCorrection(original=word, corrected=word, changed=False, confidence=1.0, reason="too-short")
        if normalized in self.vocabulary_set:
            return TokenCorrection(original=word, corrected=preserve_case(word, normalized), changed=False, confidence=1.0, reason="in-vocabulary")

        shortlist = self.suggest(normalized, limit=10)
        if not shortlist:
            return TokenCorrection(original=word, corrected=word, changed=False, confidence=0.0, reason="no-candidates")

        best = shortlist[0]
        best_score = self._score_candidate(normalized, best)
        identity_score = self._score_candidate(normalized, normalized) if normalized in self.vocabulary_set else 0.0
        profile = analyze_token(word, self.language_profile, self.safety_policy)
        corrected_text, reason, confidence = decide_action(
            DecisionContext(
                original=word,
                normalized=normalized,
                best_candidate=best,
                best_score=best_score,
                identity_score=identity_score,
                margin=best_score - identity_score,
                profile=profile,
                policy=self.safety_policy,
            )
        )
        if corrected_text == word and reason == "best-candidate" and best_score < self.similarity_threshold:
            corrected_text = word
            reason = "below-threshold"
            confidence = best_score
        elif corrected_text != word:
            corrected_text = preserve_case(word, corrected_text)
        return TokenCorrection(
            original=word,
            corrected=corrected_text,
            changed=word != corrected_text,
            confidence=confidence,
            reason=reason,
        )

    def correct_word(self, word: str) -> str:
        return self.correct_token(word).corrected

    def correct_text(self, text: str) -> tuple[str, list[TokenCorrection]]:
        pieces: list[str] = []
        corrections: list[TokenCorrection] = []
        for piece in TOKEN_OR_SEPARATOR_RE.findall(text):
            if any(char.isalpha() for char in piece):
                correction = self.correct_token(piece)
                corrections.append(correction)
                pieces.append(correction.corrected)
            else:
                pieces.append(piece)
        return "".join(pieces), corrections

    def _candidate_pool(self, normalized: str) -> list[str]:
        initial = normalized[:1]
        pool = self._by_initial.get(initial, [])
        if len(pool) < 10:
            return [token for token in self.vocabulary if abs(len(token) - len(normalized)) <= 2] or self.vocabulary
        return [token for token in pool if abs(len(token) - len(normalized)) <= 2] or pool

    @staticmethod
    def _levenshtein(left: str, right: str) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                insertion = current[j - 1] + 1
                deletion = previous[j] + 1
                substitution = previous[j - 1] + (left_char != right_char)
                current.append(min(insertion, deletion, substitution))
            previous = current
        return previous[-1]
