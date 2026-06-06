# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from eib_spellchecker.modeling.base import TokenCorrection
from eib_spellchecker.modeling.manifest import ArtifactManifest
from eib_spellchecker.modeling.policy import (
    DecisionContext,
    LanguageProfile,
    SafetyPolicy,
    analyze_token,
    char_ngrams,
    decide_action,
    soft_frequency_score,
)
from eib_spellchecker.utils.text import clean_token, normalize_text, preserve_case

TOKEN_OR_SEPARATOR_RE = re.compile(r"(\w+|\W+)", flags=re.UNICODE)


@dataclass
class SubwordArtifact:
    model_type: str
    language: str
    vocabulary: list[str]
    frequencies: dict[str, int]
    min_correction_length: int
    max_candidates: int
    min_ngram: int
    max_ngram: int
    jaccard_weight: float
    edit_weight: float
    frequency_weight: float
    score_threshold: float
    safety_policy: dict = field(default_factory=dict)


class SubwordSpellChecker:
    def __init__(
        self,
        vocabulary: Iterable[str],
        frequencies: dict[str, int] | None = None,
        *,
        language: str = "unknown",
        min_correction_length: int = 3,
        max_candidates: int = 32,
        min_ngram: int = 2,
        max_ngram: int = 4,
        jaccard_weight: float = 0.5,
        edit_weight: float = 0.35,
        frequency_weight: float = 0.15,
        score_threshold: float = 0.58,
        safety_policy: SafetyPolicy | None = None,
    ) -> None:
        self.language = language
        self.vocabulary = sorted(set(vocabulary))
        self.vocabulary_set = set(self.vocabulary)
        self.frequencies = frequencies or {}
        self.min_correction_length = min_correction_length
        self.max_candidates = max_candidates
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.jaccard_weight = jaccard_weight
        self.edit_weight = edit_weight
        self.frequency_weight = frequency_weight
        self.score_threshold = score_threshold
        self.safety_policy = safety_policy or SafetyPolicy()
        self.language_profile = LanguageProfile.from_vocabulary(self.vocabulary)
        self._token_ngrams: dict[str, set[str]] = {}
        self._ngram_index: dict[str, set[str]] = {}
        for token in self.vocabulary:
            grams = char_ngrams(token, min_n=min_ngram, max_n=max_ngram)
            self._token_ngrams[token] = grams
            for gram in grams:
                self._ngram_index.setdefault(gram, set()).add(token)

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "SubwordSpellChecker":
        artifact_dir = Path(artifact_dir)
        payload = json.loads((artifact_dir / "model.json").read_text(encoding="utf-8"))
        return cls(
            vocabulary=payload["vocabulary"],
            frequencies=payload.get("frequencies", {}),
            language=payload.get("language", "unknown"),
            min_correction_length=payload.get("min_correction_length", 3),
            max_candidates=payload.get("max_candidates", 32),
            min_ngram=payload.get("min_ngram", 2),
            max_ngram=payload.get("max_ngram", 4),
            jaccard_weight=payload.get("jaccard_weight", 0.5),
            edit_weight=payload.get("edit_weight", 0.35),
            frequency_weight=payload.get("frequency_weight", 0.15),
            score_threshold=payload.get("score_threshold", 0.58),
            safety_policy=SafetyPolicy.from_mapping(payload.get("safety_policy")),
        )

    @staticmethod
    def write_artifact(
        artifact_dir: str | Path,
        *,
        language: str,
        vocabulary: list[str],
        frequencies: dict[str, int],
        min_correction_length: int,
        max_candidates: int,
        min_ngram: int,
        max_ngram: int,
        jaccard_weight: float,
        edit_weight: float,
        frequency_weight: float,
        score_threshold: float,
        safety_policy: SafetyPolicy | None = None,
    ) -> Path:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = SubwordArtifact(
            model_type="subword",
            language=language,
            vocabulary=vocabulary,
            frequencies=frequencies,
            min_correction_length=min_correction_length,
            max_candidates=max_candidates,
            min_ngram=min_ngram,
            max_ngram=max_ngram,
            jaccard_weight=jaccard_weight,
            edit_weight=edit_weight,
            frequency_weight=frequency_weight,
            score_threshold=score_threshold,
            safety_policy=(safety_policy or SafetyPolicy()).to_dict(),
        )
        (artifact_dir / "model.json").write_text(json.dumps(asdict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        ArtifactManifest(
            artifact_version="2",
            backend="subword",
            language=language,
            entrypoint="eib_spellchecker.modeling.subword:SubwordSpellChecker",
            payload_file="model.json",
        ).write(artifact_dir)
        return artifact_dir

    def _levenshtein(self, left: str, right: str) -> int:
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

    def _score_candidate(self, normalized: str, candidate: str) -> float:
        grams = char_ngrams(normalized, min_n=self.min_ngram, max_n=self.max_ngram)
        cand_grams = self._token_ngrams.get(candidate) or char_ngrams(candidate, min_n=self.min_ngram, max_n=self.max_ngram)
        union = len(grams | cand_grams)
        jaccard = (len(grams & cand_grams) / union) if union else 0.0
        edit_similarity = 1.0 - (self._levenshtein(normalized, candidate) / max(len(normalized), len(candidate), 1))
        freq_score = soft_frequency_score(self.frequencies.get(candidate, 0))
        score = (
            self.jaccard_weight * jaccard
            + self.edit_weight * max(0.0, edit_similarity)
            + self.frequency_weight * freq_score
        )
        return max(0.0, min(1.0, score))

    def suggest(self, word: str, limit: int = 5) -> list[str]:
        normalized = normalize_text(clean_token(word), lowercase=True, strip_accents_flag=False)
        if not normalized:
            return []
        grams = char_ngrams(normalized, min_n=self.min_ngram, max_n=self.max_ngram)
        pool: set[str] = set()
        for gram in grams:
            pool.update(self._ngram_index.get(gram, set()))
        if not pool:
            pool = {token for token in self.vocabulary if abs(len(token) - len(normalized)) <= 3}
        if normalized in self.vocabulary_set:
            pool.add(normalized)
        scored = [
            (candidate, self._score_candidate(normalized, candidate))
            for candidate in pool
            if abs(len(candidate) - len(normalized)) <= 4
        ]
        scored.sort(key=lambda item: (-item[1], -self.frequencies.get(item[0], 0), item[0]))
        return [candidate for candidate, _ in scored[:limit]]

    def correct_token(self, word: str) -> TokenCorrection:
        cleaned = clean_token(word)
        if not cleaned:
            return TokenCorrection(original=word, corrected=word, changed=False, confidence=1.0, reason="empty-token")
        normalized = normalize_text(clean_token(word), lowercase=True, strip_accents_flag=False)
        if len(normalized) < self.min_correction_length:
            return TokenCorrection(original=word, corrected=word, changed=False, confidence=1.0, reason="too-short")
        if normalized in self.vocabulary_set:
            return TokenCorrection(original=word, corrected=preserve_case(word, normalized), changed=False, confidence=1.0, reason="in-vocabulary")
        candidates = self.suggest(normalized, limit=self.max_candidates)
        if not candidates:
            return TokenCorrection(original=word, corrected=word, changed=False, confidence=0.0, reason="no-candidates")
        scored = [(candidate, self._score_candidate(normalized, candidate)) for candidate in candidates if candidate != normalized]
        if not scored:
            return TokenCorrection(original=word, corrected=word, changed=False, confidence=0.0, reason="no-candidates")
        scored.sort(key=lambda item: (-item[1], -self.frequencies.get(item[0], 0), item[0]))
        best_candidate, best_score = scored[0]
        profile = analyze_token(word, self.language_profile, self.safety_policy)
        identity_score = 0.18 + (0.22 * profile.known_char_coverage)
        corrected_text, reason, confidence = decide_action(
            DecisionContext(
                original=word,
                normalized=normalized,
                best_candidate=best_candidate,
                best_score=best_score,
                identity_score=identity_score,
                margin=best_score - identity_score,
                profile=profile,
                policy=self.safety_policy,
            )
        )
        if corrected_text == word and reason == "best-candidate" and best_score < self.score_threshold:
            corrected_text = word
            reason = "below-threshold"
            confidence = best_score
        elif corrected_text != word:
            corrected_text = preserve_case(word, corrected_text)
        return TokenCorrection(
            original=word,
            corrected=corrected_text,
            changed=(corrected_text != word),
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
                pieces.append(correction.corrected)
                corrections.append(correction)
            else:
                pieces.append(piece)
        return "".join(pieces), corrections
