# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from eib_spellchecker.modeling.base import TokenCorrection
from eib_spellchecker.modeling.lexical import LexicalSpellChecker
from eib_spellchecker.modeling.manifest import ArtifactManifest
from eib_spellchecker.modeling.policy import (
    DecisionContext,
    LanguageProfile,
    SafetyPolicy,
    analyze_token,
    decide_action,
)
from eib_spellchecker.utils.text import clean_token, normalize_text, preserve_case

TOKEN_OR_SEPARATOR_RE = re.compile(r"(\w+|\W+)", flags=re.UNICODE)
PAD = "<pad>"
UNK = "<unk>"



def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False



def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pack_padded_sequence
    except Exception as exc:
        raise RuntimeError(
            "El backend torch-hybrid-reranker requiere PyTorch. Instala el extra con: pip install -e .[torch]"
        ) from exc
    return torch, nn, F, pack_padded_sequence


@dataclass
class TorchRerankerMetadata:
    model_type: str
    language: str
    max_length: int
    chars: list[str]
    embedding_dim: int
    hidden_size: int
    candidate_limit: int
    min_correction_length: int
    similarity_threshold: float
    score_threshold: float
    weights_file: str
    vocabulary_file: str
    safety_policy: dict = field(default_factory=dict)


class CharPairRerankerModelBase:
    def __init__(self, embedding_dim: int, hidden_size: int, vocab_size: int) -> None:
        torch, nn, _, pack_padded_sequence = require_torch()
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.encoder = nn.GRU(
                    embedding_dim,
                    hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
                feature_dim = hidden_size * 8 + 1
                self._pack_padded_sequence = pack_padded_sequence
                self.head = nn.Sequential(
                    nn.Linear(feature_dim, hidden_size * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size * 2, 1),
                )

            def encode(self, tokens, lengths):
                packed = self._pack_padded_sequence(
                    self.embedding(tokens),
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, hidden = self.encoder(packed)
                return torch.cat([hidden[-2], hidden[-1]], dim=-1)

            def forward(self, noisy_tokens, noisy_lengths, candidate_tokens, candidate_lengths):
                noisy_repr = self.encode(noisy_tokens, noisy_lengths)
                cand_repr = self.encode(candidate_tokens, candidate_lengths)
                length_feature = ((candidate_lengths - noisy_lengths).abs().float() / 10.0).unsqueeze(-1)
                features = torch.cat(
                    [
                        noisy_repr,
                        cand_repr,
                        (noisy_repr - cand_repr).abs(),
                        noisy_repr * cand_repr,
                        length_feature,
                    ],
                    dim=-1,
                )
                return self.head(features).squeeze(-1)
        self.module = _Model()


class TorchHybridSpellChecker:
    def __init__(
        self,
        metadata: TorchRerankerMetadata,
        artifact_dir: str | Path,
        vocabulary: list[str],
        frequencies: dict[str, int],
    ) -> None:
        self.metadata = metadata
        self.language = metadata.language
        self.artifact_dir = Path(artifact_dir)
        self.safety_policy = SafetyPolicy.from_mapping(metadata.safety_policy)
        self.lexical = LexicalSpellChecker(
            vocabulary=vocabulary,
            frequencies=frequencies,
            language=metadata.language,
            min_correction_length=metadata.min_correction_length,
            similarity_threshold=metadata.similarity_threshold,
            safety_policy=self.safety_policy,
        )
        self.vocabulary = vocabulary
        self.vocabulary_set = set(vocabulary)
        self.frequencies = frequencies
        self.language_profile = LanguageProfile.from_vocabulary(vocabulary)
        self.char_to_index = {char: index for index, char in enumerate(metadata.chars)}
        self.model = self._load_model()

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "TorchHybridSpellChecker":
        artifact_dir = Path(artifact_dir)
        metadata = TorchRerankerMetadata(**json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8")))
        vocab_payload = json.loads((artifact_dir / metadata.vocabulary_file).read_text(encoding="utf-8"))
        return cls(
            metadata=metadata,
            artifact_dir=artifact_dir,
            vocabulary=vocab_payload["vocabulary"],
            frequencies=vocab_payload.get("frequencies", {}),
        )

    @staticmethod
    def write_artifact(
        artifact_dir: str | Path,
        *,
        language: str,
        max_length: int,
        chars: list[str],
        embedding_dim: int,
        hidden_size: int,
        candidate_limit: int,
        min_correction_length: int,
        similarity_threshold: float,
        score_threshold: float,
        weights_file: str,
        vocabulary_file: str,
        vocabulary: list[str],
        frequencies: dict[str, int],
        safety_policy: SafetyPolicy | None = None,
    ) -> Path:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metadata = TorchRerankerMetadata(
            model_type="torch-hybrid-reranker",
            language=language,
            max_length=max_length,
            chars=chars,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            candidate_limit=candidate_limit,
            min_correction_length=min_correction_length,
            similarity_threshold=similarity_threshold,
            score_threshold=score_threshold,
            weights_file=weights_file,
            vocabulary_file=vocabulary_file,
            safety_policy=(safety_policy or SafetyPolicy()).to_dict(),
        )
        (artifact_dir / "metadata.json").write_text(
            json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (artifact_dir / vocabulary_file).write_text(
            json.dumps({"vocabulary": vocabulary, "frequencies": frequencies}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ArtifactManifest(
            artifact_version="2",
            backend="torch-hybrid-reranker",
            language=language,
            entrypoint="eib_spellchecker.modeling.torch_reranker:TorchHybridSpellChecker",
            payload_file=weights_file,
            metadata_file="metadata.json",
        ).write(artifact_dir)
        return artifact_dir

    def _load_model(self):
        torch, _, _, _ = require_torch()
        model = CharPairRerankerModelBase(
            embedding_dim=self.metadata.embedding_dim,
            hidden_size=self.metadata.hidden_size,
            vocab_size=len(self.metadata.chars),
        ).module
        state = torch.load(self.artifact_dir / self.metadata.weights_file, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    def _encode_word(self, word: str) -> tuple[list[int], int]:
        normalized = normalize_text(clean_token(word), lowercase=True, strip_accents_flag=False)
        trimmed = normalized[: self.metadata.max_length]
        ids = [self.char_to_index.get(char, self.char_to_index[UNK]) for char in trimmed]
        length = max(len(ids), 1)
        ids.extend([self.char_to_index[PAD]] * (self.metadata.max_length - len(ids)))
        return ids, length

    def _score_candidates(self, word: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        torch, _, _, _ = require_torch()
        noisy_ids, noisy_length = self._encode_word(word)
        noisy_batch = torch.tensor([noisy_ids for _ in candidates], dtype=torch.long)
        noisy_lengths = torch.tensor([noisy_length for _ in candidates], dtype=torch.long)
        cand_encoded = [self._encode_word(candidate) for candidate in candidates]
        cand_batch = torch.tensor([ids for ids, _ in cand_encoded], dtype=torch.long)
        cand_lengths = torch.tensor([length for _, length in cand_encoded], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(noisy_batch, noisy_lengths, cand_batch, cand_lengths)
            scores = torch.sigmoid(logits).cpu().tolist()
        return [float(score) for score in scores]

    def _candidate_list(self, normalized: str, limit: int) -> list[str]:
        lexical_candidates = self.lexical.suggest(normalized, limit=max(limit, self.metadata.candidate_limit))
        if normalized not in lexical_candidates:
            lexical_candidates.insert(0, normalized)
        return list(dict.fromkeys(lexical_candidates))

    def suggest(self, word: str, limit: int = 5) -> list[str]:
        cleaned = clean_token(word)
        if not cleaned:
            return []
        normalized = normalize_text(cleaned, lowercase=True, strip_accents_flag=False)
        candidates = self._candidate_list(normalized, limit=max(limit, self.metadata.candidate_limit))
        if not candidates:
            return []
        scores = self._score_candidates(normalized, candidates)
        ranked = sorted(
            zip(candidates, scores, strict=False),
            key=lambda item: (-item[1], self.lexical._levenshtein(normalized, item[0]), -self.frequencies.get(item[0], 0)),
        )
        ranked = [candidate for candidate, _score in ranked if candidate != normalized]
        return ranked[:limit]

    def correct_token(self, word: str) -> TokenCorrection:
        cleaned = clean_token(word)
        if not cleaned:
            return TokenCorrection(original=word, corrected=word, changed=False, confidence=1.0, reason="empty-token")
        normalized = normalize_text(cleaned, lowercase=True, strip_accents_flag=False)
        if len(normalized) < self.metadata.min_correction_length:
            return TokenCorrection(original=word, corrected=word, changed=False, confidence=1.0, reason="too-short")
        if normalized in self.vocabulary_set:
            return TokenCorrection(original=word, corrected=preserve_case(word, normalized), changed=False, confidence=1.0, reason="in-vocabulary")
        candidates = self._candidate_list(normalized, self.metadata.candidate_limit)
        if not candidates:
            return TokenCorrection(original=word, corrected=word, changed=False, confidence=0.0, reason="no-candidates")
        scores = self._score_candidates(normalized, candidates)
        ranked = sorted(zip(candidates, scores, strict=False), key=lambda item: (-item[1], item[0]))
        best_candidate, best_score = ranked[0]
        identity_score = next((score for candidate, score in ranked if candidate == normalized), 0.0)
        profile = analyze_token(word, self.language_profile, self.safety_policy)
        risk_profile = profile.proper_like or profile.loanword_like or profile.out_of_domain_like
        alt_forced = False
        if best_candidate == normalized and len(ranked) > 1 and not risk_profile:
            alt_candidate, alt_score = ranked[1]
            lexical_similarity = 1.0 - (self.lexical._levenshtein(normalized, alt_candidate) / max(len(normalized), len(alt_candidate), 1))
            if alt_score >= identity_score - 0.08 and lexical_similarity >= self.metadata.similarity_threshold:
                best_candidate, best_score = alt_candidate, alt_score
                if alt_score >= self.metadata.score_threshold:
                    alt_forced = True
        if alt_forced:
            corrected_text = preserve_case(word, best_candidate)
            return TokenCorrection(
                original=word,
                corrected=corrected_text,
                changed=corrected_text != word,
                confidence=best_score,
                reason="reranked-alt",
            )
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
        if corrected_text == word and reason == "best-candidate" and best_score < self.metadata.score_threshold:
            fallback = self.lexical.correct_token(word)
            return TokenCorrection(
                original=word,
                corrected=fallback.corrected,
                changed=fallback.corrected != word,
                confidence=best_score,
                reason="fallback-lexical" if fallback.corrected != word else "below-threshold",
            )
        if corrected_text != word:
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



def build_char_vocab(tokens: Iterable[str]) -> list[str]:
    chars = {char for token in tokens for char in token}
    return [PAD, UNK] + sorted(chars)
