from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from eib_spellchecker.utils.text import clean_token, normalize_text


DEFAULT_LOANWORD_CHARS = set("fwzqxvk")


@dataclass
class TokenProfile:
    normalized: str
    title_case: bool
    all_caps: bool
    proper_like: bool
    loanword_like: bool
    out_of_domain_like: bool
    known_char_coverage: float
    unknown_char_ratio: float


@dataclass
class SafetyPolicy:
    enabled: bool = True
    protect_title_case: bool = True
    protect_all_caps: bool = True
    proper_name_threshold: float = 0.86
    loanword_threshold: float = 0.92
    out_of_domain_threshold: float = 0.93
    abstain_margin: float = 0.08
    max_unknown_char_ratio: float = 0.35
    min_known_char_coverage: float = 0.45
    score_boost_for_identity: float = 0.03
    loanword_chars: str = "fwzqxvk"

    @classmethod
    def from_mapping(cls, payload: dict | None) -> "SafetyPolicy":
        if not payload:
            return cls()
        defaults = cls().__dict__
        merged = {**defaults, **payload}
        return cls(**merged)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class LanguageProfile:
    alphabet: set[str]
    ngrams: set[str]
    weighted_chars: set[str]

    @classmethod
    def from_vocabulary(cls, vocabulary: Iterable[str]) -> "LanguageProfile":
        alphabet: set[str] = set()
        ngrams: set[str] = set()
        weighted_chars: dict[str, int] = {}
        for token in vocabulary:
            normalized = normalize_text(clean_token(token), lowercase=True, strip_accents_flag=False)
            if not normalized:
                continue
            for ch in normalized:
                alphabet.add(ch)
                weighted_chars[ch] = weighted_chars.get(ch, 0) + 1
            padded = f"<{normalized}>"
            for n in (2, 3):
                for i in range(len(padded) - n + 1):
                    ngrams.add(padded[i : i + n])
        common = sorted(weighted_chars.items(), key=lambda item: (-item[1], item[0]))
        return cls(alphabet=alphabet, ngrams=ngrams, weighted_chars={c for c, _ in common[:24]})


@dataclass
class DecisionContext:
    original: str
    normalized: str
    best_candidate: str
    best_score: float
    identity_score: float
    margin: float
    profile: TokenProfile
    policy: SafetyPolicy



def char_ngrams(word: str, min_n: int = 2, max_n: int = 3) -> set[str]:
    normalized = normalize_text(clean_token(word), lowercase=True, strip_accents_flag=False)
    if not normalized:
        return set()
    padded = f"<{normalized}>"
    grams: set[str] = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(padded) - n + 1):
            grams.add(padded[i : i + n])
    return grams



def analyze_token(token: str, language_profile: LanguageProfile, policy: SafetyPolicy) -> TokenProfile:
    cleaned = clean_token(token)
    normalized = normalize_text(cleaned, lowercase=True, strip_accents_flag=False)
    if not normalized:
        return TokenProfile(
            normalized="",
            title_case=False,
            all_caps=False,
            proper_like=False,
            loanword_like=False,
            out_of_domain_like=False,
            known_char_coverage=1.0,
            unknown_char_ratio=0.0,
        )
    chars = list(normalized)
    known_chars = sum(1 for ch in chars if ch in language_profile.alphabet)
    known_char_coverage = known_chars / max(len(chars), 1)
    unknown_char_ratio = 1.0 - known_char_coverage
    grams = char_ngrams(normalized)
    known_ngrams = sum(1 for gram in grams if gram in language_profile.ngrams)
    ngram_coverage = known_ngrams / max(len(grams), 1)
    title_case = token.istitle()
    all_caps = token.isupper() and len(token) > 1
    proper_like = title_case or (all_caps and policy.protect_all_caps)
    loanword_chars = set(policy.loanword_chars or DEFAULT_LOANWORD_CHARS)
    loanword_hits = sum(1 for ch in chars if ch in loanword_chars and ch not in language_profile.weighted_chars)
    loanword_like = loanword_hits >= 1 or (unknown_char_ratio >= 0.2 and ngram_coverage < 0.35)
    out_of_domain_like = known_char_coverage < policy.min_known_char_coverage or ngram_coverage < 0.3
    return TokenProfile(
        normalized=normalized,
        title_case=title_case,
        all_caps=all_caps,
        proper_like=proper_like,
        loanword_like=loanword_like,
        out_of_domain_like=out_of_domain_like,
        known_char_coverage=ngram_coverage,
        unknown_char_ratio=unknown_char_ratio,
    )



def adjusted_identity_score(identity_score: float, profile: TokenProfile, policy: SafetyPolicy) -> float:
    score = identity_score
    if profile.proper_like and policy.protect_title_case:
        score += 0.06
    if profile.loanword_like:
        score += 0.04
    if profile.out_of_domain_like:
        score += 0.05
    if profile.unknown_char_ratio > policy.max_unknown_char_ratio:
        score += 0.03
    return min(score, 1.0)



def decide_action(context: DecisionContext) -> tuple[str, str, float]:
    policy = context.policy
    profile = context.profile
    if not policy.enabled:
        if context.best_candidate == context.normalized:
            return context.original, "identity-best", context.identity_score
        return context.best_candidate, "best-candidate", context.best_score

    safe_identity = adjusted_identity_score(context.identity_score, profile, policy)
    if context.best_candidate == context.normalized:
        return context.original, "identity-best", safe_identity
    if profile.proper_like and context.best_score < policy.proper_name_threshold:
        return context.original, "protected-proper-name", safe_identity
    if profile.loanword_like and context.best_score < policy.loanword_threshold:
        return context.original, "protected-loanword", safe_identity
    if profile.out_of_domain_like and context.best_score < policy.out_of_domain_threshold:
        return context.original, "protected-out-of-domain", safe_identity
    risk_profile = profile.proper_like or profile.loanword_like or profile.out_of_domain_like
    if context.best_score - safe_identity < policy.abstain_margin and (risk_profile or context.best_score < 0.72):
        return context.original, "abstained-low-margin", safe_identity
    return context.best_candidate, "best-candidate", context.best_score



def soft_frequency_score(freq: int) -> float:
    if freq <= 0:
        return 0.0
    return min(1.0, sqrt(freq) / 12.0)
