# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eib_spellchecker.data.pairs import load_pairs_auto
from eib_spellchecker.inference.service import ArtifactSpellChecker
from eib_spellchecker.utils.text import tokenize_words


@dataclass
class EvaluationResult:
    total: int
    correct: int
    accuracy: float
    details: list[dict]


@dataclass
class BenchmarkResult:
    total_examples: int
    exact_match_accuracy: float
    token_accuracy: float
    cer_before: float
    cer_after: float
    cer_improvement: float
    examples: list[dict]


def levenshtein_distance(left: str, right: str) -> int:
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


def char_error_rate(predicted: str, gold: str) -> float:
    return levenshtein_distance(predicted, gold) / max(len(gold), 1)


def _aligned_token_matches(predicted: str, gold: str) -> tuple[int, int]:
    pred_tokens = tokenize_words(predicted)
    gold_tokens = tokenize_words(gold)
    total = max(len(pred_tokens), len(gold_tokens))
    if total == 0:
        return 0, 0
    correct = 0
    for i in range(total):
        pred = pred_tokens[i] if i < len(pred_tokens) else None
        ref = gold_tokens[i] if i < len(gold_tokens) else None
        correct += int(pred == ref and pred is not None)
    return correct, total


def evaluate_artifact(artifact_dir: str | Path, dataset: str | Path) -> EvaluationResult:
    checker = ArtifactSpellChecker.from_artifact_dir(artifact_dir)
    pairs = load_pairs_auto(dataset)
    details = []
    correct = 0
    for noisy, gold in pairs:
        predicted = checker.correct_word(noisy)
        is_correct = predicted == gold
        correct += int(is_correct)
        details.append({
            "noisy": noisy,
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
        })
    total = len(pairs)
    return EvaluationResult(
        total=total,
        correct=correct,
        accuracy=(correct / total) if total else 0.0,
        details=details,
    )


def benchmark_artifact(
    artifact_dir: str | Path,
    dataset: str | Path,
    *,
    noisy_column: str = "Input",
    gold_column: str = "Output",
    limit: int | None = None,
) -> BenchmarkResult:
    checker = ArtifactSpellChecker.from_artifact_dir(artifact_dir)
    pairs = load_pairs_auto(dataset, noisy_column=noisy_column, gold_column=gold_column)
    if limit is not None:
        pairs = pairs[:limit]

    exact_matches = 0
    token_correct = 0
    token_total = 0
    cer_before_sum = 0.0
    cer_after_sum = 0.0
    examples = []

    for noisy, gold in pairs:
        predicted, details = checker.correct_text(noisy)
        exact_matches += int(predicted == gold)
        aligned_correct, aligned_total = _aligned_token_matches(predicted, gold)
        token_correct += aligned_correct
        token_total += aligned_total
        before = char_error_rate(noisy, gold)
        after = char_error_rate(predicted, gold)
        cer_before_sum += before
        cer_after_sum += after
        if len(examples) < 20:
            examples.append({
                "noisy": noisy,
                "gold": gold,
                "predicted": predicted,
                "changed_tokens": sum(int(detail.changed) for detail in details),
                "exact_match": predicted == gold,
                "cer_before": before,
                "cer_after": after,
            })

    total = len(pairs)
    cer_before = (cer_before_sum / total) if total else 0.0
    cer_after = (cer_after_sum / total) if total else 0.0
    improvement = ((cer_before - cer_after) / cer_before) if cer_before else 0.0
    return BenchmarkResult(
        total_examples=total,
        exact_match_accuracy=(exact_matches / total) if total else 0.0,
        token_accuracy=(token_correct / token_total) if token_total else 0.0,
        cer_before=cer_before,
        cer_after=cer_after,
        cer_improvement=improvement,
        examples=examples,
    )
