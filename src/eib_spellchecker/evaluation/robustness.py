from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from eib_spellchecker.data.pairs import load_pairs_auto
from eib_spellchecker.evaluation.metrics import _aligned_token_matches, benchmark_artifact, char_error_rate
from eib_spellchecker.inference.service import ArtifactSpellChecker
from eib_spellchecker.modeling.policy import SafetyPolicy, analyze_token
from eib_spellchecker.utils.text import tokenize_words


@dataclass
class CleanCorpusResult:
    total_lines: int
    total_tokens: int
    changed_lines: int
    changed_tokens: int
    unchanged_token_rate: float
    changed_token_rate: float
    examples: list[dict]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class OpenVocabBucket:
    name: str
    total: int
    exact_match_accuracy: float
    token_accuracy: float
    cer_improvement: float


@dataclass
class OpenVocabResult:
    summary: dict
    buckets: list[dict]
    examples: list[dict]

    def to_dict(self) -> dict:
        return {"summary": self.summary, "buckets": self.buckets, "examples": self.examples}



def _iter_text_lines(path: str | Path, text_column: str = "sentence"):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".tsv"}:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                yield line.strip()
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV sin cabecera: {path}")
            if text_column not in reader.fieldnames:
                first_col = reader.fieldnames[0]
                text_column = first_col
            for row in reader:
                text = (row.get(text_column) or "").strip()
                if text:
                    yield text
        return
    raise ValueError(f"Formato no soportado para clean benchmark: {path}")



def benchmark_clean_corpus(
    artifact_dir: str | Path,
    dataset: str | Path,
    *,
    text_column: str = "sentence",
    limit: int | None = None,
) -> CleanCorpusResult:
    checker = ArtifactSpellChecker.from_artifact_dir(artifact_dir)
    lines = list(_iter_text_lines(dataset, text_column=text_column))
    if limit is not None:
        lines = lines[:limit]
    total_lines = len(lines)
    total_tokens = 0
    changed_lines = 0
    changed_tokens = 0
    examples: list[dict] = []
    for line in lines:
        corrected, details = checker.correct_text(line)
        line_changed = any(detail.changed for detail in details)
        if line_changed:
            changed_lines += 1
        total_tokens += len(details)
        changed_tokens += sum(int(detail.changed) for detail in details)
        if len(examples) < 20 and line_changed:
            examples.append({
                "original": line,
                "corrected": corrected,
                "changes": [detail.__dict__ for detail in details if detail.changed],
            })
    changed_token_rate = (changed_tokens / total_tokens) if total_tokens else 0.0
    return CleanCorpusResult(
        total_lines=total_lines,
        total_tokens=total_tokens,
        changed_lines=changed_lines,
        changed_tokens=changed_tokens,
        unchanged_token_rate=1.0 - changed_token_rate,
        changed_token_rate=changed_token_rate,
        examples=examples,
    )



def benchmark_sentence_variants(
    artifact_dir: str | Path,
    dataset: str | Path,
    *,
    sentence_column: str = "sentence",
    error_prefix: str = "error_",
    limit: int | None = None,
):
    path = Path(dataset)
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV sin cabecera: {path}")
        error_columns = [name for name in reader.fieldnames if name.startswith(error_prefix)]
        if sentence_column not in reader.fieldnames or not error_columns:
            raise ValueError(f"El CSV debe contener {sentence_column!r} y columnas {error_prefix}*")
        for row in reader:
            gold = (row.get(sentence_column) or "").strip()
            if not gold:
                continue
            for error_column in error_columns:
                noisy = (row.get(error_column) or "").strip()
                if noisy:
                    pairs.append((noisy, gold))
                    if limit is not None and len(pairs) >= limit:
                        break
            if limit is not None and len(pairs) >= limit:
                break
    temp = path.parent / f".__tmp_variants_{path.stem}.tsv"
    temp.write_text("\n".join(f"{noisy}\t{gold}" for noisy, gold in pairs) + ("\n" if pairs else ""), encoding="utf-8")
    try:
        return benchmark_artifact(artifact_dir, temp, limit=limit)
    finally:
        if temp.exists():
            temp.unlink()



def benchmark_open_vocab(
    artifact_dir: str | Path,
    dataset: str | Path,
    *,
    noisy_column: str = "Input",
    gold_column: str = "Output",
    limit: int | None = None,
) -> OpenVocabResult:
    checker = ArtifactSpellChecker.from_artifact_dir(artifact_dir)
    backend = checker.backend
    vocabulary_set = getattr(backend, "vocabulary_set", set())
    profile = getattr(backend, "language_profile", None)
    policy = getattr(backend, "safety_policy", SafetyPolicy())
    if profile is None:
        from eib_spellchecker.modeling.policy import LanguageProfile
        profile = LanguageProfile.from_vocabulary(vocabulary_set)

    pairs = load_pairs_auto(dataset, noisy_column=noisy_column, gold_column=gold_column)
    if limit is not None:
        pairs = pairs[:limit]

    bucket_payload: dict[str, dict] = {
        "all": {"total": 0, "exact": 0, "token_correct": 0, "token_total": 0, "before": 0.0, "after": 0.0},
        "seen": {"total": 0, "exact": 0, "token_correct": 0, "token_total": 0, "before": 0.0, "after": 0.0},
        "unseen": {"total": 0, "exact": 0, "token_correct": 0, "token_total": 0, "before": 0.0, "after": 0.0},
        "proper_like": {"total": 0, "exact": 0, "token_correct": 0, "token_total": 0, "before": 0.0, "after": 0.0},
        "loanword_like": {"total": 0, "exact": 0, "token_correct": 0, "token_total": 0, "before": 0.0, "after": 0.0},
        "out_of_domain_like": {"total": 0, "exact": 0, "token_correct": 0, "token_total": 0, "before": 0.0, "after": 0.0},
    }
    examples: list[dict] = []
    for noisy, gold in pairs:
        predicted, _details = checker.correct_text(noisy)
        token_profile = analyze_token(noisy, profile, policy)
        bucket_names = ["all"]
        bucket_names.append("seen" if gold in vocabulary_set else "unseen")
        if token_profile.proper_like:
            bucket_names.append("proper_like")
        if token_profile.loanword_like:
            bucket_names.append("loanword_like")
        if token_profile.out_of_domain_like:
            bucket_names.append("out_of_domain_like")
        aligned_correct, aligned_total = _aligned_token_matches(predicted, gold)
        before = char_error_rate(noisy, gold)
        after = char_error_rate(predicted, gold)
        for bucket_name in bucket_names:
            bucket = bucket_payload[bucket_name]
            bucket["total"] += 1
            bucket["exact"] += int(predicted == gold)
            bucket["token_correct"] += aligned_correct
            bucket["token_total"] += aligned_total
            bucket["before"] += before
            bucket["after"] += after
        if len(examples) < 20:
            examples.append({
                "noisy": noisy,
                "gold": gold,
                "predicted": predicted,
                "gold_seen": gold in vocabulary_set,
                "proper_like": token_profile.proper_like,
                "loanword_like": token_profile.loanword_like,
                "out_of_domain_like": token_profile.out_of_domain_like,
            })

    buckets: list[dict] = []
    for name, bucket in bucket_payload.items():
        total = bucket["total"]
        if total == 0:
            continue
        cer_before = bucket["before"] / total
        cer_after = bucket["after"] / total
        buckets.append({
            "name": name,
            "total": total,
            "exact_match_accuracy": bucket["exact"] / total,
            "token_accuracy": (bucket["token_correct"] / bucket["token_total"]) if bucket["token_total"] else 0.0,
            "cer_improvement": ((cer_before - cer_after) / cer_before) if cer_before else 0.0,
        })
    summary = {
        "total_examples": len(pairs),
        "num_buckets": len(buckets),
        "artifact_backend": checker.describe()["backend"],
    }
    return OpenVocabResult(summary=summary, buckets=buckets, examples=examples)
