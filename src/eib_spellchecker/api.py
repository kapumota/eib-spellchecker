# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from eib_spellchecker.evaluation.metrics import benchmark_artifact
from eib_spellchecker.inference.service import ArtifactSpellChecker


class CorrectRequest(BaseModel):
    text: str


class SuggestRequest(BaseModel):
    word: str
    limit: int = 5


class BenchmarkPreviewRequest(BaseModel):
    pairs: list[tuple[str, str]]


@lru_cache(maxsize=1)
def get_checker() -> ArtifactSpellChecker:
    artifact_dir = os.getenv("EIB_ARTIFACT_DIR")
    if not artifact_dir:
        raise RuntimeError("Define la variable EIB_ARTIFACT_DIR antes de iniciar la API.")
    return ArtifactSpellChecker.from_artifact_dir(artifact_dir)


app = FastAPI(title="eib-spellchecker", version="0.6.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    try:
        return get_checker().describe()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/correct")
def correct(request: CorrectRequest) -> dict:
    try:
        corrected, details = get_checker().correct_text(request.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "original": request.text,
        "corrected": corrected,
        "tokens": [detail.__dict__ for detail in details],
    }


@app.post("/suggest")
def suggest(request: SuggestRequest) -> dict:
    try:
        suggestions = get_checker().suggest(request.word, limit=request.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"word": request.word, "suggestions": suggestions}


@app.post("/benchmark-preview")
def benchmark_preview(request: BenchmarkPreviewRequest) -> dict:
    checker = get_checker()
    exact = 0
    rows = []
    for noisy, gold in request.pairs[:20]:
        predicted, _ = checker.correct_text(noisy)
        exact += int(predicted == gold)
        rows.append({"noisy": noisy, "gold": gold, "predicted": predicted, "exact_match": predicted == gold})
    total = len(rows)
    return {"preview_examples": total, "preview_exact_match_accuracy": exact / total if total else 0.0, "rows": rows}
