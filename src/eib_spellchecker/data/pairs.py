from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def load_pairs_tsv(path: str | Path) -> list[tuple[str, str]]:
    path = Path(path)
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        noisy, gold = line.split("\t", maxsplit=1)
        pairs.append((noisy.strip(), gold.strip()))
    return pairs


def load_pairs_csv(
    path: str | Path,
    noisy_column: str = "Input",
    gold_column: str = "Output",
) -> list[tuple[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"El CSV no tiene cabecera: {path}")
        if noisy_column not in reader.fieldnames or gold_column not in reader.fieldnames:
            raise ValueError(
                f"El CSV debe contener columnas {noisy_column!r} y {gold_column!r}. Encontradas: {reader.fieldnames}"
            )
        pairs = []
        for row in reader:
            noisy = (row.get(noisy_column) or "").strip()
            gold = (row.get(gold_column) or "").strip()
            if noisy or gold:
                pairs.append((noisy, gold))
        return pairs


def load_pairs_auto(
    path: str | Path,
    *,
    noisy_column: str = "Input",
    gold_column: str = "Output",
) -> list[tuple[str, str]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return load_pairs_tsv(path)
    if suffix == ".csv":
        return load_pairs_csv(path, noisy_column=noisy_column, gold_column=gold_column)
    raise ValueError(f"Formato de pares no soportado: {path}")


def write_pairs_tsv(path: str | Path, pairs: Iterable[tuple[str, str]]) -> None:
    path = Path(path)
    lines = [f"{noisy}\t{gold}" for noisy, gold in pairs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
