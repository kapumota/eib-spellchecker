from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

LANGUAGE_NAMES = {
    "ash": "Asháninka",
    "shi": "Shipibo-Konibo",
    "ya": "Yanesha'",
    "yi": "Yine",
}


@dataclass
class CorpusEntry:
    code: str
    language: str
    lines: int
    tokens: int
    path: str


@dataclass
class PairedDatasetEntry:
    language_code: str
    language: str
    variant: str
    rows: int
    path: str


@dataclass
class ModelEntry:
    language_code: str | None
    language: str | None
    family: str
    variant: str
    size_hint: str | None
    path: str


@dataclass
class LegacyInventory:
    corpora: list[CorpusEntry]
    paired_datasets: list[PairedDatasetEntry]
    models: list[ModelEntry]

    def to_dict(self) -> dict:
        return {
            "corpora": [asdict(x) for x in self.corpora],
            "paired_datasets": [asdict(x) for x in self.paired_datasets],
            "models": [asdict(x) for x in self.models],
            "summary": {
                "num_corpora": len(self.corpora),
                "num_paired_datasets": len(self.paired_datasets),
                "num_models": len(self.models),
                "languages": sorted({x.language for x in self.corpora} | {x.language for x in self.paired_datasets if x.language}),
            },
        }


def _guess_language_code(name: str) -> str | None:
    match = re.search(r'_(ash|shi|ya|yi)(?:\.|_)', name)
    if match:
        return match.group(1)
    if '/asha/' in name:
        return 'ash'
    if '/yane/' in name:
        return 'ya'
    if 'Shipibo' in name or '_shi' in name:
        return 'shi'
    if 'Yine' in name or '_yi' in name:
        return 'yi'
    return None


def inventory_from_zips(
    dataset_zip: str | Path | None = None,
    augmentation_zip: str | Path | None = None,
    subword_zip: str | Path | None = None,
) -> LegacyInventory:
    corpora: list[CorpusEntry] = []
    paired_datasets: list[PairedDatasetEntry] = []
    models: list[ModelEntry] = []

    if dataset_zip:
        with zipfile.ZipFile(dataset_zip) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith('.txt') and '/dataset/' in f'/{n}'):
                code = Path(name).stem
                text = zf.read(name).decode('utf-8', errors='ignore')
                lines = [line for line in text.splitlines() if line.strip()]
                token_count = sum(len(line.split()) for line in lines)
                corpora.append(CorpusEntry(
                    code=code,
                    language=LANGUAGE_NAMES.get(code, code),
                    lines=len(lines),
                    tokens=token_count,
                    path=name,
                ))

    if augmentation_zip:
        with zipfile.ZipFile(augmentation_zip) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith('.csv') and '_data_' in n and 'seq2seq+atencion' in n):
                code = _guess_language_code(name) or 'unknown'
                variant = Path(name).stem.split('_data_')[0]
                rows = _count_csv_rows(zf, name)
                paired_datasets.append(PairedDatasetEntry(
                    language_code=code,
                    language=LANGUAGE_NAMES.get(code, code),
                    variant=variant,
                    rows=rows,
                    path=name,
                ))
            for name in sorted(n for n in zf.namelist() if n.endswith('.h5')):
                code = _guess_language_code(name)
                family = 'seq2seq-attention'
                variant = _variant_from_model_name(Path(name).name)
                size_hint = _size_hint_from_name(Path(name).name)
                models.append(ModelEntry(
                    language_code=code,
                    language=LANGUAGE_NAMES.get(code, code) if code else None,
                    family=family,
                    variant=variant,
                    size_hint=size_hint,
                    path=name,
                ))

    if subword_zip:
        with zipfile.ZipFile(subword_zip) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith('.h5')):
                code = _guess_language_code(name)
                family = 'subword-bpe' if '/bpe/' in name else 'subword-syllable'
                variant = _variant_from_model_name(Path(name).name)
                size_hint = _size_hint_from_name(Path(name).name)
                models.append(ModelEntry(
                    language_code=code,
                    language=LANGUAGE_NAMES.get(code, code) if code else None,
                    family=family,
                    variant=variant,
                    size_hint=size_hint,
                    path=name,
                ))
    return LegacyInventory(corpora=corpora, paired_datasets=paired_datasets, models=models)


def _variant_from_model_name(filename: str) -> str:
    stem = Path(filename).stem
    if 'remix_common' in stem:
        return 'remix_common'
    if 'remix_keyboard' in stem:
        return 'remix_keyboard'
    if 'remix_syllable' in stem:
        return 'remix_syllable'
    if 'bpe' in stem:
        match = re.search(r'bpe_(.+?)_kd$', stem)
        return f'bpe_{match.group(1)}' if match else 'bpe'
    if 'syllable' in stem:
        return 'syllable'
    if '_kd' in stem:
        parts = stem.split('_')
        return parts[-2] if len(parts) >= 2 else stem
    return stem


def _size_hint_from_name(filename: str) -> str | None:
    for hint in ['2.5k', '5k', '7.5k', '10k', '46']:
        if hint in filename:
            return hint
    return None


def _count_csv_rows(zf: zipfile.ZipFile, name: str) -> int:
    with zf.open(name, 'r') as handle:
        text = io.TextIOWrapper(handle, encoding='utf-8', errors='ignore', newline='')
        reader = csv.reader(text)
        next(reader, None)
        return sum(1 for _ in reader)
