from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

LANGUAGE_NAMES = {
    "ash": "Asháninka",
    "shi": "Shipibo-Konibo",
    "ya": "Yanesha'",
    "yi": "Yine",
}


@dataclass
class PairedDatasetEntry:
    language_code: str
    language: str
    variant: str
    rows: int
    path: str


@dataclass
class SentenceVariantEntry:
    language_code: str
    language: str
    rows: int
    error_columns: list[str]
    path: str


@dataclass
class SentenceCorpusEntry:
    language_code: str
    language: str
    rows: int
    path: str


@dataclass
class HistoricalScoreEntry:
    metric: str
    language: str
    language_code: str
    system: str
    value: float


@dataclass
class ExcelsInventory:
    paired_datasets: list[PairedDatasetEntry]
    sentence_variant_datasets: list[SentenceVariantEntry]
    sentence_corpora: list[SentenceCorpusEntry]
    historical_scores: list[HistoricalScoreEntry]

    def to_dict(self) -> dict:
        payload = {
            "paired_datasets": [asdict(x) for x in self.paired_datasets],
            "sentence_variant_datasets": [asdict(x) for x in self.sentence_variant_datasets],
            "sentence_corpora": [asdict(x) for x in self.sentence_corpora],
            "historical_scores": [asdict(x) for x in self.historical_scores],
        }
        payload["summary"] = {
            "num_paired_datasets": len(self.paired_datasets),
            "num_sentence_variant_datasets": len(self.sentence_variant_datasets),
            "num_sentence_corpora": len(self.sentence_corpora),
            "num_historical_score_entries": len(self.historical_scores),
            "total_paired_rows": sum(x.rows for x in self.paired_datasets),
            "total_sentence_rows": sum(x.rows for x in self.sentence_variant_datasets),
            "languages": sorted({x.language for x in self.paired_datasets} | {x.language for x in self.sentence_variant_datasets} | {x.language for x in self.sentence_corpora}),
        }
        return payload


def _guess_language_code(name: str) -> str | None:
    match = re.search(r'_(ash|shi|ya|yi)(?:\.|_)', name)
    if match:
        return match.group(1)
    lowered = name.lower()
    if 'ashaninka' in lowered:
        return 'ash'
    if 'shipibo' in lowered or 'sk' in lowered:
        return 'shi'
    if 'yanesha' in lowered or 'yane' in lowered:
        return 'ya'
    if 'yine' in lowered:
        return 'yi'
    return None


def inventory_from_zip(excels_zip: str | Path) -> ExcelsInventory:
    paired_datasets: list[PairedDatasetEntry] = []
    sentence_variant_datasets: list[SentenceVariantEntry] = []
    sentence_corpora: list[SentenceCorpusEntry] = []
    historical_scores: list[HistoricalScoreEntry] = []

    with zipfile.ZipFile(excels_zip) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith('.csv') and '_data_' in name:
                code = _guess_language_code(name) or 'unknown'
                variant = Path(name).stem.split('_data_')[0]
                rows, _ = _csv_rows_and_fields(zf, name)
                paired_datasets.append(PairedDatasetEntry(code, LANGUAGE_NAMES.get(code, code), variant, rows, name))
            elif name.endswith('.csv') and re.search(r'/df_(ash|shi|ya|yi)\.csv$', name):
                code = _guess_language_code(name) or 'unknown'
                rows, fields = _csv_rows_and_fields(zf, name)
                sentence_variant_datasets.append(SentenceVariantEntry(
                    code,
                    LANGUAGE_NAMES.get(code, code),
                    rows,
                    [field for field in fields if field.startswith('error_')],
                    name,
                ))
            elif name.endswith('.csv') and re.search(r'/df_(ash|shi|ya|yi)_sentences\.csv$', name):
                code = _guess_language_code(name) or 'unknown'
                rows, _ = _csv_rows_and_fields(zf, name)
                sentence_corpora.append(SentenceCorpusEntry(code, LANGUAGE_NAMES.get(code, code), rows, name))
            elif name.endswith('Score.txt'):
                text = zf.read(name).decode('utf-8', errors='ignore')
                historical_scores.extend(parse_score_text(text))

    return ExcelsInventory(paired_datasets, sentence_variant_datasets, sentence_corpora, historical_scores)


def parse_score_file(path: str | Path) -> list[HistoricalScoreEntry]:
    return parse_score_text(Path(path).read_text(encoding='utf-8', errors='ignore'))


def parse_score_text(text: str) -> list[HistoricalScoreEntry]:
    entries: list[HistoricalScoreEntry] = []
    current_metric: str | None = None
    current_language_code: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {'CharacTER', 'BLEU', 'Accuracy'}:
            current_metric = line
            current_language_code = None
            continue

        header_match = re.fullmatch(r'(Yine|Shipibo-konibo|Ashaninka|Yanesha)(46)', line, flags=re.IGNORECASE)
        if header_match:
            current_language_code = {
                'yine': 'yi',
                'shipibo-konibo': 'shi',
                'ashaninka': 'ash',
                'yanesha': 'ya',
            }[header_match.group(1).lower()]
            continue

        assign_match = re.match(r'(.+?)\s*=\s*([0-9][0-9 .]*)$', line)
        if assign_match and current_metric and current_language_code:
            value = _parse_numeric(assign_match.group(2))
            if value is not None:
                entries.append(HistoricalScoreEntry(
                    metric=current_metric,
                    language=LANGUAGE_NAMES[current_language_code],
                    language_code=current_language_code,
                    system=assign_match.group(1).strip(),
                    value=value,
                ))
            continue

        baseline_match = re.match(r'([0-9][0-9 .]*)\s+([^=]+?)$', line)
        if baseline_match and current_metric:
            value = _parse_numeric(baseline_match.group(1))
            remainder = baseline_match.group(2).strip()
            if value is None:
                continue
            remainder = remainder.replace('---', '--').replace('mejor', '').strip()
            remainder = remainder.lstrip('-').strip()
            code = _guess_language_code(remainder) or current_language_code
            if code is None:
                continue
            entries.append(HistoricalScoreEntry(
                metric=current_metric,
                language=LANGUAGE_NAMES.get(code, code),
                language_code=code,
                system=remainder,
                value=value,
            ))
            continue

    return entries


def summarize_scores(entries: list[HistoricalScoreEntry]) -> dict:
    grouped: dict[str, dict[str, list[HistoricalScoreEntry]]] = {}
    for entry in entries:
        grouped.setdefault(entry.metric, {}).setdefault(entry.language_code, []).append(entry)

    best_by_metric_language = {}
    for metric, per_lang in grouped.items():
        best_by_metric_language[metric] = {}
        for code, items in per_lang.items():
            best = min(items, key=lambda x: x.value) if metric == 'CharacTER' else max(items, key=lambda x: x.value)
            best_by_metric_language[metric][code] = asdict(best)

    return {
        'entries': [asdict(x) for x in entries],
        'summary': {
            'num_entries': len(entries),
            'metrics': sorted(grouped.keys()),
            'languages': sorted({entry.language for entry in entries}),
            'best_by_metric_language': best_by_metric_language,
        },
    }


def _parse_numeric(raw: str) -> float | None:
    cleaned = raw.replace(' ', '')
    if cleaned.endswith('.'):  # defensive
        cleaned = cleaned[:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _csv_rows_and_fields(zf: zipfile.ZipFile, name: str) -> tuple[int, list[str]]:
    with zf.open(name, 'r') as handle:
        text = io.TextIOWrapper(handle, encoding='utf-8', errors='ignore', newline='')
        reader = csv.DictReader(text)
        fieldnames = reader.fieldnames or []
        return sum(1 for _ in reader), fieldnames
