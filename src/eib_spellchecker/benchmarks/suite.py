from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from eib_spellchecker.catalog.excels import LANGUAGE_NAMES
from eib_spellchecker.evaluation.metrics import benchmark_artifact


@dataclass
class SuiteDatasetResult:
    language_code: str
    language: str
    variant: str
    dataset_path: str
    artifact_dir: str
    total_examples: int
    exact_match_accuracy: float
    token_accuracy: float
    cer_before: float
    cer_after: float
    cer_improvement: float


@dataclass
class SuiteResult:
    datasets: list[SuiteDatasetResult]
    summary: dict

    def to_dict(self) -> dict:
        return {
            'datasets': [asdict(x) for x in self.datasets],
            'summary': self.summary,
        }


def benchmark_suite(artifact_root: str | Path, datasets_root: str | Path, *, limit: int | None = None) -> SuiteResult:
    artifact_root = Path(artifact_root)
    datasets_root = Path(datasets_root)
    datasets: list[SuiteDatasetResult] = []
    for path in sorted(datasets_root.glob('*_data_*.csv')):
        match = re.search(r'(?P<variant>.+?)_data_(?P<code>ash|shi|ya|yi)\.csv$', path.name)
        if not match:
            continue
        variant = match.group('variant')
        code = match.group('code')
        artifact_dir = artifact_root / code
        if not artifact_dir.exists():
            continue
        result = benchmark_artifact(artifact_dir, path, limit=limit)
        datasets.append(SuiteDatasetResult(
            language_code=code,
            language=LANGUAGE_NAMES.get(code, code),
            variant=variant,
            dataset_path=path.name,
            artifact_dir=str(artifact_dir.relative_to(artifact_root.parent)) if artifact_root.parent in artifact_dir.parents else artifact_dir.name,
            total_examples=result.total_examples,
            exact_match_accuracy=result.exact_match_accuracy,
            token_accuracy=result.token_accuracy,
            cer_before=result.cer_before,
            cer_after=result.cer_after,
            cer_improvement=result.cer_improvement,
        ))

    if datasets:
        summary = {
            'num_datasets': len(datasets),
            'languages': sorted({x.language for x in datasets}),
            'avg_exact_match_accuracy': sum(x.exact_match_accuracy for x in datasets) / len(datasets),
            'avg_token_accuracy': sum(x.token_accuracy for x in datasets) / len(datasets),
            'avg_cer_improvement': sum(x.cer_improvement for x in datasets) / len(datasets),
            'best_dataset_by_exact_match': asdict(max(datasets, key=lambda x: x.exact_match_accuracy)),
            'best_dataset_by_cer_improvement': asdict(max(datasets, key=lambda x: x.cer_improvement)),
        }
    else:
        summary = {
            'num_datasets': 0,
            'languages': [],
            'avg_exact_match_accuracy': 0.0,
            'avg_token_accuracy': 0.0,
            'avg_cer_improvement': 0.0,
        }
    return SuiteResult(datasets=datasets, summary=summary)
