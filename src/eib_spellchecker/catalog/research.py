# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import h5py
except ModuleNotFoundError:  # optional dependency for legacy HDF5 inspection
    h5py = None

from eib_spellchecker.catalog.legacy import LANGUAGE_NAMES, _guess_language_code, _variant_from_model_name
from eib_spellchecker.evaluation.metrics import char_error_rate, _aligned_token_matches


@dataclass
class H5ModelSummary:
    keras_version: str | None
    backend: str | None
    input_length: int | None
    encoder_input_dim: int | None
    decoder_input_dim: int | None
    lstm_units: int | None
    output_dim: int | None
    layer_names: list[str]


@dataclass
class ResearchRun:
    slug: str
    source_zip: str
    language_code: str | None
    language: str | None
    family: str
    variant: str
    run_dir: str
    weights_path: str | None
    candidates_path: str | None
    reference_path: str | None
    test_input_path: str | None
    test_output_path: str | None
    vocab_path: str | None
    model_summary: H5ModelSummary | None


@dataclass
class ResearchBenchmark:
    slug: str
    language_code: str | None
    language: str | None
    family: str
    variant: str
    source_zip: str
    total_examples: int
    exact_match_accuracy: float
    token_accuracy: float
    avg_cer: float
    examples: list[dict]


@dataclass
class ResearchInventory:
    runs: list[ResearchRun]

    def to_dict(self) -> dict:
        payload = {
            'runs': [asdict(x) for x in self.runs],
        }
        payload['summary'] = {
            'num_runs': len(self.runs),
            'num_with_weights': sum(1 for x in self.runs if x.weights_path),
            'num_with_candidates_and_reference': sum(1 for x in self.runs if x.candidates_path and x.reference_path),
            'languages': sorted({x.language for x in self.runs if x.language}),
            'families': sorted({x.family for x in self.runs}),
        }
        return payload


def inspect_h5_model(blob: bytes) -> H5ModelSummary:
    if h5py is None:
        raise RuntimeError('h5py is required to inspect legacy .h5 model files')
    with h5py.File(io.BytesIO(blob), 'r') as handle:
        attrs = dict(handle.attrs)
        model_config = attrs.get('model_config')
        config = json.loads(model_config) if model_config else {}
        layers = config.get('config', {}).get('layers', []) if isinstance(config, dict) else []

        input_length = None
        encoder_input_dim = None
        decoder_input_dim = None
        lstm_units = None
        output_dim = None
        layer_names: list[str] = []

        for layer in layers:
            layer_names.append(layer.get('name', ''))
            class_name = layer.get('class_name')
            layer_config = layer.get('config', {})
            if class_name == 'InputLayer' and input_length is None:
                batch_shape = layer_config.get('batch_input_shape')
                if isinstance(batch_shape, list) and len(batch_shape) >= 2:
                    input_length = batch_shape[1]
            elif class_name == 'Embedding':
                if encoder_input_dim is None:
                    encoder_input_dim = layer_config.get('input_dim')
                elif decoder_input_dim is None:
                    decoder_input_dim = layer_config.get('input_dim')
            elif class_name == 'LSTM' and lstm_units is None:
                lstm_units = layer_config.get('units')
            elif class_name == 'TimeDistributed':
                inner = layer_config.get('layer', {}).get('config', {})
                if inner.get('activation') == 'softmax':
                    output_dim = inner.get('units')

        return H5ModelSummary(
            keras_version=str(attrs.get('keras_version')) if attrs.get('keras_version') is not None else None,
            backend=str(attrs.get('backend')) if attrs.get('backend') is not None else None,
            input_length=input_length,
            encoder_input_dim=encoder_input_dim,
            decoder_input_dim=decoder_input_dim,
            lstm_units=lstm_units,
            output_dim=output_dim,
            layer_names=layer_names,
        )


def _family_from_path(path: str) -> str:
    if '/bpe/' in path:
        return 'subword-bpe'
    if '/silabas/' in path:
        return 'subword-syllable'
    return 'seq2seq-attention'


def _variant_from_path(path: str) -> str:
    if '/bpe/' in path:
        match = re.search(r'/((?:2\.5|5|7\.5|10)k)/', path)
        if match:
            return f'bpe_{match.group(1)}'
    if '/silabas/' in path:
        return 'syllable'
    lowered = path.lower()
    for label in ['common', 'keyboard', 'syllable', 'remix_common', 'remix_keyboard', 'remix_syllable']:
        if label in lowered:
            return label
    return _variant_from_model_name(Path(path).name)


def _collect_runs_from_zip(zip_path: str | Path) -> list[ResearchRun]:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        run_dirs = sorted({str(Path(name).parent) for name in names if name.endswith('.h5') or name.endswith('candidatos.txt') or name.endswith('referencia.txt')})
        runs: list[ResearchRun] = []
        for run_dir in run_dirs:
            related = [name for name in names if str(Path(name).parent) == run_dir]
            weights_path = next((name for name in related if name.endswith('.h5')), None)
            candidates_path = next((name for name in related if name.endswith('candidatos.txt')), None)
            reference_path = next((name for name in related if name.endswith('referencia.txt')), None)
            test_input_path = next((name for name in related if re.search(r'(?:^|/)x_test(?:_|\.)', name)), None)
            test_output_path = next((name for name in related if re.search(r'(?:^|/)y_test(?:_|\.)', name)), None)
            vocab_path = next((name for name in related if name.endswith('.vocab')), None)
            probe = weights_path or candidates_path or reference_path or run_dir
            code = _guess_language_code(probe)
            language = LANGUAGE_NAMES.get(code, code) if code else None
            family = _family_from_path(probe)
            variant = _variant_from_path(probe)
            slug = re.sub(r'[^a-z0-9]+', '-', f"{zip_path.stem}-{family}-{code or 'unk'}-{variant}-{Path(run_dir).name}".lower()).strip('-')
            model_summary = None
            if weights_path:
                try:
                    model_summary = inspect_h5_model(zf.read(weights_path))
                except Exception:
                    model_summary = None
            runs.append(ResearchRun(
                slug=slug,
                source_zip=zip_path.name,
                language_code=code,
                language=language,
                family=family,
                variant=variant,
                run_dir=run_dir,
                weights_path=weights_path,
                candidates_path=candidates_path,
                reference_path=reference_path,
                test_input_path=test_input_path,
                test_output_path=test_output_path,
                vocab_path=vocab_path,
                model_summary=model_summary,
            ))
        return runs


def inventory_research_runs(*, augmentation_zip: str | Path | None = None, subword_zip: str | Path | None = None) -> ResearchInventory:
    runs: list[ResearchRun] = []
    for zip_path in [augmentation_zip, subword_zip]:
        if zip_path:
            runs.extend(_collect_runs_from_zip(zip_path))
    return ResearchInventory(runs=sorted(runs, key=lambda x: (x.family, x.language or '', x.variant, x.slug)))


def _read_lines(zf: zipfile.ZipFile, name: str) -> list[str]:
    return [line.rstrip('\n\r') for line in zf.read(name).decode('utf-8', errors='ignore').splitlines()]


def benchmark_research_runs(*, augmentation_zip: str | Path | None = None, subword_zip: str | Path | None = None) -> list[ResearchBenchmark]:
    inventory = inventory_research_runs(augmentation_zip=augmentation_zip, subword_zip=subword_zip)
    zip_map = {}
    for path in [augmentation_zip, subword_zip]:
        if path:
            zip_map[Path(path).name] = zipfile.ZipFile(path)

    benchmarks: list[ResearchBenchmark] = []
    try:
        for run in inventory.runs:
            if not (run.candidates_path and run.reference_path):
                continue
            zf = zip_map[run.source_zip]
            candidates = _read_lines(zf, run.candidates_path)
            references = _read_lines(zf, run.reference_path)
            total = min(len(candidates), len(references))
            if total == 0:
                continue
            exact = 0
            token_correct = 0
            token_total = 0
            cer_sum = 0.0
            examples = []
            for predicted, gold in zip(candidates[:total], references[:total]):
                exact += int(predicted == gold)
                aligned_correct, aligned_total = _aligned_token_matches(predicted, gold)
                token_correct += aligned_correct
                token_total += aligned_total
                cer_sum += char_error_rate(predicted, gold)
                if len(examples) < 5:
                    examples.append({'predicted': predicted, 'gold': gold, 'exact_match': predicted == gold})
            benchmarks.append(ResearchBenchmark(
                slug=run.slug,
                language_code=run.language_code,
                language=run.language,
                family=run.family,
                variant=run.variant,
                source_zip=run.source_zip,
                total_examples=total,
                exact_match_accuracy=exact / total,
                token_accuracy=(token_correct / token_total) if token_total else 0.0,
                avg_cer=cer_sum / total,
                examples=examples,
            ))
    finally:
        for zf in zip_map.values():
            zf.close()

    return sorted(benchmarks, key=lambda x: (x.language or '', x.family, x.variant, x.slug))


def summarize_research_benchmarks(benchmarks: list[ResearchBenchmark]) -> dict:
    payload = {
        'benchmarks': [asdict(x) for x in benchmarks],
        'summary': {
            'num_runs': len(benchmarks),
            'languages': sorted({x.language for x in benchmarks if x.language}),
        },
    }
    if benchmarks:
        payload['summary'].update({
            'avg_exact_match_accuracy': sum(x.exact_match_accuracy for x in benchmarks) / len(benchmarks),
            'avg_token_accuracy': sum(x.token_accuracy for x in benchmarks) / len(benchmarks),
            'avg_cer': sum(x.avg_cer for x in benchmarks) / len(benchmarks),
            'best_by_exact_match': asdict(max(benchmarks, key=lambda x: x.exact_match_accuracy)),
            'best_by_lowest_cer': asdict(min(benchmarks, key=lambda x: x.avg_cer)),
        })
        by_language = {}
        for benchmark in benchmarks:
            if not benchmark.language:
                continue
            current = by_language.get(benchmark.language)
            if current is None or benchmark.exact_match_accuracy > current.exact_match_accuracy:
                by_language[benchmark.language] = benchmark
        payload['summary']['best_by_language'] = {language: asdict(result) for language, result in sorted(by_language.items())}
    return payload


def benchmark_markdown(summary: dict) -> str:
    lines = [
        '# Research model benchmarks',
        '',
        f"Runs evaluados: {summary['summary'].get('num_runs', 0)}",
        '',
    ]
    best_by_language = summary['summary'].get('best_by_language', {})
    if best_by_language:
        lines.extend([
            '## Mejor corrida por idioma',
            '',
            '| Idioma | Familia | Variante | Exact match | Token acc. | CER | Ejemplos |',
            '|---|---|---:|---:|---:|---:|---:|',
        ])
        for language, item in best_by_language.items():
            lines.append(
                f"| {language} | {item['family']} | {item['variant']} | {item['exact_match_accuracy']:.4f} | {item['token_accuracy']:.4f} | {item['avg_cer']:.4f} | {item['total_examples']} |"
            )
        lines.append('')
    if summary['summary'].get('num_runs', 0):
        best = summary['summary']['best_by_exact_match']
        lines.extend([
            '## Mejor corrida global',
            '',
            f"- **{best['language']} / {best['family']} / {best['variant']}**",
            f"- Exact match: **{best['exact_match_accuracy']:.4f}**",
            f"- Token accuracy: **{best['token_accuracy']:.4f}**",
            f"- CER: **{best['avg_cer']:.4f}**",
            '',
        ])
    return '\n'.join(lines)
