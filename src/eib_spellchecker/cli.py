# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eib_spellchecker.benchmarks.reporting import write_report
from eib_spellchecker.benchmarks.suite import benchmark_suite
from eib_spellchecker.catalog.excels import inventory_from_zip as inventory_excels_zip, parse_score_file, summarize_scores
from eib_spellchecker.catalog.legacy import inventory_from_zips
from eib_spellchecker.catalog.research import (
    benchmark_markdown,
    benchmark_research_runs,
    inventory_research_runs,
    summarize_research_benchmarks,
)
from eib_spellchecker.config import load_config
from eib_spellchecker.demo.gradio_app import launch_demo
from eib_spellchecker.evaluation.metrics import benchmark_artifact, evaluate_artifact
from eib_spellchecker.evaluation.robustness import benchmark_clean_corpus, benchmark_open_vocab, benchmark_sentence_variants
from eib_spellchecker.inference.service import ArtifactSpellChecker
from eib_spellchecker.modeling.manifest import load_manifest
from eib_spellchecker.research.packaging import extract_research_runs
from eib_spellchecker.training.lexical import train_lexical_model
from eib_spellchecker.training.seq2seq import package_legacy_seq2seq, train_seq2seq_model
from eib_spellchecker.training.subword import train_subword_model
from eib_spellchecker.training.torch_reranker import train_torch_reranker_model



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eib-spellchecker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrena el backend léxico.")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output-dir", required=True)

    train_seq2seq_parser = subparsers.add_parser("train-seq2seq", help="Entrena el backend seq2seq opcional.")
    train_seq2seq_parser.add_argument("--config", required=True)
    train_seq2seq_parser.add_argument("--output-dir", required=True)

    train_torch_parser = subparsers.add_parser("train-torch-reranker", help="Entrena el backend moderno PyTorch híbrido.")
    train_torch_parser.add_argument("--config", required=True)
    train_torch_parser.add_argument("--pairs", nargs="+", required=True)
    train_torch_parser.add_argument("--output-dir", required=True)
    train_torch_parser.add_argument("--noisy-column", default="Input")
    train_torch_parser.add_argument("--gold-column", default="Output")
    train_torch_parser.add_argument("--limit", type=int, default=None)

    train_subword_parser = subparsers.add_parser("train-subword", help="Entrena el backend subword para vocabulario abierto.")
    train_subword_parser.add_argument("--config", required=True)
    train_subword_parser.add_argument("--output-dir", required=True)

    package_parser = subparsers.add_parser("package-legacy-seq2seq", help="Empaqueta pesos existentes del modelo legado.")
    package_parser.add_argument("--config", required=True)
    package_parser.add_argument("--weights", required=True)
    package_parser.add_argument("--output-dir", required=True)

    correct_parser = subparsers.add_parser("correct", help="Corrige texto usando un artefacto.")
    correct_parser.add_argument("--artifact-dir", required=True)
    correct_parser.add_argument("--text", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evalúa pares TSV o CSV palabra-a-palabra.")
    evaluate_parser.add_argument("--artifact-dir", required=True)
    evaluate_parser.add_argument("--dataset", required=True)

    benchmark_parser = subparsers.add_parser("benchmark-csv", help="Ejecuta benchmark de oraciones sobre CSV Input,Output o TSV.")
    benchmark_parser.add_argument("--artifact-dir", required=True)
    benchmark_parser.add_argument("--dataset", required=True)
    benchmark_parser.add_argument("--noisy-column", default="Input")
    benchmark_parser.add_argument("--gold-column", default="Output")
    benchmark_parser.add_argument("--limit", type=int, default=None)
    benchmark_parser.add_argument("--output", default=None)

    suite_parser = subparsers.add_parser("benchmark-suite", help="Ejecuta benchmark agregado sobre múltiples CSV *_data_<lang>.csv.")
    suite_parser.add_argument("--artifact-root", required=True)
    suite_parser.add_argument("--datasets-root", required=True)
    suite_parser.add_argument("--limit", type=int, default=None)
    suite_parser.add_argument("--output", default=None)

    open_vocab_parser = subparsers.add_parser("benchmark-open-vocab", help="Separa métricas por seen/unseen, nombres propios y préstamos probables.")
    open_vocab_parser.add_argument("--artifact-dir", required=True)
    open_vocab_parser.add_argument("--dataset", required=True)
    open_vocab_parser.add_argument("--noisy-column", default="Input")
    open_vocab_parser.add_argument("--gold-column", default="Output")
    open_vocab_parser.add_argument("--limit", type=int, default=None)
    open_vocab_parser.add_argument("--output", default=None)

    clean_parser = subparsers.add_parser("benchmark-clean", help="Mide sobrecorrección en corpus limpio o CSV de oraciones.")
    clean_parser.add_argument("--artifact-dir", required=True)
    clean_parser.add_argument("--dataset", required=True)
    clean_parser.add_argument("--text-column", default="sentence")
    clean_parser.add_argument("--limit", type=int, default=None)
    clean_parser.add_argument("--output", default=None)

    sentence_parser = subparsers.add_parser("benchmark-sentence-variants", help="Evalúa CSV con sentence + error_0..error_n.")
    sentence_parser.add_argument("--artifact-dir", required=True)
    sentence_parser.add_argument("--dataset", required=True)
    sentence_parser.add_argument("--sentence-column", default="sentence")
    sentence_parser.add_argument("--error-prefix", default="error_")
    sentence_parser.add_argument("--limit", type=int, default=None)
    sentence_parser.add_argument("--output", default=None)

    inspect_parser = subparsers.add_parser("inspect-artifact", help="Muestra el manifest de un artefacto.")
    inspect_parser.add_argument("--artifact-dir", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="Inventaría corpus, datasets y modelos de los ZIP heredados.")
    inventory_parser.add_argument("--dataset-zip", default=None)
    inventory_parser.add_argument("--augmentation-zip", default=None)
    inventory_parser.add_argument("--subword-zip", default=None)
    inventory_parser.add_argument("--output", default=None)

    research_inventory_parser = subparsers.add_parser("inventory-research", help="Inventaría corridas de investigación con pesos, candidatos y referencias.")
    research_inventory_parser.add_argument("--augmentation-zip", default=None)
    research_inventory_parser.add_argument("--subword-zip", default=None)
    research_inventory_parser.add_argument("--output", default=None)

    research_benchmark_parser = subparsers.add_parser("benchmark-research", help="Calcula métricas desde candidatos/referencias guardados en ZIP heredados.")
    research_benchmark_parser.add_argument("--augmentation-zip", default=None)
    research_benchmark_parser.add_argument("--subword-zip", default=None)
    research_benchmark_parser.add_argument("--output", default=None)
    research_benchmark_parser.add_argument("--markdown-output", default=None)

    extract_parser = subparsers.add_parser("extract-research-runs", help="Extrae corridas heredadas a un model zoo local.")
    extract_parser.add_argument("--output-root", required=True)
    extract_parser.add_argument("--augmentation-zip", default=None)
    extract_parser.add_argument("--subword-zip", default=None)

    excels_parser = subparsers.add_parser("inventory-excels", help="Inventaría datasets y reportes del excels.zip.")
    excels_parser.add_argument("--excels-zip", required=True)
    excels_parser.add_argument("--output", default=None)

    score_parser = subparsers.add_parser("score-report", help="Convierte Score.txt en JSON estructurado.")
    score_parser.add_argument("--score-file", required=True)
    score_parser.add_argument("--output", default=None)

    demo_parser = subparsers.add_parser("gradio-demo", help="Levanta una demo visual con Gradio.")
    demo_parser.add_argument("--artifact-dir", required=True)
    demo_parser.add_argument("--benchmark-report", default=None)
    demo_parser.add_argument("--open-vocab-report", default=None)

    return parser



def command_train(args: argparse.Namespace) -> int:
    artifact_dir = train_lexical_model(load_config(args.config), args.output_dir)
    print(artifact_dir)
    return 0



def command_train_seq2seq(args: argparse.Namespace) -> int:
    artifact_dir = train_seq2seq_model(load_config(args.config), args.output_dir)
    print(artifact_dir)
    return 0



def command_train_torch_reranker(args: argparse.Namespace) -> int:
    artifact_dir = train_torch_reranker_model(
        load_config(args.config),
        args.output_dir,
        pair_files=args.pairs,
        noisy_column=args.noisy_column,
        gold_column=args.gold_column,
        limit=args.limit,
    )
    print(artifact_dir)
    return 0



def command_train_subword(args: argparse.Namespace) -> int:
    artifact_dir = train_subword_model(load_config(args.config), args.output_dir)
    print(artifact_dir)
    return 0



def command_package_legacy_seq2seq(args: argparse.Namespace) -> int:
    artifact_dir = package_legacy_seq2seq(load_config(args.config), args.weights, args.output_dir)
    print(artifact_dir)
    return 0



def command_correct(args: argparse.Namespace) -> int:
    checker = ArtifactSpellChecker.from_artifact_dir(args.artifact_dir)
    corrected, details = checker.correct_text(args.text)
    print(json.dumps({
        "original": args.text,
        "corrected": corrected,
        "tokens": [detail.__dict__ for detail in details],
    }, ensure_ascii=False, indent=2))
    return 0



def command_evaluate(args: argparse.Namespace) -> int:
    result = evaluate_artifact(args.artifact_dir, args.dataset)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0



def command_benchmark_csv(args: argparse.Namespace) -> int:
    result = benchmark_artifact(
        args.artifact_dir,
        args.dataset,
        noisy_column=args.noisy_column,
        gold_column=args.gold_column,
        limit=args.limit,
    )
    payload = result.__dict__
    if args.output:
        write_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def command_benchmark_suite(args: argparse.Namespace) -> int:
    payload = benchmark_suite(args.artifact_root, args.datasets_root, limit=args.limit).to_dict()
    if args.output:
        write_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def command_benchmark_open_vocab(args: argparse.Namespace) -> int:
    payload = benchmark_open_vocab(
        args.artifact_dir,
        args.dataset,
        noisy_column=args.noisy_column,
        gold_column=args.gold_column,
        limit=args.limit,
    ).to_dict()
    if args.output:
        write_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def command_benchmark_clean(args: argparse.Namespace) -> int:
    payload = benchmark_clean_corpus(
        args.artifact_dir,
        args.dataset,
        text_column=args.text_column,
        limit=args.limit,
    ).to_dict()
    if args.output:
        write_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def command_benchmark_sentence_variants(args: argparse.Namespace) -> int:
    payload = benchmark_sentence_variants(
        args.artifact_dir,
        args.dataset,
        sentence_column=args.sentence_column,
        error_prefix=args.error_prefix,
        limit=args.limit,
    ).__dict__
    if args.output:
        write_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def command_inspect_artifact(args: argparse.Namespace) -> int:
    print(json.dumps(load_manifest(args.artifact_dir), ensure_ascii=False, indent=2))
    return 0



def command_inventory(args: argparse.Namespace) -> int:
    inventory = inventory_from_zips(
        dataset_zip=args.dataset_zip,
        augmentation_zip=args.augmentation_zip,
        subword_zip=args.subword_zip,
    )
    payload = inventory.to_dict()
    if args.output:
        write_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def command_inventory_research(args: argparse.Namespace) -> int:
    payload = inventory_research_runs(
        augmentation_zip=args.augmentation_zip,
        subword_zip=args.subword_zip,
    ).to_dict()
    if args.output:
        write_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def command_benchmark_research(args: argparse.Namespace) -> int:
    payload = summarize_research_benchmarks(
        benchmark_research_runs(
            augmentation_zip=args.augmentation_zip,
            subword_zip=args.subword_zip,
        )
    )
    if args.output:
        write_report(args.output, payload)
    if args.markdown_output:
        path = Path(args.markdown_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(benchmark_markdown(payload), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def command_extract_research_runs(args: argparse.Namespace) -> int:
    created = extract_research_runs(
        args.output_root,
        augmentation_zip=args.augmentation_zip,
        subword_zip=args.subword_zip,
    )
    print(json.dumps({'output_root': args.output_root, 'created': [str(p) for p in created]}, ensure_ascii=False, indent=2))
    return 0



def command_inventory_excels(args: argparse.Namespace) -> int:
    payload = inventory_excels_zip(args.excels_zip).to_dict()
    if args.output:
        write_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def command_score_report(args: argparse.Namespace) -> int:
    payload = summarize_scores(parse_score_file(args.score_file))
    if args.output:
        write_report(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0



def command_gradio_demo(args: argparse.Namespace) -> int:
    launch_demo(
        args.artifact_dir,
        benchmark_report=args.benchmark_report,
        open_vocab_report=args.open_vocab_report,
    )
    return 0



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        return command_train(args)
    if args.command == "train-seq2seq":
        return command_train_seq2seq(args)
    if args.command == "train-torch-reranker":
        return command_train_torch_reranker(args)
    if args.command == "train-subword":
        return command_train_subword(args)
    if args.command == "package-legacy-seq2seq":
        return command_package_legacy_seq2seq(args)
    if args.command == "correct":
        return command_correct(args)
    if args.command == "evaluate":
        return command_evaluate(args)
    if args.command == "benchmark-csv":
        return command_benchmark_csv(args)
    if args.command == "benchmark-suite":
        return command_benchmark_suite(args)
    if args.command == "benchmark-open-vocab":
        return command_benchmark_open_vocab(args)
    if args.command == "benchmark-clean":
        return command_benchmark_clean(args)
    if args.command == "benchmark-sentence-variants":
        return command_benchmark_sentence_variants(args)
    if args.command == "inspect-artifact":
        return command_inspect_artifact(args)
    if args.command == "inventory":
        return command_inventory(args)
    if args.command == "inventory-research":
        return command_inventory_research(args)
    if args.command == "benchmark-research":
        return command_benchmark_research(args)
    if args.command == "extract-research-runs":
        return command_extract_research_runs(args)
    if args.command == "inventory-excels":
        return command_inventory_excels(args)
    if args.command == "score-report":
        return command_score_report(args)
    if args.command == "gradio-demo":
        return command_gradio_demo(args)
    parser.error("Comando no soportado")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
