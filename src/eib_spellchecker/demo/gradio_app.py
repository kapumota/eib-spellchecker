# Codigo base de EIB Spellchecker.
# Implementa una interfaz Gradio didactica para probar artefactos de correccion.

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from eib_spellchecker.inference.service import ArtifactSpellChecker


LANGUAGE_NAMES = {
    "ash": "Ashaninka",
    "shi": "Shipibo-Konibo",
    "ya": "Yanesha",
    "yi": "Yine",
    "demo": "Demo",
}


REAL_ARTIFACTS = {
    "Ashaninka": "artifacts/lexical/ash",
    "Shipibo-Konibo": "artifacts/lexical/shi",
    "Yanesha": "artifacts/lexical/ya",
    "Yine": "artifacts/lexical/yi",
}


def find_project_root() -> Path:
    current = Path.cwd()

    for path in [current, *current.parents]:
        if (path / "pyproject.toml").exists():
            return path

    return current


def load_report(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None

    report_path = Path(path)

    if not report_path.exists():
        return None

    return json.loads(report_path.read_text(encoding="utf-8"))


def load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "La demostración con graficos requiere matplotlib. "
            "Instala el extra con: pip install -e .[demo]"
        ) from exc

    return plt


def read_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(artifact_dir) / "manifest.json"

    if not manifest_path.exists():
        return {}

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def normalize_language(language: str | None, artifact_dir: str | Path) -> str:
    if language:
        return language

    path = str(artifact_dir).replace("\\", "/")

    for code in ["ash", "shi", "ya", "yi", "demo"]:
        if f"/{code}" in path or path.endswith(code):
            return code

    return "desconocido"


def language_label(language: str) -> str:
    return LANGUAGE_NAMES.get(language, language)


def is_demo_artifact(language: str, artifact_dir: str | Path) -> bool:
    path = str(artifact_dir).replace("\\", "/")
    return language == "demo" or "subword/demo" in path


def row_value(row: dict[str, str], candidates: list[str]) -> str | None:
    lowered = {key.lower(): value for key, value in row.items()}

    for candidate in candidates:
        value = row.get(candidate)
        if value:
            return value.strip()

        value = lowered.get(candidate.lower())
        if value:
            return value.strip()

    return None


def examples_from_tsv(path: Path, limit: int = 5) -> list[str]:
    if not path.exists():
        return []

    examples: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        except csv.Error:
            dialect = csv.excel_tab

        reader = csv.DictReader(handle, dialect=dialect)

        if reader.fieldnames:
            for row in reader:
                value = row_value(
                    row,
                    [
                        "Input",
                        "input",
                        "noisy",
                        "error",
                        "source",
                        "original",
                        "text",
                        "sentence",
                    ],
                )
                if value:
                    examples.append(value)
                if len(examples) >= limit:
                    break

    return examples


def examples_from_csv(path: Path, limit: int = 5) -> list[str]:
    if not path.exists():
        return []

    examples: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            value = row_value(
                row,
                [
                    "Input",
                    "input",
                    "noisy",
                    "error",
                    "source",
                    "original",
                    "text",
                    "sentence",
                    "error_0",
                ],
            )
            if value:
                examples.append(value)
            if len(examples) >= limit:
                break

    return examples


def examples_from_corpus(path: Path, limit: int = 5) -> list[str]:
    if not path.exists():
        return []

    examples: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean:
            continue

        examples.append(clean)

        if len(examples) >= limit:
            break

    return examples


def discover_examples(artifact_dir: str | Path, language: str) -> list[str]:
    root = find_project_root()
    examples: list[str] = []

    if language == "demo":
        examples.extend(examples_from_tsv(root / "examples" / "demo_pairs.tsv"))

    if language in {"ash", "shi", "ya", "yi"}:
        examples.extend(
            examples_from_csv(
                root / "data" / "samples" / "excels" / f"keyboard_data_{language}.csv"
            )
        )
        examples.extend(
            examples_from_csv(
                root / "data" / "samples" / "excels" / f"common_data_{language}.csv"
            )
        )
        examples.extend(examples_from_corpus(root / "data" / "corpora" / f"{language}.txt"))

    if not examples:
        if language == "demo":
            examples = [
                "jakn nete",
                "jakon nete joi",
                "nete jakn",
            ]
        else:
            examples = [
                "Escribe una palabra o frase tomada del corpus de la lengua cargada.",
            ]

    unique_examples: list[str] = []

    for example in examples:
        if example not in unique_examples:
            unique_examples.append(example)

    return unique_examples[:8]


def artifact_explanation(
    artifact_dir: str | Path,
    language: str,
    backend_name: str,
    manifest: dict[str, Any],
) -> str:
    artifact_path = str(artifact_dir)
    language_name = language_label(language)
    backend_from_manifest = manifest.get("backend", "no especificado")

    if is_demo_artifact(language, artifact_dir):
        warning = """
> **Nota importante:** este artefacto es solo de demostracion.  
> Por eso aparece `Idioma: demo`. No representa directamente Ashaninka, Shipibo-Konibo, Yanesha o Yine.  
> Para probar lenguas reales, abre la demostración con `artifacts/lexical/ash`, `artifacts/lexical/shi`, `artifacts/lexical/ya` o `artifacts/lexical/yi`.
"""
    else:
        warning = """
> Este artefacto corresponde a una lengua real del proyecto.  
> Los ejemplos sugeridos se toman, cuando es posible, de los datos asociados a esa lengua.
"""

    return f"""
### Artefacto cargado

- **Ruta:** `{artifact_path}`
- **Idioma mostrado:** `{language_name}`
- **Codigo interno de idioma:** `{language}`
- **Backend en ejecucion:** `{backend_name}`
- **Backend en manifest:** `{backend_from_manifest}`

{warning}
"""


def real_artifacts_markdown() -> str:
    rows = []

    for language, artifact in REAL_ARTIFACTS.items():
        rows.append(
            f"| {language} | `{artifact}` | "
            f"`python -m eib_spellchecker.cli gradio-demo --artifact-dir {artifact}` |"
        )

    table = "\n".join(rows)

    return f"""
### Artefactos reales disponibles

| Lengua | Artefacto | Comando |
|---|---|---|
{table}

El artefacto `artifacts/subword/demo` sirve para probar la interfaz y el backend subword de demostracion.
"""


def metric_from_report(report: dict[str, Any], names: list[str]) -> float | None:
    for name in names:
        value = report.get(name)
        if isinstance(value, (int, float)):
            return float(value)

    metrics = report.get("metrics")

    if isinstance(metrics, dict):
        for name in names:
            value = metrics.get(name)
            if isinstance(value, (int, float)):
                return float(value)

    return None


def build_live_correction_plot(details: list[Any]):
    plt = load_pyplot()

    changed_tokens = sum(1 for detail in details if detail.changed)
    unchanged_tokens = max(len(details) - changed_tokens, 0)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Conservados", "Corregidos"], [unchanged_tokens, changed_tokens])
    ax.set_title("Correccion por token")
    ax.set_ylabel("Cantidad")
    ax.set_ylim(0, max(len(details), 1))
    fig.tight_layout()

    return fig


def build_cer_plot(report: dict[str, Any] | None):
    if not report:
        return None

    cer_before = metric_from_report(report, ["cer_before", "CER_before", "cer_antes"])
    cer_after = metric_from_report(report, ["cer_after", "CER_after", "cer_despues"])

    if cer_before is None or cer_after is None:
        return None

    plt = load_pyplot()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["CER antes", "CER despues"], [cer_before, cer_after])
    ax.set_title("Reduccion de error de caracteres")
    ax.set_ylabel("CER")
    fig.tight_layout()

    return fig


def build_open_vocab_plot(report: dict[str, Any] | None):
    if not report:
        return None

    buckets = report.get("buckets")

    if not isinstance(buckets, list) or not buckets:
        return None

    names: list[str] = []
    values: list[float] = []

    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue

        name = str(bucket.get("name", bucket.get("bucket", "bucket")))
        value = bucket.get("exact_match_accuracy", bucket.get("accuracy"))

        if isinstance(value, (int, float)):
            names.append(name)
            values.append(float(value))

    if not names:
        return None

    plt = load_pyplot()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(names, values)
    ax.set_title("Exactitud por tipo de vocabulario")
    ax.set_ylabel("Exact match accuracy")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    return fig


def correction_rows(details: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for detail in details:
        rows.append(
            {
                "original": detail.original,
                "corregido": detail.corrected,
                "cambiado": detail.changed,
                "razon": getattr(detail, "reason", None),
                "confianza": getattr(detail, "confidence", None),
            }
        )

    return rows


def launch_demo(
    artifact_dir: str | Path,
    benchmark_report: str | Path | None = None,
    open_vocab_report: str | Path | None = None,
) -> None:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError(
            "La demostración requiere Gradio. Instala el extra con: pip install -e .[demo]"
        ) from exc

    checker = ArtifactSpellChecker.from_artifact_dir(artifact_dir)
    manifest = read_manifest(artifact_dir)

    language = normalize_language(getattr(checker, "language", None), artifact_dir)
    backend_name = checker.backend.__class__.__name__
    examples = discover_examples(artifact_dir, language)

    benchmark_payload = load_report(benchmark_report)
    open_vocab_payload = load_report(open_vocab_report)

    def select_example(example: str) -> str:
        return example

    def correct_text(text: str):
        if not text.strip():
            return "", [], None

        corrected, details = checker.correct_text(text)
        rows = correction_rows(details)
        live_plot = build_live_correction_plot(details)

        return corrected, rows, live_plot

    with gr.Blocks(title="Demostración eib-spellchecker") as demo:
        gr.Markdown("# Demostración eib-spellchecker ")
        gr.Markdown(
            artifact_explanation(
                artifact_dir=artifact_dir,
                language=language,
                backend_name=backend_name,
                manifest=manifest,
            )
        )

        with gr.Tab("Guia rapida"):
            gr.Markdown(
                """
### Como usar esta pantalla

1. Revisa primero que artefacto se cargo.
2. Si dice `Idioma: demo`, estas usando un artefacto de demostracion.
3. Para lenguas reales, ejecuta la demostración con uno de los artefactos lexicales.
4. Usa los ejemplos precargados o escribe una palabra tomada del corpus correspondiente.
5. Presiona **Corregir**.
6. Revisa el texto corregido, los cambios por token y el grafico.
"""
            )
            gr.Markdown(real_artifacts_markdown())

        with gr.Tab("Correccion interactiva"):
            example_selector = gr.Dropdown(
                choices=examples,
                value=examples[0],
                label="Ejemplos sugeridos para el artefacto cargado",
            )

            source = gr.Textbox(
                label="Texto de entrada",
                lines=5,
                value=examples[0],
                placeholder="Escribe aqui una palabra o frase para corregir.",
            )

            example_selector.change(
                select_example,
                inputs=[example_selector],
                outputs=[source],
            )

            target = gr.Textbox(label="Texto corregido", lines=5)
            changes = gr.JSON(label="Detalle por token")
            live_plot = gr.Plot(label="Resumen de tokens conservados y corregidos")

            run = gr.Button("Corregir")
            run.click(
                correct_text,
                inputs=[source],
                outputs=[target, changes, live_plot],
            )

        with gr.Tab("Artefactos por lengua"):
            gr.Markdown(real_artifacts_markdown())

        with gr.Tab("Reporte de benchmark"):
            cer_plot = build_cer_plot(benchmark_payload)

            if cer_plot is not None:
                gr.Plot(value=cer_plot, label="CER antes vs CER despues")
            else:
                gr.Markdown(
                    """
No se cargo un reporte compatible con metricas CER.

Para generar un reporte:

```bash
python -m eib_spellchecker.cli benchmark-csv \\
  --artifact-dir artifacts/subword/demo \\
  --dataset examples/demo_pairs.tsv \\
  --output reports/subword_demo_benchmark.json
```
"""
                )

        with gr.Tab("Vocabulario abierto"):
            open_vocab_plot = build_open_vocab_plot(open_vocab_payload)

            if open_vocab_plot is not None:
                gr.Plot(value=open_vocab_plot, label="Exactitud por vocabulario")
            else:
                gr.Markdown(
                    """
No se cargo un reporte compatible de vocabulario abierto.

Para generar un reporte:

```bash
python -m eib_spellchecker.cli benchmark-open-vocab \\
  --artifact-dir artifacts/subword/demo \\
  --dataset examples/demo_pairs.tsv \\
  --output reports/subword_demo_open_vocab.json
```
"""
                )

    demo.launch()
