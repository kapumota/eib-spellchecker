#!/usr/bin/env bash
set -euo pipefail

# Validacion reproducible para EIB Spellchecker.
# El objetivo es comprobar instalacion, pruebas, CLI, API y benchmarks de demostracion.

echo "Validando EIB Spellchecker..."

echo "1. Verificando estructura minima"
test -f README.md
test -f LICENSE
test -f pyproject.toml
test -d src/eib_spellchecker
test -d tests
test -d artifacts/subword/demo
test -f artifacts/subword/demo/manifest.json

echo "2. Verificando que no existan archivos generados versionados"
if git ls-files | grep -E '(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|htmlcov|build|dist|\.eggs)(/|$)|\.pyc$|\.pyo$|\.egg-info/' > /tmp/eib_generated_files.txt; then
  echo "Error: hay archivos generados versionados:"
  cat /tmp/eib_generated_files.txt
  exit 1
fi

echo "3. Verificando metadatos del paquete"
python -m pip check
python - <<'PY'
from importlib.metadata import version
print("Paquete instalado:", version("eib-spellchecker"))
PY

echo "4. Ejecutando pruebas automatizadas"
python -m pytest

echo "5. Verificando importacion de API"
python - <<'PY'
from eib_spellchecker.api import app
print("API FastAPI importada:", getattr(app, "title", "sin titulo"))
PY

echo "6. Verificando CLI de inferencia con backend subword"
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/subword/demo \
  --text "jakn nete" > /tmp/eib_cli_subword.txt

test -s /tmp/eib_cli_subword.txt

echo "7. Ejecutando benchmark de texto limpio"
python -m eib_spellchecker.cli benchmark-clean \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_corpus.txt > /tmp/eib_benchmark_clean.txt

test -s /tmp/eib_benchmark_clean.txt

echo "8. Ejecutando benchmark open-vocabulary"
python -m eib_spellchecker.cli benchmark-open-vocab \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_pairs.tsv > /tmp/eib_benchmark_open_vocab.txt

test -s /tmp/eib_benchmark_open_vocab.txt

echo "Validacion completada correctamente."
