### eib-spellchecker

**EIB-Spellchecker** es una plataforma de corrección ortográfica para lenguas de bajo recurso, diseñada como **software modular, instalable, testeable y desplegable**. La arquitectura del sistema sigue un enfoque **artifact-driven**, donde cada corrector se empaqueta como un artefacto autocontenido y puede ser consumido de forma uniforme desde CLI, API y demostración visual.

La versión actual incorpora una capa explícita de evaluación de **robustez**, una estrategia más orientada a **vocabulario abierto (open-vocabulary)** y mecanismos para reducir la **sobrecorrección** en tokens ya válidos.

#### Principios técnicos del sistema

- **Arquitectura multibackend**: el sistema desacopla la interfaz de inferencia del mecanismo interno de corrección.
- **Artifact-driven deployment**: cada backend se distribuye como artefacto versionable, con `manifest.json` y metadatos de ejecución.
- **Open-vocabulary robustness**: la inferencia no depende exclusivamente de vocabulario cerrado; se incorporan estrategias subléxicas para mejorar el comportamiento ante OOV.
- **Selective abstention**: el sistema intenta evitar correcciones agresivas cuando la evidencia es débil, especialmente en nombres propios, préstamos y texto fuera de dominio.
- **Research asset preservation**: se mantienen y catalogan artefactos heredados de investigación para comparación y trazabilidad experimental.
- **Benchmark-oriented development**: el proyecto incluye utilidades para medir exactitud, mejora en CER, sobrecorrección y robustez ante variantes de error.

### Capacidades principales

#### 1. Backend subword
Se incorpora un backend **`subword`** basado en n-gramas de caracteres como subunidades. Este backend está orientado a escenarios de:

- palabras no vistas en entrenamiento,
- formas largas o morfológicamente complejas,
- ruido menos controlado,
- corrección en contexto de vocabulario abierto.

#### 2. Política de seguridad en inferencia
La capa de inferencia añade señales de decisión como:

- `reason`
- `confidence`

y aplica una política más conservadora para:

- **nombres propios probables**,
- **préstamos probables**,
- **tokens fuera de dominio**,
- casos donde la mejor corrección no supera con claridad al original.

#### 3. Evaluación ampliada de robustez
La plataforma incorpora benchmarks específicos para:

- **open vocabulary**,
- **texto limpio** y detección de sobrecorrección,
- **variantes de oración** con múltiples errores por instancia.

#### Backends disponibles

| Backend | Estado | Framework | Uso recomendado |
|---|---|---|---|
| `lexical` | estable | Python puro | baseline interpretable, baja dependencia, demostración rápida |
| `subword` | estable en v6 | Python puro | vocabulario abierto, OOV, ruido menos controlado |
| `torch-hybrid-reranker` | moderno | PyTorch | reranking sobre candidatos a partir de pares `Input,Output` |
| `legacy-seq2seq` | experimental / compatibilidad | TensorFlow / Keras | preservación y comparación de experimentos heredados |

#### Artefactos incluidos

- `artifacts/lexical/ash`
- `artifacts/lexical/shi`
- `artifacts/lexical/ya`
- `artifacts/lexical/yi`
- `artifacts/torch/demo`
- `artifacts/subword/demo`

#### Instalación rápida

```bash
python -m venv .eib
source .eib/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev,demo,torch,research]
```

#### Flujo mínimo de uso

```bash
### 1. inferencia con backend subword
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/subword/demo \
  --text "jakn nete"

### 2. inferencia con backend torch
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/torch/demo \
  --text "jakn nete"

### 3. benchmark de sobrecorrección en texto limpio
python -m eib_spellchecker.cli benchmark-clean \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_corpus.txt

### 4. benchmark de vocabulario abierto
python -m eib_spellchecker.cli benchmark-open-vocab \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_pairs.tsv
```

#### Comandos técnicos más útiles

#### Entrenamiento de backend subword

```bash
python -m eib_spellchecker.cli train-subword \
  --config configs/demo_subword.yaml \
  --output-dir artifacts/subword/demo_new
```

#### Benchmark de sobrecorrección en corpus limpio

```bash
python -m eib_spellchecker.cli benchmark-clean \
  --artifact-dir artifacts/lexical/ash \
  --dataset data/corpora/ash.txt \
  --limit 200
```

#### Benchmark open-vocabulary

```bash
python -m eib_spellchecker.cli benchmark-open-vocab \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_pairs.tsv
```

#### Benchmark de variantes por oración

Este comando espera un CSV con una columna `sentence` y columnas `error_0`, `error_1`, ..., `error_n`.

```bash
python -m eib_spellchecker.cli benchmark-sentence-variants \
  --artifact-dir artifacts/lexical/ash \
  --dataset data/samples/excels/df_ash.csv \
  --limit 100
```

> Nota: para este benchmark deben usarse archivos como `df_ash.csv`, `df_shi.csv`, `df_ya.csv` o `df_yi.csv`.  
> Los archivos `*_sentences.csv` contienen solo texto limpio y no incluyen columnas `error_*`.

#### API y demostración visual

#### Gradio

```bash
python -m eib_spellchecker.cli gradio-demo \
  --artifact-dir artifacts/subword/demo
```

#### FastAPI / Swagger

Linux/macOS:

```bash
export EIB_ARTIFACT_DIR=artifacts/subword/demo
uvicorn eib_spellchecker.api:app --reload
```

Windows PowerShell:

```powershell
$env:EIB_ARTIFACT_DIR="artifacts/subword/demo"
uvicorn eib_spellchecker.api:app --reload
```

Luego abre:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

#### Qué evalúa el sistema

La plataforma no se limita a medir exactitud puntual. Dependiendo del benchmark y del backend, puede analizar:

- **exact match accuracy**,
- **token accuracy**,
- **CER before / after**,
- **CER improvement**,
- **sobrecorrección** en tokens válidos,
- comportamiento frente a **vocabulario abierto**,
- robustez ante **errores múltiples por oración**.

#### Alcance actual

La v6 representa una mejora importante en ingeniería y evaluación, especialmente en:

- palabras nuevas no vistas,
- nombres propios probables,
- préstamos probables,
- texto fuera de dominio,
- reducción de sobrecorrección,
- variantes de oración con ruido múltiple.

#### Limitaciones

El sistema todavía no equivale a una solución de **generalización abierta total**. La validación más fuerte aún requiere:

- errores humanos reales anotados,
- más datos fuera de dominio,
- evaluación humana por idioma,
- calibración más fina de confianza y abstención,
- reranking contextual más fuerte sobre candidatos.

#### CI recomendado

Para GitHub Actions, instala al menos los extras de desarrollo y PyTorch:

```yaml
name: ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e '.[dev,torch,demo,research]'
      - run: pytest
```

En `tests/test_torch_reranker.py`, conviene omitir los tests si `torch` no está instalado:

```python
import pytest
pytest.importorskip("torch", reason="PyTorch no está instalado; se omiten tests del backend torch")
```

#### Estructura orientativa del proyecto

```text
eib-spellchecker/
├── artifacts/
├── configs/
├── data/
├── docs/
├── examples/
├── model_zoo/
├── reports/
├── src/eib_spellchecker/
└── tests/
```

#### Documentación adicional

- Guía de ejecución: `docs/RUNBOOK.md`
- Benchmarks y reportes: `reports/`
- Activos de investigación heredados: `model_zoo/legacy_runs/`

#### Filosofía de evolución

EIB-Spellchecker v6 debe entenderse como una plataforma en transición desde corrección ortográfica basada en ruido semi-controlado hacia un sistema más robusto de **open-vocabulary correction**. La dirección futura del proyecto prioriza:

- backends híbridos carácter + subpalabra,
- reranking contextual liviano,
- políticas explícitas de abstención,
- evaluación con errores humanos auténticos,
- mejor manejo de nombres propios, préstamos y dominios no vistos.
