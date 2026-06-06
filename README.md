### eib-spellchecker

**EIB Spellchecker** es una plataforma de corrección ortográfica para lenguas de bajo recurso. Está diseñada como software modular, instalable, testeable y desplegable. Su arquitectura usa un enfoque de **despliegue guiado por artefactos**, donde cada corrector se empaqueta como un artefacto autocontenido y puede ser consumido de forma uniforme desde la línea de comandos, una API web o una demostración visual.

La versión actual `0.6.0` incorpora evaluación explícita de robustez, una estrategia orientada a **vocabulario abierto** y mecanismos para reducir la **sobrecorrección** en tokens que ya son válidos.

#### Principios técnicos del sistema

* **Arquitectura con múltiples backends**: el sistema desacopla la interfaz de inferencia del mecanismo interno de corrección.
* **Despliegue guiado por artefactos**: cada backend se distribuye como un artefacto versionable, con `manifest.json` y metadatos de ejecución.
* **Robustez con vocabulario abierto**: la inferencia no depende exclusivamente de un vocabulario cerrado. Se incorporan estrategias subléxicas para mejorar el comportamiento ante palabras no vistas.
* **Abstención selectiva**: el sistema intenta evitar correcciones agresivas cuando la evidencia es débil, especialmente en nombres propios, préstamos y texto fuera de dominio.
* **Preservación de activos de investigación**: se mantienen y catalogan artefactos heredados para comparación, migración y trazabilidad experimental.
* **Desarrollo orientado a evaluación**: el proyecto incluye utilidades para medir exactitud, mejora en CER, sobrecorrección y robustez ante variantes de error.

### Capacidades principales

#### 1. Backend subword

El backend `subword` usa n-gramas de caracteres como subunidades. Está orientado a escenarios donde aparecen:

* palabras no vistas durante el entrenamiento,
* formas largas o morfológicamente complejas,
* ruido menos controlado,
* corrección con vocabulario abierto.

#### 2. Política de seguridad en inferencia

La capa de inferencia incorpora señales de decisión como:

* `reason`
* `confidence`

Con estas señales, el sistema aplica una política más conservadora en casos como:

* nombres propios probables,
* préstamos probables,
* tokens fuera de dominio,
* situaciones donde la mejor corrección no supera con claridad al token original.

#### 3. Evaluación ampliada de robustez

La plataforma incorpora benchmarks específicos para:

* vocabulario abierto,
* texto limpio y detección de sobrecorrección,
* variantes de oración con múltiples errores por instancia.

### Backends disponibles

| Backend                 | Estado                        | Framework          | Uso recomendado                                                  |
| ----------------------- | ----------------------------- | ------------------ | ---------------------------------------------------------------- |
| `lexical`               | estable                       | Python puro        | Línea base interpretable, baja dependencia y demostración rápida |
| `subword`               | estable en `0.6.0`            | Python puro        | Vocabulario abierto, palabras no vistas y ruido menos controlado |
| `torch-hybrid-reranker` | experimental moderno          | PyTorch            | Reordenamiento de candidatos a partir de pares `Input,Output`    |
| `legacy-seq2seq`        | experimental / compatibilidad | TensorFlow / Keras | Preservación y comparación de experimentos heredados             |

### Artefactos incluidos

#### Artefactos lexicales

```text
artifacts/lexical/ash
artifacts/lexical/shi
artifacts/lexical/ya
artifacts/lexical/yi
```

#### Artefactos modernos de demostración

```text
artifacts/subword/demo
artifacts/torch/demo
```

#### Activos heredados de investigación

```text
model_zoo/legacy_runs
```

### Instalación rápida

```bash
python -m venv .eib
source .eib/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev,demo,torch,research]
```

En Windows PowerShell:

```powershell
python -m venv .eib
.\.eib\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev,demo,torch,research]
```

### Flujo mínimo de uso

#### 1. Inferencia con backend subword

```bash
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/subword/demo \
  --text "jakn nete"
```

#### 2. Inferencia con backend torch

```bash
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/torch/demo \
  --text "jakn nete"
```

#### 3. Benchmark de sobrecorrección en texto limpio

```bash
python -m eib_spellchecker.cli benchmark-clean \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_corpus.txt
```

#### 4. Benchmark de vocabulario abierto

```bash
python -m eib_spellchecker.cli benchmark-open-vocab \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_pairs.tsv
```

### Comandos técnicos útiles

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

#### Benchmark de vocabulario abierto

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

Nota: para este benchmark deben usarse archivos como `df_ash.csv`, `df_shi.csv`, `df_ya.csv` o `df_yi.csv`. Los archivos `*_sentences.csv` contienen solo texto limpio y no incluyen columnas `error_*`.

### API y demostración visual

#### Demostración con Gradio

```bash
python -m eib_spellchecker.cli gradio-demo \
  --artifact-dir artifacts/subword/demo
```

#### API con FastAPI y Swagger

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

* Swagger UI: `http://127.0.0.1:8000/docs`
* OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

### Evaluación del sistema

La plataforma no se limita a medir exactitud puntual. Dependiendo del benchmark y del backend, puede analizar:

* exactitud por coincidencia exacta,
* exactitud por token,
* CER antes y después de la corrección,
* mejora relativa de CER,
* sobrecorrección en tokens válidos,
* comportamiento frente a vocabulario abierto,
* robustez ante errores múltiples por oración.

### Alcance actual

La versión `0.6.0` representa una mejora importante en ingeniería y evaluación, especialmente en:

* palabras nuevas no vistas,
* nombres propios probables,
* préstamos probables,
* texto fuera de dominio,
* reducción de sobrecorrección,
* variantes de oración con ruido múltiple,
* preservación de modelos y reportes heredados.

### Limitaciones

El sistema todavía no equivale a una solución de generalización abierta total. La validación más fuerte aún requiere:

* errores humanos reales anotados,
* más datos fuera de dominio,
* evaluación humana por idioma,
* calibración más fina de confianza y abstención,
* reordenamiento contextual más fuerte sobre candidatos,
* comparación sistemática entre backends lexicales, subléxicos y neuronales.

### CI recomendado

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

pytest.importorskip(
    "torch",
    reason="PyTorch no está instalado; se omiten tests del backend torch",
)
```

### Estructura del proyecto

```text
eib-spellchecker/
├── .github/workflows/
├── artifacts/
├── configs/
├── data/
├── docs/
├── examples/
├── model_zoo/legacy_runs/
├── reports/
├── scripts/
├── src/eib_spellchecker/
├── tests/
├── Dockerfile
├── Dockerfile.torch
├── Makefile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

### Descripción de carpetas principales

#### `src/eib_spellchecker/`

Contiene el código base del paquete, incluyendo CLI, API, configuración, backends, evaluación, entrenamiento, demostración e inferencia.

#### `artifacts/`

Contiene artefactos autocontenidos usados por los backends de corrección. Cada artefacto debe incluir metadatos suficientes para reproducir o consumir el corrector.

#### `configs/`

Contiene archivos YAML de configuración para entrenamiento, inferencia o experimentación.

#### `examples/`

Contiene archivos pequeños de ejemplo para demostraciones, pruebas manuales y ejecución rápida.

#### `reports/`

Contiene reportes JSON de evaluación y benchmarks.

#### `model_zoo/legacy_runs/`

Contiene activos heredados de investigación, corridas experimentales y modelos preservados para comparación o migración.

#### `tests/`

Contiene pruebas automatizadas para validar componentes de API, backends, evaluación, inventarios, parsers y robustez.

### Documentación adicional

* Guía de ejecución: `docs/RUNBOOK.md`
* Reportes y benchmarks: `reports/`
* Configuraciones: `configs/`
* Activos heredados de investigación: `model_zoo/legacy_runs/`

### Filosofía de evolución

EIB Spellchecker debe entenderse como una plataforma en transición desde corrección ortográfica basada en ruido semi-controlado hacia un sistema más robusto de corrección con vocabulario abierto.

La dirección futura del proyecto prioriza:

* backends híbridos carácter + subpalabra,
* reordenamiento contextual liviano,
* políticas explícitas de abstención,
* evaluación con errores humanos auténticos,
* mejor manejo de nombres propios, préstamos y dominios no vistos,
* empaquetado reproducible de artefactos,
* reportes comparables entre versiones.
