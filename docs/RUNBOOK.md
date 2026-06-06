### Cómo correr el proyecto paso a paso (v0.6)

#### 1. Crear o activar un entorno virtual

### Linux / macOS

```bash
python -m venv .eib
source .eib/bin/activate
```

##### Windows PowerShell

```powershell
python -m venv .eib
.\.eib\Scripts\Activate.ps1
```

> Si ya tienes un entorno activo, por ejemplo `.eib`, no crees otro. Instala directamente dentro de ese entorno.

#### 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -e .[dev,demo,torch]
```

Opcional para research legacy con `.h5`:

```bash
pip install -e .[research]
```

Opcional para TensorFlow/Keras legado:

```bash
pip install -e .[tensorflow]
```

#### 4. Verificar instalación

```bash
pytest
```

#### 5. Probar artefactos incluidos

##### Subword demostración

```bash
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/subword/demo \
  --text "jakn nete"
```

##### Demostración de Torch 

```bash
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/torch/demo \
  --text "jakn nete"
```

##### Lexical por idioma

```bash
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/lexical/ash \
  --text "abiotero ojempeci"
```

#### 6. Interpretar razones por token

La salida puede incluir:

- `best-candidate`
- `protected-proper-name`
- `protected-loanword`
- `protected-out-of-domain`
- `abstained-low-margin`
- `in-vocabulary`

Estas señales ayudan a inspeccionar la política de abstención y el control de sobrecorrección.

#### 7. Entrenar backend subword

```bash
python -m eib_spellchecker.cli train-subword \
  --config configs/demo_subword.yaml \
  --output-dir artifacts/subword/demo_new
```

Probarlo:

```bash
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/subword/demo_new \
  --text "jakn nete"
```

#### 8. Entrenar backend torch moderno

```bash
python -m eib_spellchecker.cli train-torch-reranker \
  --config configs/demo_torch.yaml \
  --pairs examples/demo_pairs.tsv \
  --output-dir artifacts/torch/demo_new
```

#### 9. Benchmark clásico

```bash
python -m eib_spellchecker.cli benchmark-csv \
  --artifact-dir artifacts/lexical/ash \
  --dataset data/samples/excels/common_data_ash.csv \
  --limit 200
```

#### 10. Benchmark agregado

```bash
python -m eib_spellchecker.cli benchmark-suite \
  --artifact-root artifacts/lexical \
  --datasets-root data/samples/excels \
  --limit 200 \
  --output reports/suite_200.json
```

#### 11. Benchmark de robustez

##### A. Sobrecorrección sobre texto limpio

```bash
python -m eib_spellchecker.cli benchmark-clean \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_corpus.txt
```

##### B. Vocabulario abierto

```bash
python -m eib_spellchecker.cli benchmark-open-vocab \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_pairs.tsv
```

##### C. Variantes por oración

Este comando requiere `sentence + error_0..error_n`.

```bash
python -m eib_spellchecker.cli benchmark-sentence-variants \
  --artifact-dir artifacts/lexical/ash \
  --dataset data/samples/excels/df_ash.csv \
  --limit 100
```

> Usa `df_ash.csv`, `df_shi.csv`, `df_ya.csv` o `df_yi.csv`.
> No uses `*_sentences.csv` para este benchmark.

#### 12. API

##### Linux/macOS

```bash
export EIB_ARTIFACT_DIR=artifacts/subword/demo
uvicorn eib_spellchecker.api:app --reload
```

##### Windows PowerShell

```powershell
$env:EIB_ARTIFACT_DIR="artifacts/subword/demo"
uvicorn eib_spellchecker.api:app --reload
```

Probar con curl:

```bash
curl -X POST http://127.0.0.1:8000/correct \
  -H "Content-Type: application/json" \
  -d '{"text":"jakn nete"}'
```

#### 13. Demostración visual

```bash
python -m eib_spellchecker.cli gradio-demo \
  --artifact-dir artifacts/subword/demo
```

#### 14. Qué backend usar

- `lexical`: baseline rápido y simple
- `subword`: mejor primer paso para palabras nuevas y menos vocabulario cerrado
- `torch-hybrid-reranker`: mejor cuando ya tienes pares `Input,Output`
- `legacy-seq2seq`: solo para compatibilidad con investigación heredada

#### 15. GitHub Actions

Workflow recomendado:

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
      - name: Upgrade pip
        run: python -m pip install --upgrade pip
      - name: Install package with extras
        run: python -m pip install -e '.[dev,torch,demo,research]'
      - name: Run tests
        run: pytest
```

En `tests/test_torch_reranker.py`, conviene agregar:

```python
import pytest
pytest.importorskip("torch", reason="PyTorch no está instalado; se omiten tests del backend torch")
```
