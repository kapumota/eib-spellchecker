# Cómo correr el proyecto paso a paso

## 1. Descomprimir el proyecto

```bash
unzip eib-spellchecker-v6.zip
cd eib-spellchecker-v6
```

## 2. Crear un entorno virtual

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Instalar dependencias

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

## 4. Verificar instalación

```bash
pytest
```

## 5. Probar artefactos ya incluidos

### Subword de demo

```bash
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/subword/demo \
  --text "jakn nete"
```

### Torch moderno de demo

```bash
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/torch/demo \
  --text "jakn nete"
```

### Ver razones por token

La salida incluye `reason` y `confidence`, por ejemplo:
- `best-candidate`
- `protected-proper-name`
- `protected-loanword`
- `protected-out-of-domain`
- `abstained-low-margin`
- `in-vocabulary`

## 6. Entrenar backend subword

```bash
python -m eib_spellchecker.cli train-subword \
  --config configs/demo_subword.yaml \
  --output-dir artifacts/subword/demo_new
```

Luego probarlo:

```bash
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/subword/demo_new \
  --text "jakn nete"
```

## 7. Entrenar backend torch moderno

```bash
python -m eib_spellchecker.cli train-torch-reranker \
  --config configs/demo_torch.yaml \
  --pairs examples/demo_pairs.tsv \
  --output-dir artifacts/torch/demo_new
```

## 8. Benchmark clásico

```bash
python -m eib_spellchecker.cli benchmark-csv \
  --artifact-dir artifacts/lexical/ash \
  --dataset data/samples/excels/common_data_ash.csv \
  --limit 200
```

## 9. Benchmark de robustez

### A. Sobrecorrección sobre texto limpio

```bash
python -m eib_spellchecker.cli benchmark-clean \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_corpus.txt
```

### B. Seen vs unseen, nombres propios y préstamos probables

```bash
python -m eib_spellchecker.cli benchmark-open-vocab \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_pairs.tsv
```

### C. Variantes de oración con `sentence + error_0..error_n`

```bash
python -m eib_spellchecker.cli benchmark-sentence-variants \
  --artifact-dir artifacts/lexical/ash \
  --dataset data/samples/excels/df_ash_sentences.csv \
  --limit 100
```

## 10. API

```bash
export EIB_ARTIFACT_DIR=artifacts/subword/demo
uvicorn eib_spellchecker.api:app --reload
```

Probar:

```bash
curl -X POST http://127.0.0.1:8000/correct \
  -H "Content-Type: application/json" \
  -d '{"text":"jakn nete"}'
```

## 11. Demo visual

```bash
python -m eib_spellchecker.cli gradio-demo \
  --artifact-dir artifacts/subword/demo
```

## 12. Qué backend usar

- `lexical`: baseline rápido y simple
- `subword`: mejor primer paso para palabras nuevas y menos vocabulario cerrado
- `torch-hybrid-reranker`: mejor cuando ya tienes pares `Input,Output`
- `legacy-seq2seq`: solo para compatibilidad con investigación heredada
