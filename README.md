# eib-spellchecker v6

Versión 6 del corrector ortográfico EIB como **software modular, instalable, testeable y desplegable**, ahora con foco explícito en **robustez**, **vocabulario abierto** y **menos sobrecorrección**.

## Qué agrega v6

- backend nuevo: **`subword`**
  - usa n-gramas de caracteres como subpalabras
  - ayuda más con palabras no vistas, formas largas y ruido abierto
- política de seguridad para inferencia
  - protege mejor **nombres propios**
  - reduce cambios agresivos en **préstamos probables**
  - intenta abstenerse más en tokens **fuera de dominio**
- benchmark de robustez
  - `benchmark-open-vocab`
  - `benchmark-clean`
  - `benchmark-sentence-variants`
- salida más informativa en `correct`
  - cada token ahora devuelve `confidence` y `reason`

## Backends disponibles

| Backend | Estado | Framework | Uso recomendado |
|---|---|---|---|
| `lexical` | estable | Python puro | baseline, demo rápida, baja dependencia |
| `subword` | nuevo en v6 | Python puro | vocabulario abierto, OOV y ruido menos controlado |
| `torch-hybrid-reranker` | moderno | PyTorch | reranking sobre candidatos con pares `Input,Output` |
| `legacy-seq2seq` | experimental/compat | TensorFlow/Keras | conservar experimentos heredados |

## Artefactos listos

- `artifacts/lexical/ash`
- `artifacts/lexical/shi`
- `artifacts/lexical/ya`
- `artifacts/lexical/yi`
- `artifacts/torch/demo`
- `artifacts/subword/demo`

## Instalación rápida

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev,demo,torch]
```

## Flujo rápido

```bash
# 1) backend subword de demo
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/subword/demo \
  --text "jakn nete"

# 2) backend torch moderno
python -m eib_spellchecker.cli correct \
  --artifact-dir artifacts/torch/demo \
  --text "jakn nete"

# 3) benchmark de robustez sobre corpus limpio
python -m eib_spellchecker.cli benchmark-clean \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_corpus.txt

# 4) benchmark de vocabulario abierto
python -m eib_spellchecker.cli benchmark-open-vocab \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_pairs.tsv
```

## Comandos nuevos más útiles

```bash
# entrenar backend subword
python -m eib_spellchecker.cli train-subword \
  --config configs/demo_subword.yaml \
  --output-dir artifacts/subword/demo_new

# medir sobrecorrección en texto limpio
python -m eib_spellchecker.cli benchmark-clean \
  --artifact-dir artifacts/lexical/ash \
  --dataset data/corpora/ash.txt \
  --limit 200

# separar métricas por seen/unseen, nombres propios y préstamos probables
python -m eib_spellchecker.cli benchmark-open-vocab \
  --artifact-dir artifacts/subword/demo \
  --dataset examples/demo_pairs.tsv

# evaluar CSV con sentence + error_0..error_n
python -m eib_spellchecker.cli benchmark-sentence-variants \
  --artifact-dir artifacts/lexical/ash \
  --dataset data/samples/excels/df_ash_sentences.csv \
  --limit 100
```

## Qué problema resuelve v6 mejor que v5

v5 medía bien rendimiento sobre ruido conocido. v6 empieza a cubrir más directamente:

- palabras nuevas no vistas
- nombres propios
- préstamos probables
- fuera de dominio
- sobrecorrección en palabras ya correctas
- variantes de oración con errores múltiples

## Límite honesto

Sigue sin ser “generalización abierta total”. La v6 mejora bastante la capa de software y evaluación, pero la validación más fuerte todavía requiere:

- errores humanos reales anotados
- más datos fuera de dominio
- evaluación humana
- calibración más fina por idioma

La guía completa quedó en `docs/RUNBOOK.md`.
