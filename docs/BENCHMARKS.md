# Benchmarking en v3.1

La v3.1 soporta tres niveles de evaluación:

1. **TSV**: `noisy<TAB>gold`
2. **CSV**: columnas `Input,Output`
3. **Benchmark agregado** sobre un directorio con archivos `*_data_<lang>.csv`

## Métricas

- `exact_match_accuracy`
- `token_accuracy`
- `cer_before`
- `cer_after`
- `cer_improvement`

## Comandos

```bash
eib-spellchecker benchmark-csv --artifact-dir artifacts/lexical/ash --dataset data/excels/common_data_ash.csv --limit 100
eib-spellchecker benchmark-suite --artifact-root artifacts/lexical --datasets-root data/excels --limit 100
```

## Recomendación

Usa `benchmark-csv` cuando quieras inspección puntual por dataset.
Usa `benchmark-suite` cuando quieras una tabla resumida por idioma y tipo de ruido para presentación.
