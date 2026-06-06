### Benchmarking en v0.6

La versión 0.6 amplía la evaluación clásica del proyecto y separa la medición en cuatro regímenes complementarios:

1. **Evaluación supervisada clásica**
   - TSV con formato `noisy<TAB>gold`
   - CSV con columnas `Input,Output`

2. **Benchmark agregado por idioma y tipo de ruido**
   - directorios con archivos `*_data_<lang>.csv`

3. **Robustez sobre texto limpio**
   - detección de **sobrecorrección**
   - inspección de comportamiento conservador sobre tokens ya válidos

4. **Robustez open-vocabulary**
   - separación entre comportamiento sobre tokens vistos y no vistos
   - seguimiento de casos compatibles con nombres propios, préstamos probables y fuera de dominio

#### Métricas principales

La plataforma reporta, según el benchmark:

- `exact_match_accuracy`
- `token_accuracy`
- `cer_before`
- `cer_after`
- `cer_improvement`

En benchmarks de robustez, además interesa observar:

- tasa de **sobrecorrección**
- comportamiento sobre **tokens OOV**
- estabilidad ante **variantes múltiples por oración**

#### Comandos principales

##### 1. Benchmark clásico sobre CSV `Input,Output`

```bash
python -m eib_spellchecker.cli benchmark-csv   --artifact-dir artifacts/lexical/ash   --dataset data/samples/excels/common_data_ash.csv   --limit 200
```

##### 2. Benchmark agregado por idioma y tipo de error

```bash
python -m eib_spellchecker.cli benchmark-suite   --artifact-root artifacts/lexical   --datasets-root data/samples/excels   --limit 200   --output reports/suite_200.json
```

##### 3. Sobrecorrección sobre texto limpio

```bash
python -m eib_spellchecker.cli benchmark-clean   --artifact-dir artifacts/subword/demo   --dataset examples/demo_corpus.txt
```

##### 4. Robustez open-vocabulary

```bash
python -m eib_spellchecker.cli benchmark-open-vocab   --artifact-dir artifacts/subword/demo   --dataset examples/demo_pairs.tsv
```

##### 5. Variantes por oración

Este benchmark espera una columna `sentence` y columnas `error_0`, `error_1`, ..., `error_n`.

```bash
python -m eib_spellchecker.cli benchmark-sentence-variants   --artifact-dir artifacts/lexical/ash   --dataset data/samples/excels/df_ash.csv   --limit 100
```

> Usa archivos como `df_ash.csv`, `df_shi.csv`, `df_ya.csv`, `df_yi.csv`.
> No uses `*_sentences.csv` para este comando, porque esos archivos contienen solo texto limpio.

#### Estrategia recomendada

- Usa `benchmark-csv` cuando quieras inspección puntual por dataset.
- Usa `benchmark-suite` cuando quieras una vista agregada por idioma y ruido.
- Usa `benchmark-clean` para medir cuánto daña el sistema entradas ya correctas.
- Usa `benchmark-open-vocab` para analizar robustez fuera de vocabulario cerrado.
- Usa `benchmark-sentence-variants` para medir degradación bajo errores múltiples por oración.

#### Interpretación práctica

Para una lectura más útil de resultados:

- un `exact_match_accuracy` alto indica corrección exacta frecuente,
- un `token_accuracy` alto indica estabilidad a nivel de token,
- `cer_after < cer_before` indica mejora real,
- una mejora fuerte en `cer_improvement` no siempre implica baja sobrecorrección,
- por eso conviene combinar benchmarks clásicos y benchmarks de robustez.

#### Recomendación de uso

Para mostrar resultados de v0.6:

1. correr `benchmark-csv` sobre un dataset concreto,
2. correr `benchmark-suite` para comparar ruido por idioma,
3. correr `benchmark-clean` para discutir sobrecorrección,
4. correr `benchmark-open-vocab` para mostrar transición hacia open-vocabulary.
