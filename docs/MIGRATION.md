# Migración sugerida desde el repositorio legado

## Qué mover

- `Modelo-corrector/Modelo-corrector1/modelo.py`
  - ya está reinterpretado en `src/eib_spellchecker/modeling/legacy_seq2seq.py`

- `Modelo-corrector/Modelo-corrector1/funciones.py`
  - dividido entre:
    - `modeling/legacy_seq2seq.py`
    - `training/seq2seq.py`
    - `utils/text.py`

- `entrenamiento.py`
  - reemplazado por `eib-spellchecker train-seq2seq`

- `evaluacion.py`
  - absorbido por `eib-spellchecker evaluate`

## Qué problemas del legado se corrigen aquí

- nombres de capas inconsistentes
- dependencia dura a rutas y archivos placeholder
- mezcla de notebooks, scripts y datos
- falta de manifiesto del modelo
- imposibilidad de elegir backend en runtime

## Estrategia recomendada

1. usar esta v2 como repo destino
2. importar corpus y notebooks con `scripts/import_legacy_repo.py`
3. entrenar primero backend léxico para tener API funcional
4. luego empaquetar o reentrenar seq2seq con el backend opcional
