### Validación reproducible

#### Objetivo

Este documento describe la validación mínima reproducible de EIB Spellchecker como software instalable, testeable y ejecutable.

La validación no depende únicamente de badges visuales en el README. El badge principal de CI debe apuntar a un workflow real de GitHub Actions. Ese workflow ejecuta `scripts/validate.sh`, que comprueba instalación, pruebas, CLI, API y benchmarks de demostración.

#### Comando principal

```bash
make validate
```

También puede ejecutarse directamente:

```bash
bash scripts/validate.sh
```

#### Qué se valida

- Estructura mínima del repositorio.
- Existencia de `README.md`, `LICENSE`, `pyproject.toml`, `src/eib_spellchecker` y `tests`.
- Existencia del artefacto de demostración `artifacts/subword/demo`.
- Ausencia de archivos generados versionados, como `__pycache__`, `.pyc`, `build`, `dist` o `.egg-info`.
- Consistencia de dependencias instaladas con `python -m pip check`.
- Ejecución de pruebas automatizadas con `pytest`.
- Importación de la API FastAPI.
- Ejecución del CLI de corrección con backend `subword`.
- Ejecución de benchmarks mínimos de texto limpio y vocabulario abierto.

#### Relación con los badges

El README muestra badges para resumir el estado técnico del proyecto. El badge más importante es `CI`, porque depende del workflow `.github/workflows/ci.yml`.

Si `scripts/validate.sh` falla, el workflow falla y el badge de CI deja de mostrar un estado exitoso.

#### Alcance

Esta validación demuestra que el proyecto funciona como software ejecutable y verificable en un entorno limpio de CI. No reemplaza una evaluación lingüística completa, una validación con hablantes expertos ni una evaluación con errores humanos reales anotados.
