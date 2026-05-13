### Migración histórica desde el repositorio legado

Este documento conserva la trazabilidad de la transición desde el repositorio original de investigación hacia la plataforma modular actual.

#### Propósito

La migración no fue un simple cambio de carpetas. El objetivo fue pasar de una estructura basada en notebooks, scripts sueltos y dependencias implícitas hacia una plataforma con:

- backends intercambiables,
- artefactos versionables,
- CLI y API unificadas,
- benchmarks reproducibles,
- separación explícita entre entrenamiento, inferencia y evaluación.

#### Reinterpretación del legado

##### `Modelo-corrector/Modelo-corrector1/modelo.py`
Reinterpretado en:

- `src/eib_spellchecker/modeling/legacy_seq2seq.py`

##### `Modelo-corrector/Modelo-corrector1/funciones.py`
Su funcionalidad se redistribuyó entre:

- `modeling/legacy_seq2seq.py`
- `training/seq2seq.py`
- `utils/text.py`

##### `entrenamiento.py`
Reemplazado por comandos explícitos de entrenamiento en CLI, por ejemplo:

- `train-seq2seq`
- `train-subword`
- `train-torch-reranker`

##### `evaluacion.py`
Absorbido por:

- `evaluate`
- `benchmark-csv`
- `benchmark-suite`
- benchmarks de robustez introducidos en v6

#### Problemas del legado que se corrigen

- nombres de capas inconsistentes,
- dependencias duras a rutas locales,
- placeholders y archivos incompletos,
- mezcla de notebooks, scripts y datos en el mismo nivel,
- falta de manifiesto del modelo,
- imposibilidad de seleccionar backend en runtime,
- evaluación acoplada a scripts puntuales,

#### Qué cambia conceptualmente en la plataforma actual

La versión consolidada deja de pensar en "un modelo único" y pasa a trabajar con una **arquitectura multibackend**:

- `lexical`
- `subword`
- `torch-hybrid-reranker`
- `legacy-seq2seq`

Cada uno se empaqueta como artefacto y puede activarse desde la misma interfaz operacional.

#### Estrategia recomendada 

1. usar la plataforma moderna como repo destino,
2. importar corpus, notebooks y pesos heredados solo como activos de investigación,
3. entrenar primero un backend operativo (`lexical`, `subword` o `torch-hybrid-reranker`),
4. conservar `legacy-seq2seq` como backend de compatibilidad, no como única ruta del sistema.

#### Estado del documento

Este documento se mantiene como referencia histórica y de trazabilidad.
Para instalación, uso y despliegue de la plataforma actual, consulta:

- `README.md`
- `RUNBOOK.md`
- `MODEL_BACKENDS.md`
