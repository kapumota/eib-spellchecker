### Despliegue en v0.6

La v6 puede desplegarse como:

- herramienta de línea de comandos,
- API FastAPI,
- demo visual con Gradio,
- imagen Docker.

La selección del backend activo se realiza por artefacto, no por una clase hardcodeada en el servicio. Esto sigue el enfoque **artifact-driven** del proyecto.

#### 1. API

#### Linux/macOS

```bash
export EIB_ARTIFACT_DIR=artifacts/subword/demo
uvicorn eib_spellchecker.api:app --reload
```

#### Windows PowerShell

```powershell
$env:EIB_ARTIFACT_DIR="artifacts/subword/demo"
uvicorn eib_spellchecker.api:app --reload
```

Luego abre:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

#### 2. Demostración visual

```bash
python -m eib_spellchecker.cli gradio-demo   --artifact-dir artifacts/subword/demo
```

También puedes levantar la demostración con otros artefactos, por ejemplo:

```bash
python -m eib_spellchecker.cli gradio-demo   --artifact-dir artifacts/lexical/ash
```

#### 3. Docker

##### Imagen base

```bash
docker build -t eib-spellchecker:v6 .
docker run --rm -p 8000:8000 eib-spellchecker:v6
```

##### Imagen PyTorch (si tu repositorio la incluye)

```bash
docker build -f Dockerfile.torch -t eib-spellchecker-torch:v6 .
docker run --rm -p 8001:8000 eib-spellchecker-torch:v6
```

#### 4. Cambiar backend activo

La API usa el artefacto apuntado por `EIB_ARTIFACT_DIR`.

Ejemplos:

##### Subword

```bash
export EIB_ARTIFACT_DIR=artifacts/subword/demo
```

##### Lexical Asháninka

```bash
export EIB_ARTIFACT_DIR=artifacts/lexical/ash
```

##### Demostración Torch

```bash
export EIB_ARTIFACT_DIR=artifacts/torch/demo
```

#### 5. Dependencias por perfil

##### Perfil moderno mínimo

```bash
pip install -e .[dev,demo,torch]
```

##### Perfil con investigación heredada

```bash
pip install -e .[dev,demo,torch,research]
```

##### Compatibilidad con legado TensorFlow/Keras

```bash
pip install -e .[tensorflow]
```

#### 6. Recomendación operativa

- usa `subword` para demos orientadas a vocabulario abierto,
- usa `lexical` para baseline rápido y dependencias mínimas,
- usa `torch-hybrid-reranker` cuando ya tienes pares `Input,Output`,
- usa `legacy-seq2seq` solo si necesitas conectar o preservar resultados históricos.

#### 7. Comprobación rápida de despliegue

```bash
python -m eib_spellchecker.cli correct   --artifact-dir artifacts/subword/demo   --text "jakn nete"
```

Si este comando funciona, la ruta principal de inferencia está operativa.

#### 8. Endpoint de prueba

Con la API levantada haz lo siguiente:

```bash
curl -X POST http://127.0.0.1:8000/correct \
  -H "Content-Type: application/json" \
  -d '{"text":"jakn nete"}'
```
