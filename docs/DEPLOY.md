# Despliegue en v3.1

## API

```bash
uvicorn eib_spellchecker.api:app --reload
```

## Demo visual

```bash
pip install -e .[demo]
eib-spellchecker gradio-demo --artifact-dir artifacts/lexical/shi
```

## Docker

Construye la imagen y monta un directorio de artefactos si quieres cambiar el backend activo.

```bash
docker build -t eib-spellchecker:3.1 .
```
