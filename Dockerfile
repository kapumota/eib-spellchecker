# Imagen base de Python
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivos de configuración y documentación
COPY pyproject.toml README.md /app/

# Copiar código fuente y artefactos
COPY src /app/src
COPY artifacts /app/artifacts

# Instalar el paquete en modo editable
RUN pip install --no-cache-dir -e .

# Variable de entorno para los artefactos
ENV EIB_ARTIFACT_DIR=/app/artifacts/lexical/ash

# Exponer el puerto de la API
EXPOSE 8000

# Comando para iniciar el servidor Uvicorn
CMD ["uvicorn", "eib_spellchecker.api:app", "--host", "0.0.0.0", "--port", "8000"]
