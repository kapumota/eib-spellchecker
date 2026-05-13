FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY artifacts /app/artifacts
RUN pip install --no-cache-dir -e .

ENV EIB_ARTIFACT_DIR=/app/artifacts/lexical/ash
EXPOSE 8000
CMD ["uvicorn", "eib_spellchecker.api:app", "--host", "0.0.0.0", "--port", "8000"]
