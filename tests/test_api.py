# Prueba automatizada de EIB Spellchecker.
# Valida comportamiento funcional, regresiones o integracion del proyecto.

from pathlib import Path

from fastapi.testclient import TestClient

from eib_spellchecker.api import app, get_checker
from eib_spellchecker.config import load_config
from eib_spellchecker.training.lexical import train_lexical_model


def test_api_correct_endpoint(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "words.txt"
    corpus.write_text("jakon nete joi", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
language: demo
normalize:
  lowercase: true
  strip_accents: false
data:
  corpus_files:
    - words.txt
lexical:
  min_frequency: 1
  max_vocabulary_size: 100
  min_correction_length: 3
  similarity_threshold: 0.7
seq2seq:
  enabled: false
""",
        encoding="utf-8",
    )
    artifact_dir = train_lexical_model(load_config(config_path), tmp_path / "artifacts")
    monkeypatch.setenv("EIB_ARTIFACT_DIR", str(artifact_dir))
    get_checker.cache_clear()

    client = TestClient(app)
    response = client.post("/correct", json={"text": "jakn joi"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["corrected"] == "jakon joi"
