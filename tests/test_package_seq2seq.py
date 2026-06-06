# Prueba automatizada de EIB Spellchecker.
# Valida comportamiento funcional, regresiones o integracion del proyecto.

from pathlib import Path

from eib_spellchecker.config import load_config
from eib_spellchecker.modeling.manifest import load_manifest
from eib_spellchecker.training.seq2seq import package_legacy_seq2seq


def test_package_legacy_seq2seq_writes_manifest(tmp_path: Path) -> None:
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
  enabled: true
  hidden_size: 32
  epochs: 1
  batch_size: 8
  validation_split: 0.1
  error_rate: 0.2
  reverse_input: true
  sample_mode: argmax
""",
        encoding="utf-8",
    )
    fake_weights = tmp_path / "model.keras"
    fake_weights.write_text("placeholder", encoding="utf-8")

    artifact_dir = package_legacy_seq2seq(load_config(config_path), fake_weights, tmp_path / "seq2seq_artifact")
    manifest = load_manifest(artifact_dir)
    assert manifest["backend"] == "legacy-seq2seq"
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "model.keras").exists()
