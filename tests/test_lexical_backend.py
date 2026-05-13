from pathlib import Path

from eib_spellchecker.config import load_config
from eib_spellchecker.inference.service import ArtifactSpellChecker
from eib_spellchecker.training.lexical import train_lexical_model


def write_config(tmp_path: Path) -> Path:
    corpus = tmp_path / "words.txt"
    corpus.write_text("jakon nete joi jakon nete", encoding="utf-8")
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
    return config_path


def test_lexical_artifact_corrects_close_word(tmp_path: Path) -> None:
    artifact_dir = train_lexical_model(load_config(write_config(tmp_path)), tmp_path / "artifacts")
    checker = ArtifactSpellChecker.from_artifact_dir(artifact_dir)
    assert checker.correct_word("jakn") == "jakon"
    corrected, details = checker.correct_text("jakn nete")
    assert corrected == "jakon nete"
    assert details[0].changed is True
