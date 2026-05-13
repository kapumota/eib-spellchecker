from pathlib import Path

from eib_spellchecker.config import load_config
from eib_spellchecker.evaluation.metrics import evaluate_artifact
from eib_spellchecker.training.lexical import train_lexical_model


def test_evaluation_metrics(tmp_path: Path) -> None:
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
    dataset = tmp_path / "pairs.tsv"
    dataset.write_text("jakn\tjakon\nnete\tnete\n", encoding="utf-8")

    result = evaluate_artifact(artifact_dir, dataset)
    assert result.total == 2
    assert result.correct == 2
    assert result.accuracy == 1.0
