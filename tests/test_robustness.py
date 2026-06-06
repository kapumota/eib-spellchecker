# Prueba automatizada de EIB Spellchecker.
# Valida comportamiento funcional, regresiones o integracion del proyecto.

from __future__ import annotations

from pathlib import Path

from eib_spellchecker.config import load_config
from eib_spellchecker.evaluation.robustness import benchmark_clean_corpus, benchmark_open_vocab, benchmark_sentence_variants
from eib_spellchecker.training.lexical import train_lexical_model



def _artifact(tmp_path: Path) -> Path:
    corpus = tmp_path / 'words.txt'
    corpus.write_text('jakon nete joi maria marina', encoding='utf-8')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        '''
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
safety_policy:
  enabled: true
''',
        encoding='utf-8',
    )
    return train_lexical_model(load_config(config_path), tmp_path / 'artifact')



def test_benchmark_clean_corpus(tmp_path: Path):
    artifact_dir = _artifact(tmp_path)
    clean_path = tmp_path / 'clean.txt'
    clean_path.write_text('jakon nete\nmaria joi\n', encoding='utf-8')
    result = benchmark_clean_corpus(artifact_dir, clean_path)
    assert result.changed_tokens == 0
    assert result.unchanged_token_rate == 1.0



def test_benchmark_sentence_variants(tmp_path: Path):
    artifact_dir = _artifact(tmp_path)
    path = tmp_path / 'variants.csv'
    path.write_text(
        'sentence,error_0,error_1\njakon nete,jakn nete,jakon nte\n',
        encoding='utf-8',
    )
    result = benchmark_sentence_variants(artifact_dir, path)
    assert result.total_examples == 2
    assert result.exact_match_accuracy >= 0.5



def test_benchmark_open_vocab(tmp_path: Path):
    artifact_dir = _artifact(tmp_path)
    path = tmp_path / 'pairs.tsv'
    path.write_text('jakn\tjakon\nMaria\tMaria\nmarina\tmarina\n', encoding='utf-8')
    result = benchmark_open_vocab(artifact_dir, path)
    payload = result.to_dict()
    bucket_names = {bucket['name'] for bucket in payload['buckets']}
    assert 'all' in bucket_names
    assert 'seen' in bucket_names
    assert 'proper_like' in bucket_names
