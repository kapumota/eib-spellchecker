from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="PyTorch no está instalado, se omiten tests del backend torch")

from eib_spellchecker.config import load_config
from eib_spellchecker.inference.service import ArtifactSpellChecker
from eib_spellchecker.training.torch_reranker import train_torch_reranker_model


def test_train_torch_reranker_demo(tmp_path: Path):
    config = load_config('configs/demo_torch.yaml')
    config.torch_reranker.epochs = 4
    config.torch_reranker.hidden_size = 24
    config.torch_reranker.embedding_dim = 16
    config.torch_reranker.batch_size = 4
    artifact_dir = train_torch_reranker_model(
        config,
        tmp_path / 'artifact',
        pair_files=['examples/demo_pairs.tsv'],
    )
    checker = ArtifactSpellChecker.from_artifact_dir(artifact_dir)
    corrected = checker.correct_word('jakn')
    assert corrected == 'jakon'
    manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['backend'] == 'torch-hybrid-reranker'


def test_torch_reranker_suggest(tmp_path: Path):
    config = load_config('configs/demo_torch.yaml')
    config.torch_reranker.epochs = 4
    config.torch_reranker.hidden_size = 24
    config.torch_reranker.embedding_dim = 16
    config.torch_reranker.batch_size = 4
    artifact_dir = train_torch_reranker_model(
        config,
        tmp_path / 'artifact',
        pair_files=['examples/demo_pairs.tsv'],
    )
    checker = ArtifactSpellChecker.from_artifact_dir(artifact_dir)
    suggestions = checker.suggest('jakn', limit=3)
    assert 'jakon' in suggestions
