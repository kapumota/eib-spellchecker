# Prueba automatizada de EIB Spellchecker.
# Valida comportamiento funcional, regresiones o integracion del proyecto.

from __future__ import annotations

from eib_spellchecker.config import load_config
from eib_spellchecker.inference.service import ArtifactSpellChecker
from eib_spellchecker.training.subword import train_subword_model



def test_train_subword_demo(tmp_path):
    config = load_config('configs/demo_subword.yaml')
    artifact_dir = train_subword_model(config, tmp_path / 'artifact')
    checker = ArtifactSpellChecker.from_artifact_dir(artifact_dir)
    assert checker.correct_word('jakn') == 'jakon'
    token = checker.correct_token('Jakn')
    assert token.corrected == 'Jakn'
    assert token.reason in {'protected-proper-name', 'abstained-low-margin', 'identity-best', 'below-threshold'}
