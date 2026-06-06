# Prueba automatizada de EIB Spellchecker.
# Valida comportamiento funcional, regresiones o integracion del proyecto.

from pathlib import Path

from eib_spellchecker.evaluation.metrics import benchmark_artifact
from eib_spellchecker.modeling.lexical import LexicalSpellChecker


def test_benchmark_csv(tmp_path: Path) -> None:
    artifact_dir = tmp_path / 'artifact'
    LexicalSpellChecker.write_artifact(
        artifact_dir,
        language='demo',
        min_correction_length=3,
        similarity_threshold=0.7,
        vocabulary=['hola', 'mundo'],
        frequencies={'hola': 10, 'mundo': 5},
    )
    dataset = tmp_path / 'pairs.csv'
    dataset.write_text('Input,Output\nhpla hola,hola hola\n', encoding='utf-8')
    result = benchmark_artifact(artifact_dir, dataset)
    assert result.total_examples == 1
    assert result.token_accuracy >= 0.5
