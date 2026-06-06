# Prueba automatizada de EIB Spellchecker.
# Valida comportamiento funcional, regresiones o integracion del proyecto.

from pathlib import Path

from eib_spellchecker.benchmarks.suite import benchmark_suite
from eib_spellchecker.modeling.lexical import LexicalSpellChecker


def test_benchmark_suite(tmp_path: Path) -> None:
    artifact_root = tmp_path / 'artifacts'
    artifact_dir = artifact_root / 'ash'
    LexicalSpellChecker.write_artifact(
        artifact_dir,
        language='ash',
        min_correction_length=2,
        similarity_threshold=0.7,
        vocabulary=['hola', 'mundo'],
        frequencies={'hola': 10, 'mundo': 5},
    )
    datasets_root = tmp_path / 'datasets'
    datasets_root.mkdir()
    (datasets_root / 'common_data_ash.csv').write_text('Input,Output\nhpla hola,hola hola\n', encoding='utf-8')
    suite = benchmark_suite(artifact_root, datasets_root, limit=1)
    assert suite.summary['num_datasets'] == 1
