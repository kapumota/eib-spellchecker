# Prueba automatizada de EIB Spellchecker.
# Valida comportamiento funcional, regresiones o integracion del proyecto.

from pathlib import Path

from eib_spellchecker.data.pairs import load_pairs_auto, write_pairs_tsv


def test_load_pairs_csv(tmp_path: Path) -> None:
    path = tmp_path / 'pairs.csv'
    path.write_text('Input,Output\nerr,ok\nfoo,bar\n', encoding='utf-8')
    assert load_pairs_auto(path) == [('err', 'ok'), ('foo', 'bar')]


def test_write_pairs_tsv(tmp_path: Path) -> None:
    path = tmp_path / 'pairs.tsv'
    write_pairs_tsv(path, [('a', 'b')])
    assert path.read_text(encoding='utf-8').strip() == 'a\tb'
