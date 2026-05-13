from pathlib import Path
from zipfile import ZipFile

from eib_spellchecker.catalog.excels import inventory_from_zip


def test_inventory_from_zip_counts(tmp_path: Path) -> None:
    zip_path = tmp_path / 'excels.zip'
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr('excels/common_data_ash.csv', 'Input,Output\na,b\n')
        zf.writestr('excels/df_ash.csv', 'sentence,error_0,error_1\nfoo,bar,baz\n')
        zf.writestr('excels/df_ash_sentences.csv', 'sentence\nfoo\n')
        zf.writestr('excels/Score.txt', 'Accuracy\n0.95 -- Ashaninka46\n')
    inventory = inventory_from_zip(zip_path)
    assert inventory.to_dict()['summary']['num_paired_datasets'] == 1
    assert inventory.to_dict()['summary']['num_sentence_variant_datasets'] == 1
    assert inventory.to_dict()['summary']['num_sentence_corpora'] == 1
    assert inventory.to_dict()['summary']['num_historical_score_entries'] == 1
