import zipfile
from pathlib import Path

from eib_spellchecker.catalog.legacy import inventory_from_zips


def make_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, 'w') as zf:
        for name, content in members.items():
            zf.writestr(name, content)


def test_inventory_from_zips(tmp_path: Path) -> None:
    dataset_zip = tmp_path / 'dataset.zip'
    augmentation_zip = tmp_path / 'augmentation.zip'
    make_zip(dataset_zip, {'dataset/ash.txt': 'uno dos\nres tres\n'})
    make_zip(augmentation_zip, {
        'aumento_datos/common_data/seq2seq+atencion/Ashaninka46/common_data_ash.csv': 'Input,Output\na,b\n',
        'aumento_datos/common_data/seq2seq+atencion/Ashaninka46/model_ash_46_kd.h5': 'weights',
    })
    inventory = inventory_from_zips(dataset_zip=dataset_zip, augmentation_zip=augmentation_zip)
    payload = inventory.to_dict()
    assert payload['summary']['num_corpora'] == 1
    assert payload['summary']['num_paired_datasets'] == 1
    assert payload['summary']['num_models'] == 1
