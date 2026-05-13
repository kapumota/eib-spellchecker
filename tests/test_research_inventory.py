import json
from pathlib import Path
from zipfile import ZipFile

import pytest
h5py = pytest.importorskip("h5py")

from eib_spellchecker.catalog.research import inventory_research_runs, benchmark_research_runs, summarize_research_benchmarks
from eib_spellchecker.research.packaging import extract_research_runs


def _write_fake_h5(path: Path) -> None:
    model_config = {
        'class_name': 'Model',
        'config': {
            'layers': [
                {'name': 'input_1', 'class_name': 'InputLayer', 'config': {'batch_input_shape': [None, 46]}},
                {'name': 'embedding_1', 'class_name': 'Embedding', 'config': {'input_dim': 27}},
                {'name': 'embedding_2', 'class_name': 'Embedding', 'config': {'input_dim': 23}},
                {'name': 'lstm_1', 'class_name': 'LSTM', 'config': {'units': 64}},
                {'name': 'time_distributed_2', 'class_name': 'TimeDistributed', 'config': {'layer': {'config': {'units': 23, 'activation': 'softmax'}}}},
            ]
        }
    }
    with h5py.File(path, 'w') as handle:
        handle.attrs['keras_version'] = '2.3.1'
        handle.attrs['backend'] = 'tensorflow'
        handle.attrs['model_config'] = json.dumps(model_config)
        handle.create_group('model_weights')


def test_inventory_and_benchmark_research_runs(tmp_path: Path) -> None:
    h5_path = tmp_path / 'model.h5'
    _write_fake_h5(h5_path)
    zip_path = tmp_path / 'augmentation.zip'
    with ZipFile(zip_path, 'w') as zf:
        zf.write(h5_path, arcname='aumento_datos/common_data/seq2seq+atencion/Ashaninka46/model_ash_46_kd.h5')
        zf.writestr('aumento_datos/common_data/seq2seq+atencion/Ashaninka46/candidatos.txt', 'hola\nadios\n')
        zf.writestr('aumento_datos/common_data/seq2seq+atencion/Ashaninka46/referencia.txt', 'hola\na dios\n')

    inventory = inventory_research_runs(augmentation_zip=zip_path)
    payload = inventory.to_dict()
    assert payload['summary']['num_runs'] == 1
    run = payload['runs'][0]
    assert run['model_summary']['input_length'] == 46
    assert run['model_summary']['lstm_units'] == 64

    benchmarks = benchmark_research_runs(augmentation_zip=zip_path)
    summary = summarize_research_benchmarks(benchmarks)
    assert summary['summary']['num_runs'] == 1
    assert summary['benchmarks'][0]['total_examples'] == 2
    assert summary['benchmarks'][0]['exact_match_accuracy'] == 0.5


def test_extract_research_runs(tmp_path: Path) -> None:
    h5_path = tmp_path / 'model.h5'
    _write_fake_h5(h5_path)
    zip_path = tmp_path / 'subword.zip'
    with ZipFile(zip_path, 'w') as zf:
        zf.write(h5_path, arcname='subword_segmentacion/bpe/common_data/asha/5k/model_asha_46_bpe_5k_kd.h5')
        zf.writestr('subword_segmentacion/bpe/common_data/asha/5k/candidatos.txt', 'a\n')
        zf.writestr('subword_segmentacion/bpe/common_data/asha/5k/referencia.txt', 'a\n')
        zf.writestr('subword_segmentacion/bpe/common_data/asha/5k/x_test_5k.bpe', 'x\n')
        zf.writestr('subword_segmentacion/bpe/common_data/asha/5k/y_test_5k.bpe', 'y\n')
        zf.writestr('subword_segmentacion/bpe/common_data/asha/5k/asha_5k.vocab', 'aa\n')

    created = extract_research_runs(tmp_path / 'model_zoo', subword_zip=zip_path)
    assert len(created) == 1
    metadata = json.loads((created[0] / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['family'] == 'subword-bpe'
    assert (created[0] / metadata['extracted_files']['weights']).exists()
