# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict
from pathlib import Path

from eib_spellchecker.catalog.research import ResearchRun, inventory_research_runs


ASSET_NAMES = ['weights_path', 'candidates_path', 'reference_path', 'test_input_path', 'test_output_path', 'vocab_path']


def _extract_member(zf: zipfile.ZipFile, name: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(name, 'r') as src, target.open('wb') as dst:
        dst.write(src.read())
    return target.name


def extract_research_runs(output_root: str | Path, *, augmentation_zip: str | Path | None = None, subword_zip: str | Path | None = None) -> list[Path]:
    output_root = Path(output_root)
    inventory = inventory_research_runs(augmentation_zip=augmentation_zip, subword_zip=subword_zip)
    zip_map = {}
    for path in [augmentation_zip, subword_zip]:
        if path:
            zip_map[Path(path).name] = zipfile.ZipFile(path)

    created: list[Path] = []
    try:
        for run in inventory.runs:
            run_dir = output_root / run.slug
            run_dir.mkdir(parents=True, exist_ok=True)
            extracted_files = {}
            zf = zip_map[run.source_zip]
            for attr in ASSET_NAMES:
                source_name = getattr(run, attr)
                if source_name:
                    extracted_files[attr.replace('_path', '')] = _extract_member(zf, source_name, run_dir / Path(source_name).name)
            metadata = asdict(run)
            metadata['extracted_files'] = extracted_files
            (run_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            created.append(run_dir)
    finally:
        for zf in zip_map.values():
            zf.close()

    return created
