from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


KEEP_PREFIXES = [
    "eib-spell-checking-master/Datos-sinteticos/",
    "eib-spell-checking-master/Modelo-corrector/",
    "eib-spell-checking-master/baseline/",
    "eib-spell-checking-master/lang_data/",
    "eib-spell-checking-master/notebooks/",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrae el repositorio legado a una carpeta local.")
    parser.add_argument("--zip", required=True, help="Ruta al ZIP original.")
    parser.add_argument("--output-dir", required=True, help="Carpeta destino.")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if any(member.startswith(prefix) for prefix in KEEP_PREFIXES):
                archive.extract(member, path=output_dir)

    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
