# Script auxiliar de EIB Spellchecker.
# Mantiene tareas de importacion, migracion o mantenimiento del proyecto.

from __future__ import annotations

import argparse

from eib_spellchecker.benchmarks.reporting import write_report
from eib_spellchecker.catalog.legacy import inventory_from_zips


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', default=None)
    parser.add_argument('--augmentation-zip', default=None)
    parser.add_argument('--subword-zip', default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    inventory = inventory_from_zips(
        dataset_zip=args.dataset_zip,
        augmentation_zip=args.augmentation_zip,
        subword_zip=args.subword_zip,
    )
    write_report(args.output, inventory.to_dict())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
