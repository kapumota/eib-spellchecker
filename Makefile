# Makefile para eib_spellchecker

.PHONY: install install-modern test run-api demo inventory train-demo-torch benchmark-demo

install:
	python -m pip install -e .[dev]

install-modern:
	python -m pip install -e .[dev,demo,torch]

test:
	pytest

run-api:
	uvicorn eib_spellchecker.api:app --reload

demo:
	python -m eib_spellchecker.cli gradio-demo --artifact-dir $(ARTIFACT)

inventory:
	python -m eib_spellchecker.cli inventory --dataset-zip $(DATASET_ZIP) --augmentation-zip $(AUGMENTATION_ZIP) --subword-zip $(SUBWORD_ZIP) --output reports/inventory.json

train-demo-torch:
	python -m eib_spellchecker.cli train-torch-reranker --config configs/demo_torch.yaml --pairs examples/demo_pairs.tsv --output-dir artifacts/torch/demo

benchmark-demo:
	python -m eib_spellchecker.cli benchmark-csv --artifact-dir $(ARTIFACT) --dataset $(DATASET) --limit 100
