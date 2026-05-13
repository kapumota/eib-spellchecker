from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArtifactManifest:
    artifact_version: str
    backend: str
    language: str
    entrypoint: str
    payload_file: str
    metadata_file: str | None = None

    def write(self, artifact_dir: str | Path) -> Path:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "manifest.json"
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def load_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    return json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
