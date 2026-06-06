# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

from pathlib import Path

from eib_spellchecker.config import AppConfig
from eib_spellchecker.data.loaders import build_frequency_table
from eib_spellchecker.modeling.lexical import LexicalSpellChecker
from eib_spellchecker.modeling.policy import SafetyPolicy



def train_lexical_model(config: AppConfig, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frequency_table = build_frequency_table(config)
    filtered = [
        (token, freq)
        for token, freq in frequency_table.most_common(config.lexical.max_vocabulary_size)
        if freq >= config.lexical.min_frequency
    ]
    LexicalSpellChecker.write_artifact(
        output_dir,
        language=config.language,
        min_correction_length=config.lexical.min_correction_length,
        similarity_threshold=config.lexical.similarity_threshold,
        vocabulary=[token for token, _ in filtered],
        frequencies={token: freq for token, freq in filtered},
        safety_policy=SafetyPolicy.from_mapping(config.safety_policy.model_dump()),
    )
    return output_dir
