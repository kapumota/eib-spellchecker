# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

from pathlib import Path

from eib_spellchecker.config import AppConfig
from eib_spellchecker.data.loaders import build_frequency_table
from eib_spellchecker.modeling.policy import SafetyPolicy
from eib_spellchecker.modeling.subword import SubwordSpellChecker



def train_subword_model(config: AppConfig, output_dir: str | Path) -> Path:
    if not config.subword.enabled:
        raise ValueError("Activa subword.enabled=true en el config para entrenar este backend.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frequency_table = build_frequency_table(config)
    filtered = [
        (token, freq)
        for token, freq in frequency_table.most_common(config.subword.max_vocabulary_size)
        if freq >= config.subword.min_frequency
    ]
    SubwordSpellChecker.write_artifact(
        output_dir,
        language=config.language,
        vocabulary=[token for token, _ in filtered],
        frequencies={token: freq for token, freq in filtered},
        min_correction_length=config.subword.min_correction_length,
        max_candidates=config.subword.max_candidates,
        min_ngram=config.subword.min_ngram,
        max_ngram=config.subword.max_ngram,
        jaccard_weight=config.subword.jaccard_weight,
        edit_weight=config.subword.edit_weight,
        frequency_weight=config.subword.frequency_weight,
        score_threshold=config.subword.score_threshold,
        safety_policy=SafetyPolicy.from_mapping(config.safety_policy.model_dump()),
    )
    return output_dir
