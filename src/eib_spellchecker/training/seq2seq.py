# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from eib_spellchecker.config import AppConfig
from eib_spellchecker.data.loaders import iter_corpus_tokens
from eib_spellchecker.modeling.legacy_seq2seq import (
    DEFAULT_NOISE_ALPHABET,
    CharacterTable,
    LegacySeq2SeqSpellChecker,
    build_seq2seq_model,
    transform_tokens,
)


def build_seq2seq_metadata_from_config(config: AppConfig) -> dict:
    vocabulary = sorted(set(iter_corpus_tokens(config)))
    if not vocabulary:
        raise ValueError("No se encontraron tokens en el corpus.")
    max_length = max(len(token) for token in vocabulary) + 2
    encoder_tokens, decoder_tokens, _ = transform_tokens(
        vocabulary,
        max_length=max_length,
        error_rate=config.seq2seq.error_rate,
        reverse_input=config.seq2seq.reverse_input,
        alphabet=DEFAULT_NOISE_ALPHABET,
    )
    encoder_chars = sorted(set("".join(encoder_tokens)))
    decoder_chars = sorted(set("".join(decoder_tokens)))
    return {
        "vocabulary": vocabulary,
        "max_length": max_length,
        "encoder_chars": encoder_chars,
        "decoder_chars": decoder_chars,
    }


def package_legacy_seq2seq(config: AppConfig, weights_path: str | Path, output_dir: str | Path) -> Path:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"No existe el archivo de pesos: {weights_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_seq2seq_metadata_from_config(config)
    target_weights = output_dir / weights_path.name
    if weights_path.resolve() != target_weights.resolve():
        shutil.copy2(weights_path, target_weights)

    LegacySeq2SeqSpellChecker.write_artifact(
        output_dir,
        language=config.language,
        hidden_size=config.seq2seq.hidden_size,
        max_length=metadata["max_length"],
        encoder_chars=metadata["encoder_chars"],
        decoder_chars=metadata["decoder_chars"],
        reverse_input=config.seq2seq.reverse_input,
        sample_mode=config.seq2seq.sample_mode,
        weights_file=target_weights.name,
    )
    return output_dir


def train_seq2seq_model(config: AppConfig, output_dir: str | Path) -> Path:
    if not config.seq2seq.enabled:
        raise ValueError("Activa seq2seq.enabled=true en el config para entrenar este backend.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_seq2seq_metadata_from_config(config)
    vocabulary = metadata["vocabulary"]
    max_length = metadata["max_length"]
    encoder_chars = metadata["encoder_chars"]
    decoder_chars = metadata["decoder_chars"]

    encoder_tokens, decoder_tokens, output_tokens = transform_tokens(
        vocabulary,
        max_length=max_length,
        error_rate=config.seq2seq.error_rate,
        reverse_input=config.seq2seq.reverse_input,
        alphabet=DEFAULT_NOISE_ALPHABET,
    )

    encoder_table = CharacterTable(encoder_chars)
    decoder_table = CharacterTable(decoder_chars)

    n = len(vocabulary)
    encoder_input_data = np.zeros((n, max_length, encoder_table.size), dtype=np.float32)
    decoder_input_data = np.zeros((n, max_length, decoder_table.size), dtype=np.float32)
    decoder_target_data = np.zeros((n, max_length, decoder_table.size), dtype=np.float32)

    for i, token in enumerate(encoder_tokens):
        encoder_input_data[i] = encoder_table.encode(token, max_length)
    for i, token in enumerate(decoder_tokens):
        decoder_input_data[i] = decoder_table.encode(token, max_length)
    for i, token in enumerate(output_tokens):
        decoder_target_data[i] = decoder_table.encode(token, max_length)

    model, _, _ = build_seq2seq_model(
        hidden_size=config.seq2seq.hidden_size,
        encoder_vocab_size=encoder_table.size,
        decoder_vocab_size=decoder_table.size,
    )
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=config.seq2seq.batch_size,
        epochs=config.seq2seq.epochs,
        validation_split=config.seq2seq.validation_split,
        verbose=1,
    )
    weights_path = output_dir / "model.keras"
    model.save(weights_path)

    LegacySeq2SeqSpellChecker.write_artifact(
        output_dir,
        language=config.language,
        hidden_size=config.seq2seq.hidden_size,
        max_length=max_length,
        encoder_chars=encoder_chars,
        decoder_chars=decoder_chars,
        reverse_input=config.seq2seq.reverse_input,
        sample_mode=config.seq2seq.sample_mode,
        weights_file=weights_path.name,
    )
    return output_dir
