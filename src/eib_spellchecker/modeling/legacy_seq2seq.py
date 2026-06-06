# Codigo base de EIB Spellchecker.
# Implementa componentes principales del paquete, la API, la CLI y los backends.

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eib_spellchecker.modeling.base import TokenCorrection
from eib_spellchecker.modeling.manifest import ArtifactManifest
from eib_spellchecker.utils.text import clean_token, normalize_text

SOS = "\t"
EOS = "*"
DEFAULT_NOISE_ALPHABET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")
TOKEN_OR_SEPARATOR_RE = re.compile(r"(\w+|\W+)", flags=re.UNICODE)


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


def require_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "El backend legacy-seq2seq requiere TensorFlow. Instala el extra con: pip install -e .[tensorflow]"
        ) from exc
    return tf


@dataclass
class Seq2SeqMetadata:
    model_type: str
    language: str
    hidden_size: int
    max_length: int
    encoder_chars: list[str]
    decoder_chars: list[str]
    reverse_input: bool
    sample_mode: str
    weights_file: str


class CharacterTable:
    def __init__(self, chars: list[str]) -> None:
        self.chars = sorted(set(chars))
        self.char_to_index = {c: i for i, c in enumerate(self.chars)}
        self.index_to_char = {i: c for i, c in enumerate(self.chars)}
        self.size = len(self.chars)

    def encode(self, token: str, n_rows: int) -> np.ndarray:
        matrix = np.zeros((n_rows, self.size), dtype=np.float32)
        for i, char in enumerate(token):
            if char in self.char_to_index:
                matrix[i, self.char_to_index[char]] = 1.0
        return matrix

    def decode(self, tensor: np.ndarray, calc_argmax: bool = True) -> tuple[np.ndarray, str]:
        indices = tensor.argmax(axis=-1) if calc_argmax else tensor
        chars = "".join(self.index_to_char[int(index)] for index in indices)
        return indices, chars

    def sample_multinomial(self, preds: np.ndarray, temperature: float = 1.0) -> tuple[int, str]:
        preds = np.reshape(preds, len(self.chars)).astype(np.float64)
        preds = np.log(np.clip(preds, 1e-8, 1.0)) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        index = int(np.argmax(probs))
        return index, self.index_to_char[index]


def _inject_noise(token: str, error_rate: float, alphabet: list[str]) -> str:
    assert 0.0 <= error_rate < 1.0
    if len(token) < 3:
        return token
    rand = random.random()
    bucket = error_rate / 4.0
    if rand < bucket:
        i = random.randrange(len(token))
        return token[:i] + random.choice(alphabet) + token[i + 1 :]
    if rand < bucket * 2:
        i = random.randrange(len(token))
        return token[:i] + token[i + 1 :]
    if rand < bucket * 3:
        i = random.randrange(len(token))
        return token[:i] + random.choice(alphabet) + token[i:]
    if rand < bucket * 4:
        i = random.randrange(len(token) - 1)
        return token[:i] + token[i + 1] + token[i] + token[i + 2 :]
    return token


def transform_tokens(
    tokens: list[str],
    *,
    max_length: int,
    error_rate: float = 0.3,
    reverse_input: bool = True,
    alphabet: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    alphabet = alphabet or DEFAULT_NOISE_ALPHABET
    encoder_tokens: list[str] = []
    decoder_tokens: list[str] = []
    output_tokens: list[str] = []
    for token in tokens:
        noisy = _inject_noise(token, error_rate=error_rate, alphabet=alphabet)
        if reverse_input:
            noisy = noisy[::-1]
        noisy = noisy + EOS * (max_length - len(noisy))
        decoder = SOS + token
        decoder = decoder + EOS * (max_length - len(decoder))
        output = decoder[1:] + EOS * (max_length - len(decoder[1:]))
        encoder_tokens.append(noisy)
        decoder_tokens.append(decoder)
        output_tokens.append(output[:max_length])
    return encoder_tokens, decoder_tokens, output_tokens


def build_seq2seq_model(hidden_size: int, encoder_vocab_size: int, decoder_vocab_size: int):
    tf = require_tensorflow()
    keras = tf.keras

    encoder_inputs = keras.layers.Input(shape=(None, encoder_vocab_size), name="encoder_data")
    encoder_lstm_1 = keras.layers.LSTM(
        hidden_size,
        recurrent_dropout=0.2,
        return_sequences=True,
        return_state=False,
        name="lstm_encoder_1",
    )
    encoder_outputs = encoder_lstm_1(encoder_inputs)
    encoder_lstm_2 = keras.layers.LSTM(
        hidden_size,
        recurrent_dropout=0.2,
        return_sequences=False,
        return_state=True,
        name="lstm_encoder_2",
    )
    _, state_h, state_c = encoder_lstm_2(encoder_outputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.layers.Input(shape=(None, decoder_vocab_size), name="decoder_data")
    decoder_lstm = keras.layers.LSTM(
        hidden_size,
        dropout=0.2,
        return_sequences=True,
        return_state=True,
        name="lstm_decoder",
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(decoder_vocab_size, activation="softmax", name="softmax_decoder")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    encoder_model = keras.Model(inputs=encoder_inputs, outputs=encoder_states)

    decoder_state_input_h = keras.layers.Input(shape=(hidden_size,))
    decoder_state_input_c = keras.layers.Input(shape=(hidden_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        inputs=[decoder_inputs] + decoder_states_inputs,
        outputs=[decoder_outputs] + decoder_states,
    )
    return model, encoder_model, decoder_model


def restore_inference_models(weights_path: str | Path, hidden_size: int):
    tf = require_tensorflow()
    keras = tf.keras
    model = keras.models.load_model(weights_path)

    encoder_inputs = model.input[0]
    encoder_lstm_1 = model.get_layer("lstm_encoder_1")
    encoder_lstm_2 = model.get_layer("lstm_encoder_2")
    encoder_outputs = encoder_lstm_1(encoder_inputs)
    _, state_h, state_c = encoder_lstm_2(encoder_outputs)
    encoder_model = keras.Model(inputs=encoder_inputs, outputs=[state_h, state_c])

    decoder_inputs = model.input[1]
    decoder_state_input_h = keras.layers.Input(shape=(hidden_size,))
    decoder_state_input_c = keras.layers.Input(shape=(hidden_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.get_layer("lstm_decoder")
    decoder_dense = model.get_layer("softmax_decoder")
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        inputs=[decoder_inputs] + decoder_states_inputs,
        outputs=[decoder_outputs, state_h, state_c],
    )
    return encoder_model, decoder_model


class LegacySeq2SeqSpellChecker:
    def __init__(self, metadata: Seq2SeqMetadata, artifact_dir: str | Path) -> None:
        self.metadata = metadata
        self.language = metadata.language
        self.artifact_dir = Path(artifact_dir)
        self.encoder_table = CharacterTable(metadata.encoder_chars)
        self.decoder_table = CharacterTable(metadata.decoder_chars)
        self._inference_models: tuple[Any, Any] | None = None

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "LegacySeq2SeqSpellChecker":
        artifact_dir = Path(artifact_dir)
        metadata = Seq2SeqMetadata(**json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8")))
        return cls(metadata=metadata, artifact_dir=artifact_dir)

    @staticmethod
    def write_artifact(
        artifact_dir: str | Path,
        *,
        language: str,
        hidden_size: int,
        max_length: int,
        encoder_chars: list[str],
        decoder_chars: list[str],
        reverse_input: bool,
        sample_mode: str,
        weights_file: str,
    ) -> Path:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metadata = Seq2SeqMetadata(
            model_type="legacy-seq2seq",
            language=language,
            hidden_size=hidden_size,
            max_length=max_length,
            encoder_chars=encoder_chars,
            decoder_chars=decoder_chars,
            reverse_input=reverse_input,
            sample_mode=sample_mode,
            weights_file=weights_file,
        )
        (artifact_dir / "metadata.json").write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
        ArtifactManifest(
            artifact_version="2",
            backend="legacy-seq2seq",
            language=language,
            entrypoint="eib_spellchecker.modeling.legacy_seq2seq:LegacySeq2SeqSpellChecker",
            payload_file=weights_file,
            metadata_file="metadata.json",
        ).write(artifact_dir)
        return artifact_dir

    def available(self) -> bool:
        return tensorflow_available() and (self.artifact_dir / self.metadata.weights_file).exists()

    def _load_models(self):
        if self._inference_models is None:
            self._inference_models = restore_inference_models(
                self.artifact_dir / self.metadata.weights_file,
                hidden_size=self.metadata.hidden_size,
            )
        return self._inference_models

    def suggest(self, word: str, limit: int = 5) -> list[str]:
        # seq2seq no ofrece una lista natural de sugerencias; devolvemos una sola hipótesis si es posible.
        try:
            prediction = self.correct_word(word)
        except RuntimeError:
            return []
        return [prediction][:limit] if prediction else []

    def correct_word(self, word: str) -> str:
        cleaned = clean_token(word)
        if not cleaned:
            return word
        if not self.available():
            raise RuntimeError(
                "El artefacto seq2seq no está listo para inferencia. Verifica TensorFlow y el archivo de pesos."
            )
        encoder_model, decoder_model = self._load_models()
        normalized = normalize_text(cleaned, lowercase=False, strip_accents_flag=False)
        token = normalized[::-1] if self.metadata.reverse_input else normalized
        token = token[: self.metadata.max_length]
        token = token + EOS * (self.metadata.max_length - len(token))
        encoder_batch = np.zeros((1, self.metadata.max_length, self.encoder_table.size), dtype=np.float32)
        encoder_batch[0] = self.encoder_table.encode(token, self.metadata.max_length)
        states = encoder_model.predict(encoder_batch, verbose=0)
        target = np.zeros((1, 1, self.decoder_table.size), dtype=np.float32)
        target[0, 0, self.decoder_table.char_to_index[SOS]] = 1.0

        decoded = ""
        for _ in range(self.metadata.max_length):
            output_tokens, h, c = decoder_model.predict([target] + states, verbose=0)
            if self.metadata.sample_mode == "multinomial":
                next_index, next_char = self.decoder_table.sample_multinomial(output_tokens[0])
            else:
                next_index = int(output_tokens[0].argmax(axis=-1)[0])
                next_char = self.decoder_table.index_to_char[next_index]
            if next_char == EOS:
                break
            decoded += next_char
            target = np.zeros((1, 1, self.decoder_table.size), dtype=np.float32)
            target[0, 0, next_index] = 1.0
            states = [h, c]
        return decoded or word

    def correct_text(self, text: str) -> tuple[str, list[TokenCorrection]]:
        pieces: list[str] = []
        corrections: list[TokenCorrection] = []
        for piece in TOKEN_OR_SEPARATOR_RE.findall(text):
            if any(char.isalpha() for char in piece):
                corrected = self.correct_word(piece)
                corrections.append(TokenCorrection(original=piece, corrected=corrected, changed=piece != corrected))
                pieces.append(corrected)
            else:
                pieces.append(piece)
        return "".join(pieces), corrections
