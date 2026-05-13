from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field


class NormalizeConfig(BaseModel):
    lowercase: bool = True
    strip_accents: bool = False


class LexicalTrainingConfig(BaseModel):
    min_frequency: int = 1
    max_vocabulary_size: int = 50_000
    min_correction_length: int = 3
    similarity_threshold: float = Field(default=0.72, ge=0.0, le=1.0)


class Seq2SeqTrainingConfig(BaseModel):
    enabled: bool = False
    hidden_size: int = 256
    epochs: int = 10
    batch_size: int = 128
    validation_split: float = Field(default=0.1, ge=0.0, lt=0.5)
    error_rate: float = Field(default=0.3, ge=0.0, lt=1.0)
    reverse_input: bool = True
    sample_mode: Literal["argmax", "multinomial"] = "argmax"


class TorchRerankerConfig(BaseModel):
    enabled: bool = False
    embedding_dim: int = 48
    hidden_size: int = 96
    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = Field(default=0.001, gt=0.0)
    validation_split: float = Field(default=0.1, ge=0.0, lt=0.5)
    max_length: int = 24
    negatives_per_positive: int = 3
    candidate_limit: int = 8
    min_correction_length: int = 3
    similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    score_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    random_seed: int = 13
    device: Literal["auto", "cpu"] = "auto"


class SubwordTrainingConfig(BaseModel):
    enabled: bool = False
    min_frequency: int = 1
    max_vocabulary_size: int = 50_000
    min_correction_length: int = 3
    jaccard_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    edit_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    frequency_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    score_threshold: float = Field(default=0.58, ge=0.0, le=1.0)
    max_candidates: int = 32
    min_ngram: int = 2
    max_ngram: int = 4


class SafetyPolicyConfig(BaseModel):
    enabled: bool = True
    protect_title_case: bool = True
    protect_all_caps: bool = True
    proper_name_threshold: float = Field(default=0.86, ge=0.0, le=1.0)
    loanword_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    out_of_domain_threshold: float = Field(default=0.93, ge=0.0, le=1.0)
    abstain_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    max_unknown_char_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    min_known_char_coverage: float = Field(default=0.45, ge=0.0, le=1.0)
    score_boost_for_identity: float = Field(default=0.03, ge=0.0, le=1.0)
    loanword_chars: str = "fwzqxvk"


class DataConfig(BaseModel):
    corpus_files: List[Path]


class AppConfig(BaseModel):
    language: str
    alphabet: str = "latin"
    normalize: NormalizeConfig = NormalizeConfig()
    data: DataConfig
    lexical: LexicalTrainingConfig = LexicalTrainingConfig()
    seq2seq: Seq2SeqTrainingConfig = Seq2SeqTrainingConfig()
    torch_reranker: TorchRerankerConfig = TorchRerankerConfig()
    subword: SubwordTrainingConfig = SubwordTrainingConfig()
    safety_policy: SafetyPolicyConfig = SafetyPolicyConfig()



def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(payload)

    resolved_files = []
    for corpus_file in config.data.corpus_files:
        resolved_files.append((path.parent / corpus_file).resolve() if not corpus_file.is_absolute() else corpus_file)
    config.data.corpus_files = resolved_files
    return config
