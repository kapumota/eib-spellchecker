from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from eib_spellchecker.config import AppConfig
from eib_spellchecker.data.loaders import build_frequency_table
from eib_spellchecker.data.pairs import load_pairs_auto
from eib_spellchecker.modeling.lexical import LexicalSpellChecker
from eib_spellchecker.modeling.policy import SafetyPolicy
from eib_spellchecker.modeling.torch_reranker import PAD, TorchHybridSpellChecker, build_char_vocab, require_torch
from eib_spellchecker.utils.text import normalize_text


def _normalize_pair(noisy: str, gold: str, config: AppConfig) -> tuple[str, str]:
    return (
        normalize_text(noisy, lowercase=config.normalize.lowercase, strip_accents_flag=config.normalize.strip_accents),
        normalize_text(gold, lowercase=config.normalize.lowercase, strip_accents_flag=config.normalize.strip_accents),
    )


class PairDataset:
    def __init__(self, examples: list[tuple[str, str, int]], char_to_index: dict[str, int], max_length: int) -> None:
        self.examples = examples
        self.char_to_index = char_to_index
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, word: str) -> tuple[list[int], int]:
        ids = [self.char_to_index.get(char, self.char_to_index['<unk>']) for char in word[: self.max_length]]
        length = max(len(ids), 1)
        ids.extend([self.char_to_index[PAD]] * (self.max_length - len(ids)))
        return ids, length

    def __getitem__(self, index: int):
        noisy, candidate, label = self.examples[index]
        noisy_ids, noisy_length = self._encode(noisy)
        cand_ids, cand_length = self._encode(candidate)
        return noisy_ids, noisy_length, cand_ids, cand_length, label



def _collate(batch):
    torch, _, _, _ = require_torch()
    noisy_ids, noisy_lengths, cand_ids, cand_lengths, labels = zip(*batch)
    return (
        torch.tensor(noisy_ids, dtype=torch.long),
        torch.tensor(noisy_lengths, dtype=torch.long),
        torch.tensor(cand_ids, dtype=torch.long),
        torch.tensor(cand_lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )



def _iter_pairs(pair_files: Iterable[str | Path], *, noisy_column: str = 'Input', gold_column: str = 'Output') -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for path in pair_files:
        pairs.extend(load_pairs_auto(path, noisy_column=noisy_column, gold_column=gold_column))
    return pairs



def _sample_negative(noisy: str, gold: str, lexical: LexicalSpellChecker, vocabulary: list[str], rng: random.Random) -> list[str]:
    candidates = [candidate for candidate in lexical.suggest(noisy, limit=12) if candidate != gold]
    if noisy != gold and noisy not in candidates:
        candidates.append(noisy)
    if len(candidates) < 4 and vocabulary:
        similar_length = [token for token in vocabulary if token != gold and abs(len(token) - len(noisy)) <= 2]
        rng.shuffle(similar_length)
        candidates.extend(similar_length[: 4 - len(candidates)])
    return list(dict.fromkeys(candidates))



def _build_examples(
    pairs: list[tuple[str, str]],
    lexical: LexicalSpellChecker,
    vocabulary: list[str],
    *,
    negatives_per_positive: int,
    rng: random.Random,
) -> list[tuple[str, str, int]]:
    examples: list[tuple[str, str, int]] = []
    for noisy, gold in pairs:
        if not noisy or not gold:
            continue
        examples.append((noisy, gold, 1))
        negatives = _sample_negative(noisy, gold, lexical, vocabulary, rng)
        rng.shuffle(negatives)
        for candidate in negatives[:negatives_per_positive]:
            examples.append((noisy, candidate, 0))
    rng.shuffle(examples)
    return examples



def _split_train_val(examples: list[tuple[str, str, int]], validation_split: float) -> tuple[list, list]:
    if not examples:
        return [], []
    cut = max(1, int(len(examples) * (1.0 - validation_split)))
    if cut >= len(examples):
        return examples, []
    return examples[:cut], examples[cut:]



def train_torch_reranker_model(
    config: AppConfig,
    output_dir: str | Path,
    *,
    pair_files: list[str | Path],
    noisy_column: str = 'Input',
    gold_column: str = 'Output',
    limit: int | None = None,
) -> Path:
    if not config.torch_reranker.enabled:
        raise ValueError('Activa torch_reranker.enabled=true en el config para entrenar este backend.')
    torch, _, F, _ = require_torch()
    from torch.utils.data import DataLoader

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.torch_reranker.random_seed)
    torch.manual_seed(config.torch_reranker.random_seed)

    frequency_table: Counter[str] = build_frequency_table(config)
    filtered = [
        (token, freq)
        for token, freq in frequency_table.most_common(config.lexical.max_vocabulary_size)
        if freq >= config.lexical.min_frequency
    ]
    vocabulary = [token for token, _ in filtered]
    frequencies = {token: freq for token, freq in filtered}

    raw_pairs = _iter_pairs(pair_files, noisy_column=noisy_column, gold_column=gold_column)
    if limit is not None:
        raw_pairs = raw_pairs[:limit]
    normalized_pairs = []
    for noisy, gold in raw_pairs:
        noisy_n, gold_n = _normalize_pair(noisy, gold, config)
        if noisy_n and gold_n:
            normalized_pairs.append((noisy_n, gold_n))
            if gold_n not in frequencies:
                frequencies[gold_n] = 1
                vocabulary.append(gold_n)
    vocabulary = sorted(set(vocabulary))
    if not normalized_pairs:
        raise ValueError('No se encontraron pares válidos para entrenar el backend torch.')

    lexical = LexicalSpellChecker(
        vocabulary=vocabulary,
        frequencies=frequencies,
        language=config.language,
        min_correction_length=config.torch_reranker.min_correction_length,
        similarity_threshold=config.torch_reranker.similarity_threshold,
    )
    examples = _build_examples(
        normalized_pairs,
        lexical,
        vocabulary,
        negatives_per_positive=config.torch_reranker.negatives_per_positive,
        rng=rng,
    )
    train_examples, val_examples = _split_train_val(examples, config.torch_reranker.validation_split)

    chars = build_char_vocab([token for pair in normalized_pairs for token in pair] + vocabulary)
    char_to_index = {char: index for index, char in enumerate(chars)}
    train_dataset = PairDataset(train_examples, char_to_index, config.torch_reranker.max_length)
    val_dataset = PairDataset(val_examples, char_to_index, config.torch_reranker.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.torch_reranker.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=config.torch_reranker.batch_size, shuffle=False, collate_fn=_collate) if val_examples else None

    from eib_spellchecker.modeling.torch_reranker import CharPairRerankerModelBase
    device_name = 'cpu'
    if config.torch_reranker.device == 'auto' and torch.cuda.is_available():
        device_name = 'cuda'
    device = torch.device(device_name)
    model = CharPairRerankerModelBase(
        embedding_dim=config.torch_reranker.embedding_dim,
        hidden_size=config.torch_reranker.hidden_size,
        vocab_size=len(chars),
    ).module.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.torch_reranker.learning_rate)

    history: list[dict] = []
    best_state = None
    best_val_loss = float('inf')

    for epoch in range(1, config.torch_reranker.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for noisy_ids, noisy_lengths, cand_ids, cand_lengths, labels in train_loader:
            noisy_ids = noisy_ids.to(device)
            noisy_lengths = noisy_lengths.to(device)
            cand_ids = cand_ids.to(device)
            cand_lengths = cand_lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(noisy_ids, noisy_lengths, cand_ids, cand_lengths)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(labels)
            train_count += len(labels)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            loss_sum = 0.0
            count = 0
            with torch.no_grad():
                for noisy_ids, noisy_lengths, cand_ids, cand_lengths, labels in val_loader:
                    noisy_ids = noisy_ids.to(device)
                    noisy_lengths = noisy_lengths.to(device)
                    cand_ids = cand_ids.to(device)
                    cand_lengths = cand_lengths.to(device)
                    labels = labels.to(device)
                    logits = model(noisy_ids, noisy_lengths, cand_ids, cand_lengths)
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                    loss_sum += float(loss.item()) * len(labels)
                    count += len(labels)
            val_loss = loss_sum / max(count, 1)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    weights_file = 'model.pt'
    torch.save(best_state, output_dir / weights_file)

    TorchHybridSpellChecker.write_artifact(
        output_dir,
        language=config.language,
        max_length=config.torch_reranker.max_length,
        chars=chars,
        embedding_dim=config.torch_reranker.embedding_dim,
        hidden_size=config.torch_reranker.hidden_size,
        candidate_limit=config.torch_reranker.candidate_limit,
        min_correction_length=config.torch_reranker.min_correction_length,
        similarity_threshold=config.torch_reranker.similarity_threshold,
        score_threshold=config.torch_reranker.score_threshold,
        weights_file=weights_file,
        vocabulary_file='vocabulary.json',
        vocabulary=vocabulary,
        frequencies=frequencies,
        safety_policy=SafetyPolicy.from_mapping(config.safety_policy.model_dump()),
    )
    (output_dir / 'training_report.json').write_text(
        json.dumps(
            {
                'language': config.language,
                'pairs': len(normalized_pairs),
                'examples': len(examples),
                'train_examples': len(train_examples),
                'validation_examples': len(val_examples),
                'history': history,
                'best_val_loss': best_val_loss,
                'pair_files': [str(path) for path in pair_files],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    return output_dir
