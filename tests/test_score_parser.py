from eib_spellchecker.catalog.excels import parse_score_text, summarize_scores


def test_parse_score_text_extracts_entries() -> None:
    entries = parse_score_text(
        'CharacTER\nYine46\nCarlo (keyboard) = 0.17\nBLEU\nYine46\nCarlo (keyboard) = 17.3\nAccuracy\n0.9606 -- yine46\n'
    )
    assert len(entries) == 3
    summary = summarize_scores(entries)
    assert summary['summary']['num_entries'] == 3
    assert 'Accuracy' in summary['summary']['metrics']
