from raglab.text import content_tokens, coverage, normalise_fact, split_sentences, tokenize


def test_tokenize_stems_and_drops_stopwords():
    tokens = tokenize("The plans are processed and the sessions expire.")
    assert "plan" in tokens  # plural stemmed
    assert "process" in tokens  # -ed stemmed
    assert "session" in tokens
    assert "the" not in tokens and "are" not in tokens  # stopwords removed before stemming
    assert "doe" not in tokens  # "does" must not survive as an artificial term


def test_tokenize_strips_sentence_final_punctuation():
    assert tokenize("payment card.") == ["payment", "card"]
    assert "14-day" in tokenize("a 14-day free trial")


def test_cjk_bigrams_are_indexed():
    tokens = tokenize("退款政策")
    assert "退款" in tokens and "款政" in tokens


def test_split_sentences_undoes_hard_wrapping():
    text = "This is a long sentence that the author\nwrapped at column eighty. Next sentence."
    assert split_sentences(text) == [
        "This is a long sentence that the author wrapped at column eighty.",
        "Next sentence.",
    ]


def test_split_sentences_keeps_bullets_and_table_rows_separate():
    text = "Rules:\n\n- first rule\n- second rule\n\nPlan: Growth  |  Price: $199 per month"
    parts = split_sentences(text)
    assert parts[0] == "Rules:"
    assert parts[1] == "first rule"
    assert parts[2] == "second rule"
    assert parts[3] == "Plan: Growth  |  Price: $199 per month"


def test_coverage_and_normalise_fact():
    assert coverage(["refund", "window"], "the refund window is 30 days") == 1.0
    assert coverage(["refund", "quantum"], "the refund window is 30 days") == 0.5
    assert coverage([], "anything") == 0.0
    assert normalise_fact("$199 per month") == "199 per month"


def test_content_tokens_drop_single_characters():
    assert content_tokens("a b cd") == ["cd"]
