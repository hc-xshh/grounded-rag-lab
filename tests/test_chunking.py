from raglab.chunking import chunk_markdown, expand_markdown_tables


def test_expand_markdown_tables_folds_header_into_rows():
    source = "\n".join(
        [
            "| Plan | Price |",
            "| --- | --- |",
            "| Growth | $199 per month |",
            "| Scale | $749 per month |",
        ]
    )
    expanded = expand_markdown_tables(source)
    lines = expanded.splitlines()
    assert lines[0] == "Table columns: Plan, Price."
    assert lines[1] == "Plan: Growth  |  Price: $199 per month"
    assert lines[2] == "Plan: Scale  |  Price: $749 per month"
    assert "---" not in expanded


def test_expand_markdown_tables_leaves_prose_alone():
    prose = "Invoices are issued on the first day of each billing period."
    assert expand_markdown_tables(prose) == prose


def test_chunk_markdown_keeps_heading_path_and_ids():
    text = "# Handbook\n\nIntro line.\n\n## Billing\n\nRefunds are issued within 30 days.\n"
    chunks = chunk_markdown(text, doc_id="handbook", doc_title="Handbook")
    assert [c.chunk_id for c in chunks] == ["handbook::000", "handbook::001"]
    assert chunks[0].heading == "Handbook"
    assert chunks[1].heading == "Handbook > Billing"
    assert "Refunds are issued within 30 days." in chunks[1].text
    assert chunks[1].indexed_text.startswith("Handbook > Billing")


def test_long_section_is_windowed_with_overlap():
    sentence = "This sentence is here to make the section long enough to be split. "
    text = "# Long\n\n" + sentence * 20
    chunks = chunk_markdown(text, doc_id="long", max_chars=300, overlap_chars=80)
    assert len(chunks) > 1
    assert all(len(c.text) <= 340 for c in chunks)  # max_chars plus one trailing sentence
    # overlap: the first sentence of window 2 also appears at the end of window 1
    assert chunks[0].text.split(".")[-2].strip() in chunks[1].text


def test_chunking_is_deterministic():
    text = "# A\n\nfirst paragraph here.\n\n## B\n\nsecond paragraph here.\n"
    first = chunk_markdown(text, doc_id="d")
    second = chunk_markdown(text, doc_id="d")
    assert [c.text for c in first] == [c.text for c in second]
