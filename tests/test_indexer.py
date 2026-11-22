from support_assistant.indexer import chunk_markdown

DOC = """# Guide

Intro text.

## No signal

Restart the phone.

Check airplane mode.

### SIM

Reseat the SIM card.

## Slow data

Check data usage.
"""


def test_chunks_carry_heading_path():
    chunks = chunk_markdown(DOC)
    paths = {c.heading_path for c in chunks}
    assert {"Guide", "Guide > No signal", "Guide > No signal > SIM", "Guide > Slow data"} <= paths


def test_digest_is_stable_and_content_sensitive():
    a, b = chunk_markdown(DOC), chunk_markdown(DOC)
    assert [c.digest for c in a] == [c.digest for c in b]
    changed = chunk_markdown(DOC.replace("Restart the phone.", "Reboot the phone."))
    assert {c.digest for c in a} != {c.digest for c in changed}


def test_long_sections_are_split():
    text = "# T\n\n" + "\n\n".join("word " * 100 for _ in range(10))
    assert len(chunk_markdown(text, max_chars=600)) > 1
