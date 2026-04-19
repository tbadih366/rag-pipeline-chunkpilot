from app.chunking import chunk_text


def test_chunk_text_overlap_behavior() -> None:
    text = " ".join([f"w{i}" for i in range(1, 31)])
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks) == 4
    assert chunks[0].split()[0] == "w1"
    assert chunks[1].split()[0] == "w9"

