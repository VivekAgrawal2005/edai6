from app.preprocessing.text_cleaner import clean_text, tokenize


def test_clean_text_removes_html_and_punctuation() -> None:
    text = "<p>Hello, World!</p>"
    assert clean_text(text) == "hello world"


def test_tokenize_has_words() -> None:
    tokens = tokenize("Please schedule a meeting tomorrow")
    assert "meeting" in [token.lower() for token in tokens]
