"""
==========================================================
Text Cleaner — NLP Preprocessing Pipeline
==========================================================
Reusable text preprocessing utilities for all ML models.
Uses NLTK for tokenization, stopword removal, and lemmatization.
"""

import re
import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# NLTK resource bootstrap (called once at app startup via main.py)
# ---------------------------------------------------------------------------
def download_nltk_data() -> None:
    """Download required NLTK data packages if not already present."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    lookup_map = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for resource in resources:
        try:
            nltk.data.find(lookup_map[resource])
        except LookupError:
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Core cleaning functions
# ---------------------------------------------------------------------------

def remove_html_tags(text: str) -> str:
    """Strip all HTML/XML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove URLs (http, https, www)."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_email_addresses(text: str) -> str:
    """Remove email addresses from text."""
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove standalone numbers (keeps alphanumeric tokens)."""
    return re.sub(r"\b\d+\b", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_email_headers(text: str) -> str:
    """Remove common email header lines (From:, To:, Subject:, etc.)."""
    # Remove lines that start with common header prefixes
    header_pattern = re.compile(
        r"^(From|To|Cc|Bcc|Subject|Date|Sent|Received|"
        r"Message-ID|Content-Type|MIME-Version|X-\w+|"
        r"Content-Transfer-Encoding|Return-Path|Delivered-To):\s*.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    return header_pattern.sub("", text)


def remove_forwarded_markers(text: str) -> str:
    """Remove forwarded/reply markers and quoted text indicators."""
    # Remove lines starting with >
    text = re.sub(r"^>+.*$", "", text, flags=re.MULTILINE)
    # Remove common forward/reply markers
    text = re.sub(
        r"-{2,}\s*(Original Message|Forwarded by|Forwarded message)\s*-{2,}",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text


# ---------------------------------------------------------------------------
# Tokenization & linguistic processing
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Tokenize text into a list of words using NLTK word_tokenize."""
    try:
        return word_tokenize(text)
    except Exception:
        # Fallback to simple split if NLTK tokenizer fails
        return text.split()


def remove_stopwords(tokens: List[str], extra_stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Remove English stopwords from token list.
    Optionally provide additional domain-specific stopwords.
    """
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
        }
    if extra_stopwords:
        stop_words.update(extra_stopwords)
    return [token for token in tokens if token.lower() not in stop_words]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatize tokens using WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    try:
        return [lemmatizer.lemmatize(token) for token in tokens]
    except LookupError:
        return tokens


def stem(tokens: List[str]) -> List[str]:
    """Stem tokens using Porter stemming as a lightweight fallback."""
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def clean_text(
    text: str,
    lowercase: bool = True,
    strip_html: bool = True,
    strip_urls: bool = True,
    strip_emails: bool = True,
    strip_punctuation: bool = True,
    strip_numbers: bool = False,
    strip_headers: bool = True,
    strip_forwarded: bool = True,
) -> str:
    """
    Apply a configurable sequence of text cleaning steps.

    Parameters
    ----------
    text : str
        Raw input text.
    lowercase : bool
        Convert to lowercase.
    strip_html : bool
        Remove HTML tags.
    strip_urls : bool
        Remove URLs.
    strip_emails : bool
        Remove email addresses.
    strip_punctuation : bool
        Remove punctuation.
    strip_numbers : bool
        Remove standalone numbers.
    strip_headers : bool
        Remove email header lines.
    strip_forwarded : bool
        Remove forwarded/reply markers.

    Returns
    -------
    str
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""

    if strip_headers:
        text = remove_email_headers(text)
    if strip_forwarded:
        text = remove_forwarded_markers(text)
    if strip_html:
        text = remove_html_tags(text)
    if strip_urls:
        text = remove_urls(text)
    if strip_emails:
        text = remove_email_addresses(text)
    if lowercase:
        text = text.lower()
    if strip_punctuation:
        text = remove_punctuation(text)
    if strip_numbers:
        text = remove_numbers(text)

    text = normalize_whitespace(text)
    return text


def preprocess_pipeline(
    text: str,
    do_lemmatize: bool = True,
    do_remove_stopwords: bool = True,
    return_tokens: bool = False,
    use_stemming: bool = False,
) -> str | List[str]:
    """
    Full NLP preprocessing pipeline: clean → tokenize → stopword removal → lemmatize.

    Parameters
    ----------
    text : str
        Raw email text.
    do_lemmatize : bool
        Whether to apply lemmatization.
    do_remove_stopwords : bool
        Whether to remove stopwords.
    return_tokens : bool
        If True, return list of tokens; if False, return joined string.

    Returns
    -------
    str or List[str]
        Preprocessed text or token list.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)

    if do_remove_stopwords:
        tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem(tokens)
    elif do_lemmatize:
        tokens = lemmatize(tokens)

    if return_tokens:
        return tokens
    return " ".join(tokens)
