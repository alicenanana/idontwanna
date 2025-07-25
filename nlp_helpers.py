import spacy
import stanza
import re
import yaml
import fitz
import unicodedata
import contractions

from nltk.corpus import stopwords
from typing import Set, List


def init_nlp(use_gpu: bool = True):
    """
    Initializes the main spaCy and Stanza NLP pipelines.

    Args:
        use_gpu (bool): Whether to use GPU for Stanza.

    Returns:
        tuple: (spaCy NLP pipeline, Stanza pipeline)
    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    stanza_pipeline = stanza.Pipeline(lang="en", processors="tokenize,ner", use_gpu=use_gpu)
    return nlp, stanza_pipeline


def get_stopwords(nlp) -> Set[str]:
    """Returns combined set of stopwords from NLTK and spaCy."""
    nltk_stops = set(stopwords.words("english"))
    spacy_stops = nlp.Defaults.stop_words
    return nltk_stops.union(spacy_stops)


def extract_text_from_pdf(pdf_path: str, start_page: int, end_page: int) -> str:
    """Extracts text from a PDF file between specified page numbers."""
    doc = fitz.open(pdf_path)
    pages = doc[start_page:end_page]
    return "\n".join(page.get_text() for page in pages)


def load_config(config_path: str = "config.yaml") -> dict:
    """Loads YAML config file and returns it as a dictionary."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def normalize_punctuation(text: str) -> str:
    """Replaces fancy quotes and dashes with standard ASCII equivalents."""
    return (
        text.replace("“", '"').replace("”", '"')
            .replace("’", "'").replace("‘", "'")
            .replace("—", "-").replace("–", "-")
    )


def clean_light(text: str) -> str:
    """Applies light cleaning including unicode normalization and noise removal."""
    text = unicodedata.normalize("NFKC", text)
    text = contractions.fix(text)
    text = normalize_punctuation(text)
    patterns = [
        r'https?://\S+', r'\S+@\S+', r'<.*?>', r'\+?\d[\d\-\(\)\s]{5,}\d'
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text)
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_text(text: str, stopwords: Set[str]) -> str:
    """Light cleaning + lowercasing + stopword removal."""
    cleaned = clean_light(text)
    tokens = cleaned.lower().split()
    return " ".join(t for t in tokens if t not in stopwords)


def segment_sentences(text: str, nlp) -> List[str]:
    """Segments preprocessed text into sentences using spaCy with sentencizer."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def clean_heavy(text: str, nlp, stopwords: Set[str]) -> str:
    """Deep NLP cleaning: lemmatization, alpha filtering, and stopword removal."""
    text = re.sub(r'[^a-z\s]', ' ', text.lower())
    doc = list(nlp.pipe([text], batch_size=1000, n_process=1))[0]
    return " ".join(
        tok.lemma_ for tok in doc
        if tok.lemma_.isalpha() and tok.lemma_ not in stopwords and tok.lemma_ != "-PRON-"
    )
