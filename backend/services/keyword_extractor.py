import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)


def extract_keywords(subject: str) -> list[str]:
    """
    Extract relevant keywords from a research subject using NLTK.

    Args:
        subject: Research subject provided by user

    Returns:
        List of keywords for ArXiv search
    """
    logger.info(f"[KEYWORDS] Extracting keywords from: {subject}")
    text = subject.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))

    keywords = [
        word for word in tokens
        if word.isalnum() and word not in stop_words and len(word) > 2
    ]

    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    logger.info(f"[KEYWORDS] Extracted {len(unique_keywords)} keywords: {unique_keywords}")
    return unique_keywords
