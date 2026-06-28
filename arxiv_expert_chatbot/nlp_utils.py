import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load spaCy model lazily
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            _nlp = None
    return _nlp


def extract_keywords_tfidf(texts: list, top_n: int = 15) -> list:
    """Extract top keywords from a list of texts using TF-IDF."""
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            max_features=500,
            ngram_range=(1, 2)
        )
        X = vec.fit_transform(texts)
        scores = X.sum(axis=0).A1
        vocab = vec.get_feature_names_out()
        ranked = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)
        # Filter out pure numbers and very short tokens
        keywords = [(w, s) for w, s in ranked if len(w) > 2 and not w.isdigit()]
        return keywords[:top_n]
    except Exception:
        return []


def extract_entities(text: str) -> dict:
    """Extract named entities using spaCy."""
    nlp = get_nlp()
    if not nlp or not text:
        return {}
    doc = nlp(text[:2000])
    entities = {}
    for ent in doc.ents:
        label = ent.label_
        if label not in entities:
            entities[label] = []
        if ent.text not in entities[label]:
            entities[label].append(ent.text)
    return entities


def get_category_stats(papers: list) -> dict:
    """Count papers by primary category."""
    counts = Counter(p.get("primary_category", "Unknown") for p in papers)
    return dict(counts.most_common())


def get_year_stats(papers: list) -> dict:
    """Count papers by publication year."""
    years = Counter()
    for p in papers:
        pub = p.get("published", "")
        if pub and len(pub) >= 4:
            years[pub[:4]] += 1
    return dict(sorted(years.items()))


def clean_text(text: str) -> str:
    """Remove LaTeX and special chars for display."""
    text = re.sub(r'\$[^$]+\$', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_keyword_cooccurrence(papers: list, top_n: int = 20) -> dict:
    """
    Build a simple keyword co-occurrence map from paper abstracts.
    Returns {keyword: [co-occurring keywords]} for network visualization.
    """
    all_texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers[:100]]
    top_keywords = [kw for kw, _ in extract_keywords_tfidf(all_texts, top_n=top_n)]

    cooccurrence = {kw: [] for kw in top_keywords}
    for text in all_texts:
        text_lower = text.lower()
        present = [kw for kw in top_keywords if kw in text_lower]
        for i, kw1 in enumerate(present):
            for kw2 in present[i + 1:]:
                if kw2 not in cooccurrence[kw1]:
                    cooccurrence[kw1].append(kw2)
                if kw1 not in cooccurrence[kw2]:
                    cooccurrence[kw2].append(kw1)

    return cooccurrence
