import re
import spacy
from gensim.models.phrases import Phrases, Phraser
from spacy.lang.de.stop_words import STOP_WORDS as GERMAN_STOPWORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FRENCH_STOPWORDS

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

with open("data/EN_STOPWORDS.txt", "r", encoding="utf-8") as f:
    EN_STOPWORDS = set(line.strip().lower() for line in f if line.strip())

GERMAN_STOPWORDS = set(GERMAN_STOPWORDS)
FRENCH_STOPWORDS = set(FRENCH_STOPWORDS)

DOMAIN_STOPWORDS = {
    "method", "methods", "approach", "approaches",
    "data", "dataset", "paper", "result", "results",
    "study", "studies", "technology", "analysis",
    "algorithm", "system", "systems", "based", "use", "theory", "propose", "proposed"
}

POST_TOKEN_FILTER = {
    "learning",
    "learn",
    "machine",
    "science",
    "model",
    "models",
    "methodology",
    "framework",
    "based_method",
    "data_set",
    "dataset",
    "system_model",
    "processing",
    "base"
}

STOPWORDS = (
    EN_STOPWORDS
    .union(GERMAN_STOPWORDS)
    .union(FRENCH_STOPWORDS)
    .union(DOMAIN_STOPWORDS)
)

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
NON_TEXT_PATTERN = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = text.replace("\n", " ")
    text = NON_TEXT_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    doc = nlp(text)
    return [t.lemma_ for t in doc]


def remove_stopwords(words):
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def apply_post_token_filter(words):
    return [w for w in words if w not in POST_TOKEN_FILTER and len(w) > 2]


def normalize_keywords(keywords):
    if not keywords:
        return []

    out = []
    for k in keywords:
        if not k:
            continue

        k = k.lower().strip()
        k = NON_TEXT_PATTERN.sub(" ", k)
        k = re.sub(r"\s+", "_", k).strip()

        if len(k) > 2:
            out.append(k)

    return out


def build_ngram_models(
    tokenized_docs,
    bigram_min_count=15,
    bigram_threshold=15,
    trigram_min_count=17,
    trigram_threshold=17
):
    bigram = Phrases(
        tokenized_docs,
        min_count=bigram_min_count,
        threshold=bigram_threshold
    )
    bigram_phraser = Phraser(bigram)

    bigram_docs = [bigram_phraser[doc] for doc in tokenized_docs]

    trigram = Phrases(
        bigram_docs,
        min_count=trigram_min_count,
        threshold=trigram_threshold
    )
    trigram_phraser = Phraser(trigram)

    return bigram_phraser, trigram_phraser


def apply_ngrams(words, bigram_model, trigram_model):
    words = bigram_model[words]
    words = trigram_model[words]
    return words


def preprocess_docs(year_groups):
    all_content_tokens = []
    doc_refs = []

    # PASS 1
    for year, docs in year_groups.items():
        for doc in docs:
            text = clean_text(doc.get("content", ""))
            words = tokenize(text)
            words = remove_stopwords(words)
            # words = apply_post_token_filter(words)

            all_content_tokens.append(words)
            doc_refs.append(doc)

    bigram_model, trigram_model = build_ngram_models(all_content_tokens)

    # PASS 2
    for i, doc in enumerate(doc_refs):
        content_tokens = all_content_tokens[i]

        content_tokens = apply_ngrams(
            content_tokens,
            bigram_model,
            trigram_model
        )
        content_tokens = apply_post_token_filter(content_tokens)
        keyword_tokens = normalize_keywords(doc.get("keywords", []))

        doc["tokens"] = content_tokens + keyword_tokens

    return year_groups