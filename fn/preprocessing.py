import re
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords



with open("data/EN_STOPWORDS.txt", "r", encoding="utf-8") as f:
    EN_STOPWORDS = set(
        line.strip().lower()
        for line in f
        if line.strip()
    )

GERMAN_STOPWORDS = set(stopwords.words("german"))

DOMAIN_STOPWORDS = {
    "method", "methods", "approach", "approaches",
    "data", "dataset", "paper", "result", "results",
    "study", "studies", "technology", "analysis",
    "algorithm", "system", "systems", "based", "use"
}

STOPWORDS = EN_STOPWORDS.union(GERMAN_STOPWORDS).union(DOMAIN_STOPWORDS)


URL_PATTERN = re.compile(r"http\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
NON_TEXT_PATTERN = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    text = text.lower()

    # remove urls
    text = URL_PATTERN.sub(" ", text)

    # remove emails
    text = EMAIL_PATTERN.sub(" ", text)

    # normalize newline / whitespace
    text = text.replace("\n", " ")

    # remove non alphabetic
    text = NON_TEXT_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str):
    return text.split()


def remove_stopwords(words):
    return [
        w for w in words
        if w not in STOPWORDS and len(w) > 2
    ]



def build_ngram_models(
    tokenized_docs,
    bigram_min_count=5,
    bigram_threshold=10,
    trigram_min_count=10,
    trigram_threshold=15
):
    """
    Build separate tunable bigram + trigram models.
    """

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
    all_tokenized = []

    for docs in year_groups.values():
        for doc in docs:
            text = doc.get("text", "")

            text = clean_text(text)
            words = tokenize(text)
            words = remove_stopwords(words)

            all_tokenized.append(words)

    bigram_model, trigram_model = build_ngram_models(all_tokenized)

    idx = 0
    for docs in year_groups.values():
        for doc in docs:
            words = all_tokenized[idx]
            words = apply_ngrams(words, bigram_model, trigram_model)

            doc["tokens"] = words
            idx += 1

    return year_groups
    