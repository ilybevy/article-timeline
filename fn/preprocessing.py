import re
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, Phraser


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split()


def remove_stopwords(words):
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def build_ngram_models(tokenized_docs, min_count=5, threshold=10):
    # bigram
    bigram = Phrases(tokenized_docs, min_count=min_count, threshold=threshold)
    bigram_phraser = Phraser(bigram)

    # apply bigram first
    bigram_docs = [bigram_phraser[doc] for doc in tokenized_docs]

    # trigram
    trigram = Phrases(bigram_docs, min_count=min_count, threshold=threshold)
    trigram_phraser = Phraser(trigram)

    return bigram_phraser, trigram_phraser


def apply_ngrams(words, bigram_model, trigram_model):
    words = bigram_model[words]
    words = trigram_model[words]
    return words


def preprocess_docs(year_groups):
    all_tokenized = []

    # 1. clean + tokenize + stopwords
    for docs in year_groups.values():
        for doc in docs:
            text = clean_text(doc["text"])
            words = tokenize(text)
            words = remove_stopwords(words)
            all_tokenized.append(words)

    # 2. build ngram
    bigram_model, trigram_model = build_ngram_models(all_tokenized)

    # 3. apply back
    idx = 0
    for docs in year_groups.values():
        for doc in docs:
            words = all_tokenized[idx]
            words = apply_ngrams(words, bigram_model, trigram_model)
            doc["tokens"] = words
            idx += 1

    return year_groups