import tomotopy as tp
import numpy as np
from collections import defaultdict


import tomotopy as tp
import os


def train_dmr(year_groups, k=20, iterations=1000, log_filename=None):
    """
    year_groups: {year: [docs]}

    log_filename: tên file log (optional)
    """

    model = tp.DMRModel(k=k)

    log_path = None
    if log_filename is not None:
        os.makedirs("output", exist_ok=True)
        log_path = os.path.join("output", log_filename)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("step\tll_per_word\n")

    for year, docs in year_groups.items():
        for doc in docs:
            words = doc["tokens"]
            model.add_doc(words, metadata=year)

    for i in range(0, iterations, 10):
        model.train(10)
        step = i + 10
        ll = model.ll_per_word

        print(f"Iter {step}: LL per word = {ll:.6f}")

        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{step}\t{ll:.6f}\n")

    if log_path is not None:
        print(f"\nSaved training log to: {log_path}")

    return model


def extract_year_topic_dist(model, year_groups):
    """
    Output:
        {year: topic_distribution_vector}
    """

    year_topic = defaultdict(list)

    for doc in model.docs:
        year = doc.metadata
        year_topic[year].append(doc.get_topic_dist())

    # average per year
    year_dist = {}

    for year, dists in year_topic.items():
        year_dist[year] = np.mean(dists, axis=0)

    return year_dist

def extract_topic_keywords(model, top_n=10):
    """
    Output:
        {
            topic_id: [(word, prob), ...]
        }
    """

    topic_keywords = {}

    for k in range(model.k):
        words = model.get_topic_words(k, top_n=top_n)
        topic_keywords[k] = [
            {
                "word": w,
                "prob": float(p)
            }
            for w, p in words
        ]

    return topic_keywords