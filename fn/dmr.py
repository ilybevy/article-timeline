import tomotopy as tp
import numpy as np
from collections import defaultdict


def train_dmr(year_groups, k=20, iterations=1000):
    """
    year_groups: {year: [docs]}
    """

    model = tp.DMRModel(k=k)

    docs_meta = []

    # add docs
    for year, docs in year_groups.items():
        for doc in docs:
            words = doc["tokens"]

            model.add_doc(words, metadata=year)
            docs_meta.append(year)

    # train
    for i in range(0, iterations, 10):
        model.train(10)
        print(f"Iter {i+10}: LL per word = {model.ll_per_word}")

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