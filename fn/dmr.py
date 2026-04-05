import tomotopy as tp
import numpy as np
from collections import defaultdict


import tomotopy as tp
import os


def train_dmr(year_groups, k=20, iterations=1000, log_filename=None):

    model = tp.DMRModel(k=k)

    doc_id_list = []

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
            doc_id_list.append(doc["id"])

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

    return model, doc_id_list

def build_doc_topic_mapping(model, doc_id_list):
    doc_topic_map = {}

    for model_doc, doc_id in zip(model.docs, doc_id_list):
        doc_topic_map[doc_id] = model_doc.get_topic_dist().tolist()

    return doc_topic_map

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

