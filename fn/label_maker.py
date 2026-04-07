import numpy as np
from collections import defaultdict


def _select_top_topics(topic_vector, threshold=0.5):
    pairs = [(i, float(p)) for i, p in enumerate(topic_vector)]
    pairs.sort(key=lambda x: x[1], reverse=True)

    selected = []
    acc = 0.0

    for tid, prob in pairs:
        selected.append((tid, prob))
        acc += prob
        if acc >= threshold:
            break

    return selected


def _get_topic_words(model, topic_id, top_n):
    return model.get_topic_words(topic_id, top_n=top_n)


def generate_period_label(
    segment_topic_vector,
    model,
    threshold=0.5,
    top_words_per_topic=50,
    top_k=10
):
    selected_topics = _select_top_topics(segment_topic_vector, threshold)

    score_map = defaultdict(float)

    for topic_id, topic_prob in selected_topics:
        words = _get_topic_words(model, topic_id, top_words_per_topic)

        for word, word_prob in words:
            score_map[word] += topic_prob * word_prob

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    top_words = [w for w, _ in ranked[:top_k]]

    return ", ".join(top_words)