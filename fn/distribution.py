from collections import defaultdict


def build_keyword_distribution(year_docs):
    """
    Build a normalized keyword probability distribution for a year's documents.
    """
    keyword_counts = defaultdict(int)

    for doc in year_docs:
        for keyword in doc.get("keywords", []):
            normalized_keyword = keyword.strip().lower()
            if normalized_keyword:
                keyword_counts[normalized_keyword] += 1

    total_keywords = sum(keyword_counts.values())
    if total_keywords == 0:
        return {}

    keyword_dist = {
        keyword: count / total_keywords
        for keyword, count in keyword_counts.items()
    }

    total_probability = sum(keyword_dist.values())
    if total_probability == 0:
        return {}

    return {
        keyword: probability / total_probability
        for keyword, probability in keyword_dist.items()
    }