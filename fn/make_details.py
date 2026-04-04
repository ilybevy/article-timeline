import json
import pandas as pd
import tomotopy as tp

from .label_maker import get_topic_keywords, generate_period_label, compute_topic_weight


def load_raw_data(raw_path: str):
    with open(raw_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_year_index(raw_data):
    year_index = {}

    for doc_id, doc in raw_data.items():
        year = doc.get("pub_year")
        if year is None:
            continue

        year_index.setdefault(year, []).append({
            "id": doc_id,
            "title": doc.get("title", ""),
            "citation_count": doc.get("cited_by_count", 0),
            "year": year
        })

    return year_index


def aggregate_segment_stats(year_index, start_year, end_year):
    total_papers = 0
    total_citations = 0

    for y in range(start_year, end_year + 1):
        docs = year_index.get(y, [])
        total_papers += len(docs)
        total_citations += sum(d["citation_count"] for d in docs)

    return total_papers, total_citations


def make_details(csv_path: str, raw_data_path: str, model_path: str, output_path: str):
    df = pd.read_csv(csv_path)

    raw_data = load_raw_data(raw_data_path)
    year_index = build_year_index(raw_data)

    model = tp.DMRModel.load(model_path)
    topic_keywords_map = get_topic_keywords(model, k=10)

    output = []

    for _, row in df.iterrows():
        start_year = int(row["start_year"])
        end_year = int(row["end_year"])
        topic_id = int(row["topic"])

        total_papers, total_citations = aggregate_segment_stats(
            year_index,
            start_year,
            end_year
        )

        keywords = topic_keywords_map.get(topic_id, [])

        topic_weight = compute_topic_weight(
            df,
            start_year,
            end_year,
            topic_id
        )

        period_label = generate_period_label(
            keywords=keywords,
            start_year=start_year,
            end_year=end_year,
            topic_weight=topic_weight
        )

        output.append({
            "start_year": start_year,
            "end_year": end_year,
            "total_papers": total_papers,
            "total_citations": total_citations,
            "period_label": period_label,
            "main_theme": "",
            "representative_papers": []
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")