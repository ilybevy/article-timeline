import json
import pandas as pd
import tomotopy as tp
import pickle
import ast
import numpy as np
from typing import Dict

from .representative_doc_maker import get_representative_papers
from .label_maker import generate_period_label
from .theme_writer import ThemeWriter  


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
            "year": year,
            "content": doc.get("content", "")  
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


def embed_fn(text: str):
    return np.random.rand(768)   


def make_details(
    csv_path: str,
    raw_data_path: str,
    model_path: str,
    output_path: str,
    mapping_path: str,
    questions: Dict[str, str]
):
    df = pd.read_csv(csv_path)
    raw_data = load_raw_data(raw_data_path)
    year_index = build_year_index(raw_data)
    model = tp.DMRModel.load(model_path)

    with open(mapping_path, "rb") as f:
        doc_topic_map = pickle.load(f)

    output = []
    theme_writer = ThemeWriter(embed_fn=embed_fn)

    for _, row in df.iterrows():
        start_year = int(row["start_year"])
        end_year = int(row["end_year"])
        topic_id = int(row["topic"])

        docs = []
        for y in range(start_year, end_year + 1):
            docs.extend(year_index.get(y, []))

        total_papers, total_citations = aggregate_segment_stats(
            year_index,
            start_year,
            end_year
        )

        words = model.get_topic_words(topic_id, top_n=20)
        keywords = [w for w, _ in words]

        period_label = generate_period_label(
            keywords=keywords,
        )

        segment_topic_vector = ast.literal_eval(row["topic_vector"])

        representative_papers = get_representative_papers(
            docs=docs,
            doc_topic_map=doc_topic_map,
            segment_topic_vector=segment_topic_vector,
            top_k=5,
            threshold=0.5
        )

        main_theme = theme_writer.generate_theme(
            papers=representative_papers,
            questions=questions
        )

        output.append({
            "start_year": start_year,
            "end_year": end_year,
            "total_papers": total_papers,
            "total_citations": total_citations,
            "period_label": period_label,
            "main_theme": main_theme,
            "representative_papers": representative_papers
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")