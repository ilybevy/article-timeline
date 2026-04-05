# Timeline Generator

Timeline Generator turns a collection of research papers into a clear timeline of thematic periods.

In simple terms, it helps answer:
- What were the main themes in each time window?
- When to choose to split the timeline segment?
- How can we summarize each period with representative papers and concise descriptions?

This README focuses on how the project works and how to run it, without heavy math.

A more detailed mathematical formulation, including the objective definition and the role of the regularization parameter λ, is provided in the accompanying technical write-up: https://www.overleaf.com/read/jjvjnkskbxbc#da4a03

The implementation follows this formulation but focuses on an engineering-first perspective: preprocessing, topic modeling, dynamic programming segmentation, and lambda-based model selection.

## What This Project Does

Given a dataset of papers (with publication years), the project:
1. Cleans and standardizes text.
2. Learns topic signals from the corpus.
3. Splits the timeline into meaningful contiguous periods.
4. Produces summary outputs for analysis and reporting.

## Input Data

Input files are JSON objects where each item is one document.

Main fields used:
- `pub_year`
- `keywords`
- `content`
- `title` (optional)

Current datasets in this workspace:
- `data/machine_translation_docs_info.json`
- `data/machine_learning_docs_info.json`

## Pipeline at a Glance

The workflow has four practical stages:

1. Data preparation
2. Topic modeling
3. Timeline segmentation
4. Period-level storytelling (labels, themes, representative papers)


## Module Responsibilities

### `fn/data_loader.py`
Reads raw JSON and groups papers by publication year.

### `fn/preprocessing.py`
Cleans text, tokenizes it, removes stopwords, and builds n-grams.

### `fn/dmr.py`
Trains the topic model and extracts topic distributions.

Why it matters:
- Produces the core signals used to detect period changes.

### `fn/segmentation.py`
Builds timeline segments from year-level topic signals.

Why it matters:
- Converts noisy year-by-year trends into coherent periods.

### `fn/metrics.py`
Scores segmentation quality.

### `fn/pipeline.py`
High-level orchestration functions.

### `fn/utils.py`
Helper functions for lambda sweep, elbow plotting, and DataFrame export.

### `fn/make_details.py`
Builds final period-level output files.

### `fn/representative_doc_maker.py`
Selects representative papers for each period.

### `fn/label_maker.py`
Generates short labels for each period.

### `fn/theme_writer.py`
Generates descriptive thematic text for each period.

### `DMR_JS_DP.ipynb`
End-to-end notebook workflow.

## Quick Start (Notebook)

Open `DMR_JS_DP.ipynb` and run cells in this order:

1. Clean datasets and save processed `.pkl` files.
2. Train DMR models.
3. Sweep lambda and view elbow charts.
4. Build final timeline segments.
5. Generate period labels and thematic summaries.

## Main Outputs

Typical outputs include:
- Trained models: `models/*.bin`
- Document-topic maps: `models/*_mapping.pkl`
- Segment tables: `output/*_df.csv`
- Detailed period summaries: `output/*.json`

## Practical Tuning Tips

- Too many segments: increase `lambda_penalty`.
- Too few segments: decrease `lambda_penalty`.
- Unstable or weak topics: increase training iterations or adjust topic count.

## Core Dependencies

Main libraries used:
- `numpy`
- `pandas`
- `matplotlib`
- `tomotopy`
- `gensim`
- `nltk`
- `faiss`
- `requests`

Install dependencies with your preferred environment manager, then run the notebook.
