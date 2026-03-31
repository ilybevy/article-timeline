# Timeline Generator

Timeline Generator is a small pipeline that turns a document collection into a timeline of periods.
The idea is simple:

- Group documents by publication year.
- Learn topic distributions from the corpus.
- Merge nearby years into coherent segments.
- Use lambda to control how many segments you get.

This README focuses on how the project works and how to run it, without heavy math.

## What This Project Solves

When you have many documents across years, it is hard to see where major theme shifts happen.
This project helps by creating contiguous time segments such as:

- 2010-2013: early phase
- 2014-2018: growth phase
- 2019-2023: mature phase

Each segment is built automatically from topic patterns in the data.

## Data Format

Input files are JSON objects where each item is one document.
Expected fields used by the loader:

- pub_year
- keywords
- content
- title (optional)

Current datasets in this workspace:

- data/machine_translation_docs_info.json
- data/machine_learning_docs_info.json

## Project Structure

- fn/data_loader.py: load raw JSON and group docs by year
- fn/preprocessing.py: clean text, tokenize, remove stopwords, build n-grams
- fn/dmr.py: train DMR topic model and extract topic distribution by year
- fn/segmentation.py: dynamic programming segmentation
- fn/metrics.py: distortion and scoring helpers
- fn/pipeline.py: high-level pipeline functions
- fn/utils.py: helper utilities (lambda sweep, plotting, dataframe export)
- DMR_JS_DP.ipynb: end-to-end notebook workflow

## Pipeline Overview

### 1) Preprocess and Save Dataset

Function:

- preprocess_and_save_dataset(raw_path, output_path)

What it does:

- Load raw documents from JSON.
- Build text per document from keywords + content.
- Clean and tokenize text.
- Remove stopwords and short tokens.
- Learn/apply bi-gram and tri-gram phrases.
- Save processed data as .pkl for faster reuse.

### 2) Train DMR Model

Function:

- train_dmr_model(dataset_path, model_path, k_topics, iterations)

What it does:

- Read processed dataset.
- Train a DMR model where metadata is publication year.
- Save trained model to disk.

### 3) Build Timeline Segments

Function:

- build_timeline_from_model(dataset_path, model_path, lambda_penalty)

What it does:

- Load processed data + trained model.
- Extract topic distribution for each year.
- Run dynamic programming to split years into contiguous segments.
- Return segments and score breakdown.

## Choosing Lambda

Use:

- sweep_lambda(dataset_path, model_path)
- plot_elbow(lambdas, costs, segments)

How to read the chart:

- Lower lambda usually gives more segments.
- Higher lambda usually gives fewer segments.
- Pick a value near the elbow where cost improvement starts to flatten.

## Main Outputs

build_timeline_from_model(...) returns a dictionary with:

- segments: list of (start_index, end_index) over sorted years
- year_distributions: topic distribution info for each year
- sorted_years: list of years in ascending order
- total_score: overall distortion score
- score_breakdown: per-segment details

If you need a tabular summary, use:

- segments_to_dataframe(result)

## Quick Start (Notebook)

Open DMR_JS_DP.ipynb and run cells in order:

1. Clean dataset and save pkl files.
2. Train DMR models.
3. Sweep lambda and plot elbow.
4. Choose lambda and generate final segments.

## Minimal Script Example

```python
from fn.pipeline import (
	preprocess_and_save_dataset,
	train_dmr_model,
	build_timeline_from_model,
)

preprocess_and_save_dataset(
	r"data\machine_translation_docs_info.json",
	r"data\translation_dataset.pkl",
)

train_dmr_model(
	r"data\translation_dataset.pkl",
	r"models\trans_dmr.bin",
	k_topics=20,
	iterations=500,
)

result = build_timeline_from_model(
	r"data\translation_dataset.pkl",
	r"models\trans_dmr.bin",
	lambda_penalty=0.25,
)

print(result["segments"])
print(result["total_score"])
```

## Dependencies

Core libraries used in code:

- numpy
- pandas
- matplotlib
- tomotopy
- gensim

Install with your preferred environment manager, then run the notebook.

## Notes

- The current main pipeline is topic-distribution based.
- Keep input years reasonably continuous for cleaner segmentation behavior.
- If scores look unstable, try increasing iterations or adjusting k_topics.
