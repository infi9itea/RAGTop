# RAGTOP

A retrieval-augmented laptop recommendation system built around multi-source data collection, distributed processing, dense retrieval, and experimental retrieval optimization.

## Pipeline

Data collection → Spark processing → chunking and embeddings → FAISS retrieval → evaluation

## Repository

* `data/` — raw, processed, and embedded data
* `notebooks/` — collection, processing, indexing, and experiments
* `retrieval/` — FAISS index, mappings, queries, and evaluation results
* `experiments/` — semantic caching and association-rule analysis
* `scripts/` — supporting utilities
* `references/` — source references


The repository contains both the data artifacts and intermediate results required to reproduce and inspect the retrieval experiments.
