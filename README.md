# Geographic analysis of Che Guaveras Diaries: Geographic Informtaion Retrieval 


This project explores geographic information retrieval techniques applied to The Motorcycle Diaries by Ernesto Che Guevara. The goal is to build a pipeline that can automatically extract place references from the text, normalize them, and link them to geographic coordinates and country metadata.

It’s part of my BA thesis in Digital Humanities, where I combine computational text analysis with geographic visualization to reconstruct the journey described in the diaries.
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Quickstart](#quickstart)
- [Data Flow](#data-flow)
- [Usage](#usage)
- [Outputs](#outputs)
- [Coding Standards](#coding-standards)
- [Known Issues](#known-issues)
- [Roadmap](#roadmap--todo)
- [Citation](#citation)

## Overview 
![alt text](https://github.com/alicenanana/idontwanna/blob/main/geo_pipeline_flowchart_pretty_png.png)


## Features (Build up of the project)

Config‑first: all parameters (input path/pages, countries, limits) in config.yaml.

Modular helpers: NLP utilities in nlp_helpers.py; gazetteer utilities in gazetteer_helpers.py.

Single‑init NLP: one import/initialization block to avoid state drift.

Persistent outputs: intermediates saved with input‑derived filenames.

Region‑aware maps: plots restricted to South America for readability.

## Repository Structure
```text
├─ config.yaml                    # paths, page window, gazetteer params (username, countries, limits)
├─ nlp_helpers.py                 # tokenization, cleaning, stopword handling, NER helpers
├─ gazetteer_helpers.py           # gazetteer queries, normalization, merges
├─ nowa_wersja.ipynb              # main analysis notebook (imports → funcs → run → viz)
├─ Feedback.md                    # external review with actionable comments
├─ data/
│  ├─ raw/                        # e.g., MotorcycleDiaries.pdf
│  └─ processed/                  # cleaned text, enriched CSVs
├─ outputs/                       # figures, maps, final tables
└─ env/                           # environment files (optional)
```

## Configuration
All run‑time parameters live in config.yaml. Use relative paths for portability.
```text
pdf:
path: "data/raw/MotorcycleDiaries.pdf" # ← avoid absolute, user‑specific paths
start_page: 28
end_page: 148

gazetteer:
username: "<your-GeoNames-username>"
countries: ["AR", "CL", "PE", "CO", "VE", "BO", "EC", "PA", "CR", "GT", "MX", "CU", "BR", "GY", "PY", "SR", "UY", "HN", "SV", "NI"]
max_rows: 1000
```
## Quickstart

1. Open nowa_wersja.ipynb.

2. Run the imports block once (consolidated at the top).

3. Run the function blocks (one concept per block: cleaning → NER → gazetteer → enrich → viz).

4. Execute the pipeline: read PDF pages from config.yaml, clean, NER, resolve, enrich, save intermediates under data/processed/ (filenames derived from the input), and then visualize.

Each execution block is written to run stand‑alone after imports + function declarations—no hidden state across cells.

## Data Flow
Ingest (PDF subset) → Clean (normalize, stopwords) → NER (spaCy) → Match/Enrich (gazetteer joins) → Filter (country allow‑list) → Persist (CSV/GeoCSV) → Visualize (South America bounds)

*Post‑merge hygiene* : drop _x/_y duplicates; consolidate longitude/latitude; remove source‑specific duplicates once unified.

## Usage
Run cells in order: imports → functions → execution → viz. Figures and tables land in outputs/.

## Outputs

- data/processed/…cleaned.csv — cleaned tokens/entities

- data/processed/…geoparsed.csv — NER + gazetteer matches

- outputs/geoparsing_final_enriched.csv — final enriched table (coords, countries)

- outputs/maps/route_south_america.png — regional map

Filenames are derived from the input PDF name + page window to stay generic for other texts.

