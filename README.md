# Geographic analysis of Che Guaveras Diaries: Geographic Informtaion Retrieval 


This project explores geographic information retrieval techniques applied to The Motorcycle Diaries by Ernesto Che Guevara. The goal is to build a pipeline that can automatically extract place references from the text, normalize them, and link them to geographic coordinates and country metadata.

It’s part of my BA thesis in Digital Humanities, where I combine computational text analysis with geographic visualization to reconstruct the journey described in the diaries.


![alt text](https://github.com/alicenanana/idontwanna/blob/main/geo_pipeline_flowchart_pretty_png.png)


# Features (Build up of the project)

Config‑first: all parameters (input path/pages, countries, limits) in config.yaml.

Modular helpers: NLP utilities in nlp_helpers.py; gazetteer utilities in gazetteer_helpers.py.

Single‑init NLP: one import/initialization block to avoid state drift.

Persistent outputs: intermediates saved with input‑derived filenames.

Region‑aware maps: plots restricted to South America for readability.
