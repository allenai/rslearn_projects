"""LLM-based fine categorization of land-cover-change annotations.

Reads an annotation-app JSON, fetches Sentinel-2 L2A imagery (via the
olmoearth_datasets data source) and high-resolution ArcGIS Wayback imagery for
each selected positive point, then prompts a Gemini vision model to assign fine
change categories. Results are cached per entry as JSON files.
"""
