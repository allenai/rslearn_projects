"""Landsat vessel feedback loop.

Turns false positives flagged in the Skylight platform into new classifier training
samples, retrains, and republishes the served Docker image. Stages (see README.md):

    pull -> create_windows -> add_to_training -> (train) -> publish
"""
