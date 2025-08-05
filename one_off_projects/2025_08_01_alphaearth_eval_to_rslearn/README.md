This is for comparison on the AlphaEarth supplemental evaluation datasets.
https://zenodo.org/records/16585402

The `create_datasets.py` converts them to rslearn dataset format. Then the data needs
to be materialized.

`rslp.helios.get_embeddings` can be used to compute embeddings with Helios model. Then
we can run `get_balanced_accuracy.py` to compute the balanced accuracy.
