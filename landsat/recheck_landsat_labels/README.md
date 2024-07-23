After Landsat model was put into Skylight integration in early July 2024, even after
raising confidence threshold from 0.5 to 0.7 there are still many false positives.

Joe did a quick check of 11 small vessels that I found, and he said 5 of them were
incorrect. So that makes it seem that the training data quality is a big issue.

The goal now is to validate the vessel labels in the training data.

Phase 1 - have Joe go through up to 1000 labels and annotate if they are correct or
not.

Phase 2 - then we can train a classification model on the new correct/incorrect
annotations, and try to see which other labels are most likely to be incorrect and
validate them too.
