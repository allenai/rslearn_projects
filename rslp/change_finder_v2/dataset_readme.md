## Dataset Details

This document has details about the different parts of the dataset.

All the parts are in JSON files compatible with the annotation app
(`rslp.change_finder_v2.annotation_app`), see [README.md](./README.md) for more details
about how to run that.

### Phase 1: Annotate based on ten years of land cover

Phase 1 is for annotation based on ten years of land cover predictions. The model is in
`data/land_cover_change/worldcover_change/config.yaml`, it processes 32x32 crops.

The [rslp/change_finder/README_landcover.md](../change_finder/README_landcover.md) has
more details about the approach. The summary is that we got 6,349 components with
change by connecting together pixels where the model predicts category X for three
years with confidence > 0.75, then a pivot year (ignore the prediction), then category
Y with confidence also > 0.75. We also only consider connected components of at least
10 pixels. Also get corresponding components with no change (high confidence in
category X across all the years). Then sub-sample up to 100 per
(src_category, dst_category) pair and convert it to points for `rslp.change_finder_v2`.

This results in 2,194 positive points. We don't look at the negative points, we just
assume they are correct.

- Original set: `/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_v2_annotation_20260523/annotations_original.json`
- Batch 2: `/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_v2_annotation_20260523/annotations_batch2_20260524.json`
- Batch 3: `/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_v2_annotation_20260523/annotations_batch3_20260606_with_timestamps.json`

The original and batch 2 files are pretty much the same (original is what was first
annotated in `rslp.change_finder` before switching to points in `rslp.change_finder_v2`,
batch 2 is a second set). Batch 3 is for the remaining points that weren't covered in
original or batch 2, and has timestamps predicted by the `rslp.change_finder_v2.annotation_timestamp_helper`
model.

## Phase 2: Annotate random outputs predicted as change

For this phase we use `rslp.change_finder_v2.lcc_model.write_jobs_random_2048` to
compute model outputs on random 2048x2048 tiles around the world. Then
`rslp/change_finder_v2/scripts/annotation_phase2/create_v2_annotations.py` processes
the outputs, specifically it samples up to one pixel where the binary change
probability is > 0.5 per output tile (skip tiles if no pixels are predicted as having
change). So the set to be annotated is just these sampled pixels.

We end up with 565 pixels. After annotation, there are 127 labeled positive, 425
labeled negative, and 13 skipped. So there were lots of false positives in this stage,
even though the model had achieved 80% precision @ >0.5 threshold in the test set.

## Phase 3: Annotate more random outputs

Phase 3 uses `rslp.change_finder_v2.scripts.annotation_phase3.write_jobs_random_2048_china`
to compute model outputs on random 2048x2048 tiles in China since there is a high rate
of change there (including diverse changes like renewable energy deployment and
re-development).

Otherwise it is similar to Phase 2. We intended to compute the outputs using a new model
trained with the Phase 2 data, but it was accidentally trained on the older dataset. So
there are probably more false positives than there would have been if Phase 2 had been
incorporated.

We end up with 237 points. After annotation, there are 64 labeled positive and 172
labeled negative (with one skipped).

## Phase 4: Annotate based on per-pixel land cover

Phase 4 is pending but it's similar to phase 1 except we use a per-pixel land cover
model to try to find smaller-scale changes. Unlike Phase 1 we also don't apply the
minimum connected component size.
