## Dataset Details

This document has details about the different phases of the OlmoEarth LCC dataset.
The published version is at https://huggingface.co/datasets/allenai/olmoearth_lcc.

All the parts are in JSON files compatible with the annotation app
(`rslp.olmoearth_lcc.annotation_app`), see [README.md](./README.md) for more details
about the format and how to run the app. Annotation JSONs are stored under
`/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_v2_annotation_20260523/`.

All phases were verified manually in the annotation app, which shows monthly
Sentinel-2 mosaics. The phases differ in how the candidate points were proposed:

- **Land-cover-based**: apply a per-year land cover model on a ten-year Sentinel-2
  dataset and look for pixels whose predicted category changes confidently.
- **Output-based**: apply the current LCC model at scale, then sample pixels that it
  predicts as change (at a threshold tuned for high recall / low precision) and label
  those. Positive labels improve recall on real changes while the many false positives
  become negative labels.
- **Tile-based / dataset-based**: seed points from external sources (Codex-generated
  tiles with known changes, or public mining datasets).

Many phases were created with `rslp.olmoearth_lcc.lcc_model.write_jobs_random_2048`
(compute LCC model outputs on random 2048x2048 tiles around the world, with
`--write_raster true`) followed by
`rslp.olmoearth_lcc.annotation_scripts.phase02_output_sampling.create_v2_annotations`
(sample up to one pixel predicted as change per output tile and write a v2 annotation
entry with one positive point centered on it). Below this is called the
**output-based pipeline**.

### Phase 1: Annotate based on ten years of land cover

We trained a model to predict per-year land cover (a 32x32-crop segmentation model
trained on WorldCover labels), then applied it on many randomly sampled locations over
a ten-year period (the ten-year dataset, see [README.md](./README.md)). Candidate
change locations were proposed by comparing the per-year land cover predictions:
specifically, we looked for places where the model was confident (> 0.75) that the
land cover was one category for three years, then ignored the next (pivot) year in
case the change is gradual, and then the model was confident that the land cover was
another category for three more years. We connected qualifying pixels into components
and only kept components of at least 10 pixels, giving 6,349 components. We also got
corresponding no-change components (high confidence in the same category across all
the years). Then we sub-sampled up to 100 per (src_category, dst_category) pair and
converted the components to points.

This results in 2,194 positive points. We don't look at the negative points, we just
assume they are correct.

- Original set: `annotations_original.json`
- Batch 2: `annotations_batch2_20260524.json`
- Batch 3: `annotations_batch3_20260606_with_timestamps.json`

The original and batch 2 files are pretty much the same (original is what was first
annotated with a polygon-based viewer before switching to points, batch 2 is a second
set). Batch 3 is for the remaining points that weren't covered in original or batch
2.

The crop-input land cover pipeline used for this phase has since been removed from
the codebase in favor of the per-pixel approach (Phase 4).

### Phase 2: Annotate random outputs predicted as change

The first use of the output-based pipeline: `write_jobs_random_2048` on random tiles
around the world, then `create_v2_annotations` samples up to one pixel where the
binary change probability is > 0.5 per output tile (skip tiles if no pixels are
predicted as having change).

We end up with 565 pixels. After annotation, there are 127 labeled positive, 425
labeled negative, and 13 skipped. So there were lots of false positives in this stage,
even though the model had achieved 80% precision @ >0.5 threshold in the test set.

### Phase 3: Annotate more random outputs (China)

Phase 3 uses
`rslp.olmoearth_lcc.annotation_scripts.phase03_random_tiles_china_africa.write_jobs_random_2048_china`
to compute model outputs on random 2048x2048 tiles in China since there is a high rate
of change there (including diverse changes like renewable energy deployment and
re-development). Otherwise it is the output-based pipeline as in Phase 2.

We intended to compute the outputs using a new model trained with the Phase 2 data,
but it was accidentally trained on the older dataset. So there are probably more false
positives than there would have been if Phase 2 had been incorporated.

We end up with 237 points. After annotation, there are 64 labeled positive and 172
labeled negative (with one skipped).

### Phase 4: Annotate based on per-pixel land cover

Similar to Phase 1, but we use a per-pixel land cover model (no spatial context)
instead of one that has a larger spatial context, so that we can find very
small-scale changes. Unlike Phase 1 we also don't apply the minimum connected
component size. See [land_cover/README.md](./land_cover/README.md):
`rslp.olmoearth_lcc.land_cover.find_change` applies the model to the ten-year dataset
and writes one v2 annotation entry per window with a qualifying change.

### Phase 5: Output-based labeling

Output-based pipeline.

### Phase 6: Codex tiles

Label random points in 0.1 x 0.1 degree tiles that are likely to have changes (e.g.
agricultural activity, wildfire, flooding, new airport, etc.). The tiles were
generated by prompting a Codex agent to propose tiles with likely changes (based on
web search and its own knowledge) and are stored as CSVs in `codex_tiles/`:

- `codex_tiles_1.csv` (25 tiles): used to sample `eval_points.json`, which was
  originally intended as an evaluation set.
- `codex_tiles_2.csv` (108 tiles): a larger set used for training points.

Both are now used as training data. Points are sampled uniformly within each tile
with `rslp.olmoearth_lcc.codex_tiles.sample_tile_points` (one 128x128 window per
point, `--points-per-tile` points per tile), and each point is then verified in the
annotation app.

### Phase 7: Output-based labeling

Output-based pipeline.

### Phase 8: Output-based labeling

Output-based pipeline.

### Phase 9: Output-based labeling, rare transitions

Output-based pipeline, but the sampled pixels were manually filtered to focus on rare
(src_land_cover, dst_land_cover) transitions that didn't appear frequently in the
labels collected so far.

### Phase 10: Per-pixel land cover in sub-Saharan Africa

Similar to Phase 4 but in sub-Saharan Africa, oversampling the tree -> bare
transition. Windows were added to the ten-year dataset with the
`create_windows_africa` workflow, and `land_cover.find_change` was run with
`--src_category tree --dst_category bare` (in addition to unrestricted runs).

### Phases 11-15: Output-based labeling focused on mining predictions

Output-based pipeline, but using
`rslp.olmoearth_lcc.annotation_scripts.phase12_mining_predictions.create_mining_annotations`
instead of `create_v2_annotations`: it selects pixels where the post-change-category
head predicts `mining` (argmax, or `post_change_mining` probability above
`--threshold`) and samples one qualifying pixel per tile. Some of these phases also
used `write_jobs_random_2048_africa` to focus on sub-Saharan Africa.

### Phase 16: Mining annotations seeded from external datasets

Label points based on a set of 10 public datasets related to artisanal and small-scale
mining in different parts of sub-Saharan Africa (100 points from each). See
[annotation_scripts/phase16_mining_datasets/README.md](./annotation_scripts/phase16_mining_datasets/README.md)
for the datasets and how the seed points were generated.

### Phase 17: Output-based labeling

Output-based pipeline.

### Phase 18: Output-based labeling

Output-based pipeline.
