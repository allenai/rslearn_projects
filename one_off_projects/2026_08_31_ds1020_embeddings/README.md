ds1020 survey-point embedding tiles (2026-08-31)
================================================

One 224x224 px (2.24 km square, 10 m) embedding tile per ds1020 survey point,
centered on the point, from a 3-month S2 + S1 + Landsat export. Same route as
the 2026_08_19 AOI run (plain rslearn dataset driven by the CLI, no GeoZarr),
sharded across Beaker jobs because there are 58,825 windows rather than seven.

The two CSVs:

| CSV                     | points | window time range                          |
|-------------------------|--------|--------------------------------------------|
| `*_points_2017.csv`     |  6,994 | month before obs date + obs month + month after (dates actually span 2017-01..2023-06) |
| `*_points_fixed.csv`    | 51,831 | May + June + July 2018 (every date is 2018-06-15) |

All points are in California / western US (lat 32.5-42, lon -124..-114.6), so
windows land in UTM zones 10-11.

Window geometry: why 224 and not 228
------------------------------------

The model runs on 16x16 crops with `overlap_size=4` (stride 12), and 224 is not
`16 + 12k` -- but no padding is needed. rslearn's `AllCropsDataset` clamps the
final crop of each row/column to the window edge (it deliberately overlaps the
previous crop more; see `rslearn/train/all_crops_dataset.py`), and `RasterMerger`
stitches that the same way it already did for the AOI run's 1024/1280/1792 px
windows, none of which were stride-aligned either. The only hard divisibility
constraint is Landsat's `zoom_offset: -1` band set, which needs even window
dimensions; 224 is even. So the tiles are exactly 224x224, centered on the
point per `add_windows` convention (projected coordinate truncated to int, then
+/- 112 px).

Temporal export: identical recipe to pretraining, 3 periods instead of 12
-------------------------------------------------------------------------

`config_90d.json` is `data/large_scale_embeddings/s2_s1_landsat_two_models.json`
with `duration: 90d` and `max_matches: 3` (still `period_duration: 30d`,
`space_mode: MOSAIC`, same sources, same band sets). Each window's start is the
first day of the month before the observation month, so the three 30-day
mosaics approximate [previous month, event month, next month]. `min_matches: 1`
means a window with a data gap still materializes with fewer mosaics; the model
config's `use_legacy_timestamps: false` handles variable mosaic counts.

What runs
---------

- `make_windows.py` -- bulk window creation (one window per CSV row, named by
  `survey_id` sanitized to `[A-Za-z0-9_.-]`; 17 fixed-CSV ids contain " - ").
  Windows are assigned to shard groups `y2017_00..y2017_03` and
  `fixed_00..fixed_25` of 2,000 rows each, the sharding unit for Beaker jobs.
- `config_90d.json` -- dataset config, declaring both arms' output layers.
- `cand_ndvi.yaml` / `distilled.yaml` -- the AOI run's model configs with batch
  sizes retuned for 9 mosaics per sample. cand_ndvi writes layer `output`;
  distilled writes `output_distilled` and needs `OE_PROJECTED_REGISTER_DIM=128`
  plus `patch_rslearn_projected_registers.py` (otherwise it would silently write
  the first 128 of 768 register dims, which is not the distilled embedding).
- `run_pipeline.sh` -- stage-selectable (`STAGES`) and group-shardable
  (`SHARD_GROUPS`): windows / prepare / materialize / check / predict. The check
  stage counts materialized layers per group and fails if any window has no
  Sentinel-2 (the one required modality); missing S1/Landsat are reported, not
  fatal, matching their `required: false` in the model config.
- `make_beaker_specs.py` -- writes `specs/windows.yaml` (one 0-GPU,
  non-preemptible window-creation task) plus `specs/cand.yaml` and
  `specs/distilled.yaml` (30 1-GPU tasks each, one per shard group). The
  imagery is shared, so it materializes exactly once: `--materialize_arm`
  (default cand) picks which arm's tasks carry prepare + materialize + check;
  the other arm is predict-only and must run second. To run only one arm, pass
  `--materialize_arm <that arm>` and never launch the other spec.

Running it
----------

    beaker dataset create one_off_projects/2026_08_31_ds1020_embeddings \
        --name ds1020-embeddings-20260831 --workspace ai2/earth-systems
    python make_beaker_specs.py --mount gabrielt/ds1020-embeddings-20260831
    beaker experiment create specs/windows.yaml      # once, first
    beaker experiment create specs/cand.yaml         # after windows completes
    beaker experiment create specs/distilled.yaml    # after cand completes

Distilled arm only (no cand_ndvi forward passes at all):

    python make_beaker_specs.py --mount gabrielt/ds1020-embeddings-20260831 \
        --materialize_arm distilled
    beaker experiment create specs/windows.yaml
    beaker experiment create specs/distilled.yaml    # after windows completes

Smoke-test first: regenerate with `--groups y2017_00` and run that single cand
task before launching all 30; confirm the checkpoint loads, mosaics
materialize, and the dequantized per-pixel L2 norm is ~1.0.

To re-run failed shards, regenerate specs with `--groups <the failed ones>`;
prepare/materialize skip completed layers, and predict rewrites the output
layer idempotently.

Scale expectations
------------------

Forward-pass area is ~305,000 km2-equivalent of pixels (58,825 x 224^2 =
2.95 G px), ~200x the AOI run, but with 9 mosaics per sample instead of 36
(~50x the AOI run's GPU work). Materialization is ~530k window-layer mosaics;
at 2,000-window shards that is ~18k mosaics and ~723k crops (~2.8k batches at
256) per cand task.

Caveats
-------

- ~1,500 fixed-CSV points share a lat/lon with another point (same date), so a
  few percent of windows are byte-identical duplicates under different names.
  Not worth deduplicating.
- 43 of the 2017-CSV points observe before March 2017, so their windows start
  in December 2016 or January 2017: Sentinel-2 L2A coverage that early is
  thinner (US L2A is largely reprocessed archive), so expect some 1-2-mosaic
  windows there.
- One torn GeoTIFF in any layer kills every task on that dataset when read via
  the eval loader; if a shard's predict dies on a read error, scan and delete
  the offending mosaic and re-run materialize for that group.
- Output per window: `layers/output/geotiff.tif` (and
  `layers/output_distilled/geotiff.tif`), 128-band int8, AlphaEarth-style
  quantization of the L2-normalized embedding, 224x224 at 10 m in the window's
  UTM zone.
