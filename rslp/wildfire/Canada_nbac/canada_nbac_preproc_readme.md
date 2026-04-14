### Preliminary note on temporal aggregation

The temporal binning only considers when a fire **started**, not how long it
lasted.  A fire spanning 16 days is assigned to a single bin (the one
containing its `start_date`); it is **not** replicated across subsequent bins
it may overlap. For the current use case this is acceptable and actually what
we want: if a cell is still burning after the first temporal bin, then it is
discarded.
TODO: we just have to make sure the negative sampling do not sample the discarded
cells.

A more subtle consequence comes from the nature of most fire polygon dataset. Each fire
event usually a single `start_date` / `last_date` pair for the **entire fire event**,
not per grid cell.  When a fire progressively burns through neighbouring cells
A → B → C → D over several weeks, every cell inherits the event's overall
ignition date after the spatial join in Step 1.  All four cells are therefore
snapped to the **same** temporal bin, even though B, C, and D may not have
started burning until later periods.
Side note: per-cell burn timing is available in the underlying MCD64A1 pixel-level
product but is lost when using FIREd's event-level aggregation.



## 0. Create adaptive Grid

Update global_config.yaml with the right bounds and grid parameters

```shell
cd ./rslp/wildfire/Canada_nbac && \
python -m data_preproc_script.preprocess.create_grid
```

## 1. Burned-area preprocessing (rename columns + spatial join + temporal merge)

Update ba_preprocess.yaml with the right config parameters
```shell
cd ./rslp/wildfire/Canada_nbac && \
python -m data_preproc_script.preprocess.burned_area
```

## 2. Temporal binning (snap fire dates to fixed-interval grid)
Update ba_preprocess.yaml with the right config parameters

Discretises continuous fire start dates into fixed-width time windows
(e.g. 8 days, controlled by `offset` in `ba_preprocess.yaml`).
For each `(grid_id, time_window)` the label answers:
*"Was this cell burned in the next N days?"*

Outputs: `temporal_grid_samples_path` (geo) and `temporal_bin_mapping_path` (CSV).

```shell
cd ./rslp/wildfire/Canada_nbac && \
python -m data_preproc_script.preprocess.temporal_grid_agg
```

## 3. Negative sampling (unified three-tier)

Draws negative (non-fire) grid cells at a configurable ratio to positives,
split equally into three tiers:

| Tier | Strategy | Reference distribution |
|------|----------|----------------------|
| 1 | FWI-uniform random | Equal draws from FWI quantile buckets of the candidate pool |
| 2 | LC + month matched | Matches `P(LC, month \| y=1, year, region)` of positives |
| 3 | LC + month + FWI-hard | Matches `P(LC, month, FWI_bin \| y=1, year, region)` with optional upward FWI bias |

All tiers preserve regional balancing (same neg:pos ratio per region).
Hierarchical backoff fills sparse strata: exact joint -> drop FWI ->
coarsen month to season -> LC only -> random.

Outputs: `negative_samples_unified.gdb` (with a `tier` column).

Update Config: `configs/sampling_unified.yaml`.

```shell
cd ./rslp/wildfire/Canada_nbac && \
python -m data_preproc_script.sampling.negative_sampling
```


On beaker cluster:
```shell
source ./.venv/bin/activate
python -m rslp.main common launch_data_materialization_jobs \
  --image hadriens/rslpomp_hspec_260226_fus2 \
  --ds_path ./ \
  --hosts+=<your-cluster-host> \
  --command+=bash \
  --command+='-c' \
  --command+='cd ./rslp/wildfire/Canada_nbac && python -m data_preproc_script.sampling.negative_sampling --select-year 2023'
  ```
