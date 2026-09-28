# Nandi v2 — running county inference

Hand-off note for whoever runs inference after the fine-tune finishes. Everything below
has been executed and verified end-to-end on one county window except the final
full-county command, which is the same command with the window filter removed.

## 1. Environment

There is **no working shared env** for this — `rslearn/.venv` alone cannot train or
predict. Three things are wrong with it and all three are patched by a `PYTHONPATH`
overlay on weka; the venv itself is untouched.

| problem | why it breaks | fix in the overlay |
|---|---|---|
| venv has `torch 2.12.0+cu130`, driver is CUDA 12.8 | `torch.cuda.is_available()` is `False` | `torch 2.8.0+cu128` + `torchvision 0.23.0+cu128` (rslearn needs `torch>=2.7.0`) |
| venv has `jsonargparse 4.49.0`, rslearn needs `>=4.50.0` | `rslearn/arg_parser.py` overrides `parse_string(content=...)`; 4.49 calls it `cfg_str`, so **every** `rslearn model *` dies with `missing 1 required positional argument: 'content'` | `jsonargparse 4.52.0` |
| missing packages | import errors | `einops`, `wandb`, `scipy`, `rtree`, `matplotlib`, `planetary_computer`, `pystac_client`, `olmoearth_pretrain_minimal` |

Order matters: `torchcu128` must come first so it shadows the venv's torch.

```bash
export ENV=/weka/dfive-default/yawenz/envs/nandi_v2
export PYTHON=/weka/dfive-default/yawenz/rslearn/.venv/bin/python
export PYTHONPATH=$ENV/torchcu128:$ENV/extra2:/weka/dfive-default/yawenz/rslearn_projects
```

Verify before doing anything else — this should print `True` and the GPU name:

```bash
$PYTHON -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.8.0+cu128 True NVIDIA A100-SXM4-80GB
```

Imports are slow the first time (torch off weka takes several minutes). That is normal,
not a hang.

## 2. Inputs (all already built — do not rebuild)

- Dataset: `/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20260921`
- Inference group: `nandi_county` — **81 windows of 1024×1024 at 10 m, EPSG:32636,
  all materialized** (12 monthly Sentinel-2 mosaics each, 2022-09 → 2023-08)
- Checkpoint: `/weka/dfive-default/yawenz/projects/20260921_nandi_v2/finetune_s2_oe12base_ws16_ps1_layerdecay/best.ckpt`
- Tile store is already populated (~50 GB), so nothing needs downloading.

## 3. Run inference

```bash
export RSLP_PREFIX=/weka/dfive-default/yawenz/projects
export WANDB_MODE=offline          # or online; a WANDB_API_KEY is in the container env
cd /weka/dfive-default/yawenz/rslearn_projects

$PYTHON -m rslearn.main model predict \
  --config data/helios/v3_nandi_polygon/common.yaml \
  --config data/helios/v3_nandi_polygon/finetune_s2.yaml \
  --config data/helios/v3_nandi_polygon/predict_s2.yaml
```

`predict` resolves `load_checkpoint_mode` to `best`, so it picks up `best.ckpt` from
`$RSLP_PREFIX/<project_name>/<run_name>/` on its own — do **not** pass `--ckpt_path`.

Expect **~2m45s per window** (measured while the GPU was also training; should be
faster on an idle GPU), so roughly **2–4 hours** for all 81. Run it detached:

```bash
setsid nohup <the command above> > predict.log 2>&1 < /dev/null &
```

To smoke-test on a single window first, add a fourth config:

```yaml
# one_window.yaml
data:
  init_args:
    predict_config:
      names: ["68608_0_69632_1024_2023-03-01T00:00:00+00:00_2023-03-31T00:00:00+00:00"]
```

## 4. Output

One GeoTIFF per window at
`<dataset>/windows/nandi_county/<window>/layers/output/class/geotiff.tif` —
`uint8`, 1024×1024, EPSG:32636, pixel value = class ID.

Class IDs are the index in `CLASS_NAMES` in `rslp/nandi/classes.py`; `CLASS_COLORS`
there is the matching colormap. Order: Maize, Sugarcane, Legumes, Vegetables, Coffee,
Tea, Grassland, Trees, Shrubland, Water, Built-up.

To get one county-wide raster, mosaic the 81 GeoTIFFs (`gdal_merge.py`, `gdalbuildvrt`,
or `rslp/nandi/scripts/merge_geotiff.py`). `rslp/nandi/scripts/cleanup_geotiff.py` is
still useful for removing small islands.

## 5. Things that will bite you

- **`selector: ["segment"]` on the RslearnWriter is required.** The model is a
  `MultiTaskModel`, so its prediction is a dict keyed by task name. Without the
  selector the writer passes that dict to the merger and you get
  `AttributeError: 'dict' object has no attribute 'shape'` — and it fails only at the
  *last* batch of a window, after all the compute is spent. Already set in
  `predict_s2.yaml`.
- **Do not set `use_map_all_crops_dataset: true`.** It re-reads 12 item groups × 3 band
  sets per crop; at 16×16 crops that is ~36 GeoTIFF reads to produce 256 pixels and
  runs ~27× slower (0.2 it/s vs 1.37 it/s). The config deliberately leaves both crop
  wrappers off so `IterableAllCropsDataset` loads each window once and slices crops in
  memory (~600 MB per data worker).
- **`crop_size: 16` must match the training window size.** The model was trained on
  16×16 windows at `patch_size: 1`; a different inference crop changes the token count
  and the attention pattern.
- **Overlap is within a window only.** `RasterMerger` trims `overlap_pixels // 2` from
  each crop edge, but predictions are *not* blended across separate rslearn windows.
  That is why the windows are 1024² rather than the v1 pipeline's 128² — it keeps the
  un-overlapped window border a negligible fraction of the map.
- **W&B run resume.** rslearn stores the run id in
  `$RSLP_PREFIX/<project>/<run>/wandb_id` and resumes with `resume='must'`. If a run
  ever started offline, that id does not exist server-side and switching to
  `WANDB_MODE=online` crashes at startup. Delete `wandb_id`; the model still resumes
  from `last.ckpt`.

## 6. Known gaps

- 65 of 2,180 training windows (3%) never materialized. They all needed one Sentinel-2
  item served only from the requester-pays `s3://sentinel-s2-l2a` JP2 bucket. The
  dataset config now filters these out with
  `query: {"earthsearch:boa_offset_applied": {"eq": true}}` (which also keeps the BOA
  radiometric offset consistent across the series), but the training set was left as-is
  so the linear-probe and fine-tune runs stayed comparable. Re-preparing the
  `polygon_grid_16` group would recover them.
- Vegetables (396 labelled px total) and Water (clustered in a few blobs) have very thin
  val/test support. Treat their per-class metrics as directional.
