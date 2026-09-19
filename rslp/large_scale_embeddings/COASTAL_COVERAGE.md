Coastal Water Coverage
======================

A plan for extending the embeddings archive over coastal water, to match the coverage
AlphaEarth Foundations publishes, and the evidence behind the cost estimates. Nothing
here is implemented yet.

The gap
-------

`list_kept_crops` in `tiling.py` drops a crop only when *every* point on a
`LAND_STEP_SIZE` (256 px, 2.56 km) lattice inside it is ocean. Any crop holding even a
sliver of land is processed in full, water included. So nearshore water is already in
the archive: Africa's land area is roughly 80,410 crop-equivalents against 85,602 crops
actually kept, which puts about 5,200 crops of water in the store already.

What is missing is water beyond the outermost land-touching crop. Because coastlines are
fractal, that boundary sits within about one crop width (20.48 km) of land almost
everywhere, and the omission grows with distance offshore.

Filling it is additive, not a rewrite
-------------------------------------

**Skipped crops were never written.** The store is sparse and a shard is exactly one
prediction window, one object, one writer, so writing a previously skipped crop creates
a new object and touches nothing that exists. No marker surgery, no rerun of land, no
concurrent-writer risk.

This is distinct from *nodata pixels inside written crops* (`-128` where every S2 mosaic
was empty). Those live inside existing shards and could only be filled by rewriting
whole shards. That is a separate and harder problem, and it is not the coastal gap.

The consequence is that the fill runs as its own pass with its own `completed_path`, for
example `completed_<year>_coastal/`. The two passes never collide and each is
independently resumable. Do not delete or edit the land pass's markers.

The mask
--------

Use AlphaEarth's own `sea_in_footprint` polygons rather than a distance buffer. It
matches AEF by construction and captures enclosed and shelf seas that a fixed buffer
would miss. The crop rule for the fill pass is:

    keep crop  iff  inside(sea_in_footprint)  AND NOT  current_land_rule(crop)

Both terms are pure functions of the masks, so the pass enumerates exactly the
previously skipped crops with no store queries and no risk of redoing land. Keep
`job_size` at 4096 so overhead amortises over up to four new crops; jobs containing no
new crops are never enumerated.

Source data is `aef_2025_coverage.{geojson,parquet}`, derived from the AEF tile index
(34,155 COGs for 2025) overlaid on Natural Earth 10m land, EPSG:4326. It must be copied
somewhere durable (GCS alongside the store) before the pipeline can read it; a local
Downloads directory is not a dependency the workers can resolve.

Scale
-----

AEF's own pixel probes put usable water at about 18.3 M km2 of the 45.4 M km2
`sea_in_footprint`, concentrated within 25 km of shore and in enclosed and shelf seas.
Applying that 0.40 usable fraction:

| scope  | sea_in_footprint | usable crops | vs current |
| ------ | ---------------- | ------------ | ---------- |
| Africa | 4 to 5.4 M km2   | 4,000-5,000  | about +5%  |
| global | 45.4 M km2       | ~45,000      | about +12% |

These are approximations from a crude geodesic calculation and should be treated as
order-of-magnitude. The figures in the AEF coverage README are authoritative.

Sequencing
----------

**The fill must complete before the global PCA fit.** PCA fit on land-only embeddings
will not represent water, and the rendered pyramids would look wrong over the new areas.
Ordering is: finish the land pass, coastal fill, then PCA.

This also argues for folding the wider mask into the *first* pass of every region not
yet started, so only regions already processed under the old mask ever need a separate
fill. For the 2025 run that means Africa alone.

Risks worth pricing first
-------------------------

`min_matches: 1` on all three sources means a crop with no imagery is skipped as missing
data. Sentinel-2 covers coastal water, but Sentinel-1 RTC and Landsat coverage offshore
is less certain, and beyond some distance the binding constraint becomes data
availability rather than our mask. AEF's probes showing non-nodata concentrated within
25 km suggest the same limit applies to us. Sample offshore crops against the datasets
API before committing to a wide mask.

The checkpoint was trained with land-heavy sampling, so water embeddings may be less
meaningful than land ones even where imagery exists. Worth a sanity check on a small AOI
before spending a full pass.

Implementation sketch
---------------------

1. Mask loader reading the coverage geojson. `shapely` is already a dependency; plain
   `json` plus `shapely` avoids adding `geopandas` or `pyarrow`.
2. A `coastal_mask_path` option threaded through `tiling`, `write_jobs` and the
   supervisor config.
3. The `AND NOT` predicate above, so the pass enumerates only newly eligible crops.
4. Tests covering the predicate, including that a land crop is never re-enumerated.
5. Thin image rebuild.

Open decisions
--------------

- Use `sea_in_footprint` as-is, or a distance threshold instead.
- Run Africa's fill separately, or wait and sweep every region at the end.
