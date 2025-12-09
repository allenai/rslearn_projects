## Tolbi Project

This project focuses on mapping a variety of cash crops (palm oil, cacao, rubber
, etc) within the Ivory Coast region. The Tolbi team sent us mainly positive
samples (and some negative samples, mainly water, building, and soil), and we
need to extract negative samples to ensure the model can correctly identify the
cash crops. The main challenge is to differentiate cacao from shrub, and palm
oil / rubber from natural trees and other tree crops.

### Extract LULC Labels from WorldCover

WorldCover 16.5M labels has been saved on WEKA:
`/weka/dfive-default/rslearn-eai/artifacts/WorldCover/final_reference_data.csv`

Get the WorldCover labels within the Ivory Coast:

```
python rslp/tolbi/scripts/process_worldcover.py --input ~/Downloads/final_reference_data.csv --output ~/Downloads/final_reference_data_ivory_coast.csv --min-lat 4.61955154569263 --max-lat 10.525361785272338 --min-lon -8.575678982928736 --max-lon -2.665272278762558
```

### Prepare Positive and Negative Samples

As an initial approach, we focuses on classifying plam oil, rubber, cacao, tree,
shrub, and others. Though the WorldCover samples may include tree samples that
are actually palm oil / rubber, or shrub that are actually cacao, it's hard to
filter out those samples (maybe the Tolbi team can help on this), give that the
natural forest is heavily fragmented by cacao, rubber, and palm.

TODO: (1) Tolbi team can help remove WorldCover clusters (each cluster includes
10x10 pixels) that are actually cash crops, (2) Another approach is to keep only
non-tree, non-shrub samples from WorldCover, and later add other trees or tree
crops and shrub samples (new annotations or from existing forest or shrub
dataset). This may be easier and faster than (1) as we don't need to go through
all clusters from WorldCover (about 1K).

Some data quality issues:

1. Overlapped polygons in ground-truth and remote sensed labels
2. Overlapped polygons in ground-truth labels (rubber1, rubber2)

After getting all the points within polygons, we got about 528K positive samples
(228K cacao, 227K palm oil, and 72K rubber). There're also 100K WorldCover
samples (tree: 30K, shrub: 27K, others: 35K).

We can start by using 10K samples per category, which results in 60K samples.

```
python rslp/tolbi/scripts/create_samples.py --pos_geojson_dir /Users/yawenz/Downloads/local/rslearn_projects/rslp/tolbi/data/geojsons/ --pos_output /Users/yawenz/Downloads/local/rslearn_projects/rslp/tolbi/data/csv/positive_samples.csv --neg_input /Users/yawenz/Downloads/local/rslearn_projects/rslp/tolbi/data/csv/final_reference_data_ivory_coast.csv --neg_output /Users/yawenz/Downloads/local/rslearn_projects/rslp/tolbi/data/csv/negative_samples.csv --sample_size 10000 --combined_output /Users/yawenz/Downloads/local/rslearn_projects/rslp/tolbi/data/csv/combined_samples.csv
```




