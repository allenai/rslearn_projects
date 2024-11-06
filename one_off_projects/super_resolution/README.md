This is code for building the dataset here:
https://huggingface.co/datasets/allenai/s2-naip

It includes Sentinel-2, NAIP, Sentinel-1, Landsat, OpenStreetMap, WorldCover.

`us_dataset/` contains the code for building the actual dataset along with
documentation, while `global_windows/` is when we wanted to test model performance in a
few locations outside the US (the dataset only covers the US so this was separate from
the dataset itself).
