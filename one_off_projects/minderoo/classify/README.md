The Minderoo rotated bounding box vessel detection model produces lots of false positives.
We tried adding negative examples but it actually increases the error rate instead of reducing it.
So now we try to train post-process classification model similar to what we're doing with Sentinel-2 and Landsat vessel detection.
