Just a bunch of model configs to compare Molmo to other models.


## 20241025

In this first comparison, we compared Swin-v2-Base (either SatlasPretrain or ImageNet),
Molmo (allenai/Molmo-7B-D-0924), and CLIP (openai/clip-vit-large-patch14-336).

The tasks are:
* Sentinel-2: vessel detection
* Maxar: vessel detection (Minderoo)
* Landsat: vessel detection
* Maxar: ecosystem category segmentation
* Maxar: vessel post-processing classification
* Landsat: vessel post-processing classification

Results are here: https://wandb.ai/eai-ai2/molmo_comparison


### Sentinel-2 vessel detection

Molmo starts out performing better than all baselines, even with backbone frozen, but
with more epochs both CLIP and SatlasPretrain RGB achieve the same performance. CLIP
actually provides the highest performance, but it is a marginal improvement (1%).

Additionally, here we are training SatlasPretrain to match CLIP's hyperparameters which
means training on smaller patches (192x192) and RGB only. With all bands and bigger
patches, it seems that SatlasPretrain provides better performance, but it may need more
experimentation.


### Maxar vessel detection (Minderoo)

CLIP and SatlasPretrain (aerial image RGB) provide about the same F1.

In mAP, SatlasPretrain actually provides the best performance.

For frozen backbone, SatlasPretrain provides the best performance on both metrics.


### Landsat vessel detection

On F1, Molmo and CLIP provide similar performance, and beat ImageNet by 5%. Results are
similar on mAP.

I didn't try the SatlasPretrain model for Landsat since ImageNet previously showed
higher performance but it could be worth experimenting with it again.


### Maxar ecosystem category segmentation

Only tried Molmo and SatlasPretrain (aerial RGB) for this. Molmo provides high
performance even with frozen backbone, up to 58% accuracy, while SatlasPretrain is 56%
accuracy.

Note that this task has only about 80 training examples and 13 validation.


### Maxar vessel post-processing

Molmo and CLIP provide 96% accuracy while SatlasPretrain (aerial RGB) only provides 92%
accuracy. This is the main task where Molmo and CLIP seem to provide a clear
improvement over SatlasPretrain.


### Landsat vessel post-processing

Here the Swin-v2-Base is initialized from the weights of the model trained for Landsat
vessel detection.

This model provides 83% accuracy. Molmo also achieves 83% accuracy, which is impressive
given that it doesn't see the detection dataset.

More experimentation may be needed, e.g. seeing if fine-tuning first on Landsat vessel
detection and then on vessel classification helps Molmo to beat the Swin-v2-Base on
both tasks.
