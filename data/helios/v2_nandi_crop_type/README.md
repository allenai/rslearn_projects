
20250729
--------

The updated `finetune_s2_20250815.yaml` and `finetune_s1_s2_20250815.yaml` is trained to perform segmentation.

The dataset consists of 32x32 windows centered at sparse point crop type labels.

The previous model is trained for classification, where it inputs the 4x4 patch at the
center of the window and predicts the crop type. This performs well, but deploying the
model is not clean since the classification task is intended to produce a vector
output.

Thus, the model is now updated to perform segmentation. It still inputs 4x4 patches,
and still essentially makes a classification prediction since it outputs logits via a
pooling decoder and the logits are copied to all of the output pixels.

We tried to train the model to input random 16x16 or 4x4 patches containing the labeled
pixel (with the other pixels marked invalid), hoping that it would have the same
performance but be able to segment more than one pixel on each forward pass. However,
this approach provided lower performance (81% instead of 85% accuracy).

20251002
--------

The config `finetune_s2_20251001.yaml` was created for Helios finetuning using only the original ground-truth 10 m pixels (6 categories, ~6K samples). A new split tag `helios_split` was added.
This dataset is also used for KNN classification in Helios. The window size and patch size were adjusted to be consistent with KNN.

20251020
--------

The config `finetune_s2_20251020.yaml` was created after we switched to the spatial splitter (128 x 128 pixel) which better mimic our inference set up. The freeze for 20 epochs and then unfreeze strategy seems working well.
