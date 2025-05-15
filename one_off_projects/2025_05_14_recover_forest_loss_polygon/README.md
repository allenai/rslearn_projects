In gs://rslearn-eai/datasets/forest_loss_driver/dataset_v1/20240912/ we only saved the
mask.png and not the vector polygon because that's all we need for training.

But now it is useful to have vector polygon in ES Studio to support the overlay display.

So this code is just to compute the polygon from mask.png and add it to
layers/label/data.geojson in each window directory.
