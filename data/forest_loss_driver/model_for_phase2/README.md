These are model configs for training the model used to predict the second phase of
annotation in Brazil/Colombia.

So we train it on Peru + (initial 500 in Brazil/Colombia) then use the model to figure
out what else to annotate.

See "Select Additional Examples to Label" in `rslp/forest_loss_driver/README.md` for
more details.
