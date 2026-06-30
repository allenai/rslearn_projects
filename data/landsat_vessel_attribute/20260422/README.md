These configs are the same as the best one from the 20260330 sweep, except they train
on additional vessel detections that Hunter provided.

config_old.yaml excludes the new data for training while config_new.yaml includes it.
I re-trained both because I also updated the splits to make sure that neither the same
scene nor the same vessel (based on MMSI) appears in both train and val. This also
means we are throwing some data away (e.g. MMSI indicates it should be in train but
scene ID indicates it should be in val => that detection is not used).
