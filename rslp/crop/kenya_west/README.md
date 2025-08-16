# Western Kenya Crop Type Segmentation

The dataset consists of 152 samples from 2021 and 267 samples from 2023, mainly for the long-rain season (March to August) in Western Kenya. Each sample is 50x50 pixel at 10m resolution. The first target is on identification of intercropping with a focus on Maize and Beans amongst other crops.

There're 11 classes (from 0 to 10 in geotiff file, we added 1 to them to use 0 as no data, so the new range is 1 to 11):

	•	Banana: 1
	•	Bean: 2
	•	Cassava: 3
	•	Cowpea: 4
	•	Maize: 5
	•	Maize-Banana: 6
	•	Maize-Bean: 7
	•	Maize-Cassava: 8
	•	Maize-Cowpea: 9
	•	Maize-Potato: 10
	•	Potato: 11


```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 1 --encoder_embedding_size 768 --image_name yawenzzzz/rslphelios1 --config_paths+=data/helios/v2_kenya_west_crop_type/frozen_s2.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_08_13_helios_finetuning --experiment_id v2_western_kenya_crop_type_segmentation_helios_base_S2_ts_ps1_frozen_crop_maize_1
```

