# Nandi - Nano
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625/ --repeats 1 --samples 0 --k 3 --embed_fname olmoearth_v1_nano.npy --label_key category --split_key helios_split --groups groundtruth_polygon_split_window_32 --metric accuracy --n_bootstraps 1000

# Nandi - Tiny
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625/ --repeats 1 --samples 0 --k 3 --embed_fname olmoearth_v1_tiny.npy --label_key category --split_key helios_split --groups groundtruth_polygon_split_window_32 --metric accuracy --n_bootstraps 1000

# Nandi - Base
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625/ --repeats 1 --samples 0 --k 3 --embed_fname olmoearth_v1_base.npy --label_key category --split_key helios_split --groups groundtruth_polygon_split_window_32 --metric accuracy --n_bootstraps 1000

# Nandi - GSE
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625/ --repeats 1 --samples 0 --k 3 --embed_fname gse --label_key category --split_key helios_split --groups groundtruth_polygon_split_window_32 --metric accuracy --n_bootstraps 1000

# AWF - Nano
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/awf_2023/ --repeats 1 --samples 0 --k 3 --embed_fname olmoearth_v1_nano.npy --label_key lulc --split_key helios_split --groups 20250822 --metric accuracy --n_bootstraps 1000

# AWF - Tiny
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/awf_2023/ --repeats 1 --samples 0 --k 3 --embed_fname olmoearth_v1_tiny.npy --label_key lulc --split_key helios_split --groups 20250822 --metric accuracy --n_bootstraps 1000

# AWF - Base
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/awf_2023/ --repeats 1 --samples 0 --k 3 --embed_fname olmoearth_v1_base.npy --label_key lulc --split_key helios_split --groups 20250822 --metric accuracy --n_bootstraps 1000

# AWF - GSE
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/awf_2023/ --repeats 1 --samples 0 --k 3 --embed_fname gse --label_key lulc --split_key helios_split --groups 20250822 --metric accuracy --n_bootstraps 1000

# Ecosystem - Nano
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/geo/dataset_v2/dataset/ --repeats 1 --samples 0 --k 3 --embed_fname olmoearth_v1_nano.npy --label_key label --split_key split --metric accuracy --n_bootstraps 1000

# Ecosystem - Tiny
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/geo/dataset_v2/dataset/ --repeats 1 --samples 0 --k 3 --embed_fname olmoearth_v1_tiny.npy --label_key label --split_key split --metric accuracy --n_bootstraps 1000

# Ecosystem - Base
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/geo/dataset_v2/dataset/ --repeats 1 --samples 0 --k 3 --embed_fname olmoearth_v1_base.npy --label_key label --split_key split --metric accuracy --n_bootstraps 1000

# Ecosystem - GSE
python one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/get_balanced_accuracy.py --ds_path /weka/dfive-default/rslearn-eai/datasets/geo/dataset_v2/dataset/ --repeats 1 --samples 0 --k 3 --embed_fname gse --label_key label --split_key split --metric accuracy --n_bootstraps 1000
