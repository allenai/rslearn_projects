These are model configuration files to qualitatively evaluate the effectiveness of
embeddings at capturing fine-grained details. It focuses on training models on the
WorldCover data (for Ai2-internal use, it is available on WEKA at
`/weka/dfive-default/rslearn-eai/datasets/worldcover/`).

- `config_worldcover_ps1.yaml`: patch_size=1, linear layer (768 -> 13).
- `config_worldcover_ps2.yaml`: patch_size=2, bilinear interpolation + linear layer (768 -> 13).
- `config_worldcover_ps4.yaml`: patch_size=4, bilinear interpolation + linear layer (768 -> 13).
- `config_worldcover_ps4_bicubic.yaml`: patch_size=4, bicubic interpolatoin + linear layer (768 -> 13).
- `config_worldcover_ps4_twolayer.yaml`: patch_size=4, linear layer (768 -> 768) + upsample features + linear layer (768 -> 13).
- `config_worldcover_ps4_fourlayer.yaml`: patch_size=4, like twolayer but four layers total (two before upsampling, two after).
- `config_worldcover_ps4_reshape.yaml`: patch_size=4, linear layer (768 -> 13x4x4), reshape to get 10 m/pixel output.
- `config_worldcover_ps4_finetune.yaml`: patch_size=4 with upsampling decoder (UNetDecoder), full fine-tuning.