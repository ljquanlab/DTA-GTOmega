# Data
Some files are > 25mb such as the train.csv for KIBA, therefore they need to be decompressed. 
The software Bandzip is recmmonded to decompress.
- data/KIBA/split_data/cold_drug/train.csv
- data/KIBA/split_data/cold_target/train.csv
- data/KIBA/split_data/cold_drug_target/train.csv


# Feature Generation

In the main directory
```commandline
mkdir data/Davis_omega
```

Then please check https://github.com/yelujiang/OmegaFeatures to generate features.
Or we upload the preprocessed features to the drive.
Davis_omega:
- https://pan.baidu.com/s/1BBbETPfRAmr6c4QY8waLJA 提取码:
- password: CUTE 

# Environment
- cuda:12.2
- memory > 50G
- device: Tesla V100 32GB/ Tesla A100 32GB

# Package
Please Check config/Environment.ipynb

# Reproduction
```commandline
python scripts/davisSerial.py
```

or Retrain the model:
```commandline
python scripts/davis.py
```
