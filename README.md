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
Due to the limit of github, we will upload the preprocessed features to the drive.

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
