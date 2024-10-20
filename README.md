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

Davis_omega: Node features + edges features (40 GB), here edges features are not needed
- https://pan.baidu.com/s/1BBbETPfRAmr6c4QY8waLJA
- password: CUTE

BindingDB_omega: Only Node features (390 MB)
- https://drive.google.com/file/d/1yJohLt2_Fot9IwzLJYpL5TI4cQ3-q2z3/view?usp=drive_link

# Environment
- cuda:12.2
- memory > 50G
- device: Tesla V100 32GB/ Tesla A100 40GB

# Package
Please Check config/Environment.ipynb

# Reproduction
1. Change the 'split_strategy' in the following scripts to reproduce the results.

2. Saved model for davis have been uploaded in saved_pth/Davis_repr
```commandline
python scripts/davisSerial.py
```

Saved model for BindingDB:
- https://drive.google.com/file/d/1QPzeAybT-Tp9OB46AKIBKx7yWKj9eURI/view?usp=sharing
- unzip BindingDB_repr.zip
- mv BindingDB_repr ./saved_pth

```commandline
python scripts/bindingDBSerial.py
```

or Retrain the model:
```commandline
python scripts/davis.py
python scripts/bindingDB.py
```
