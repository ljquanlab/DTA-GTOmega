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
Or we upload the preprocessed features to the drive. Download move them into ./data folder. Please check ./data/path.png to see the an example.



Davis_omega: Node features + edges features (40 GB)
- https://pan.baidu.com/s/1BBbETPfRAmr6c4QY8waLJA
- password: CUTE

We utilized the complete file to run our experiments, however file size can be reduced to less than 1GB by removing edge features if your device does not has this size of memory, the following are jupyter codes.
```commandline
!unzip Davis_omega
import pickle as pkl
import os
os.chdir('Davis_omega/omega_2')
for fi in os.listdir('./'):
  with open(fi, 'rb') as f:
    o = pkl.load(f)
  o['struct_edge'] = 0
  with open(fi, 'wb') as f:
    pkl.dump(o, f)
```

Then you should modify the code in Dataloader.DatasetDefination in line 366 and line 395:

```commandline
if dataset != 'DrugBank' and dataset != 'BindingDB' and dataset != 'Davis':
```

```commandline
# 366
if dataset != 'DrugBank' and dataset != 'BindingDB' and dataset != 'Davis':

# 395
if dataset == 'DrugBank' or dataset == 'BindingDB' or dataset=='Davis':
```

BindingDB_omega: Only Node features (390 MB). (Since the edge features needs about >200GB. We cannot save it.)
- https://drive.google.com/file/d/1yJohLt2_Fot9IwzLJYpL5TI4cQ3-q2z3/view?usp=drive_link

KIBA_omega: Node features + edges features (23 GB)
- https://pan.baidu.com/s/1j-P8-AFt2jh92-QjwMiVgA
- password: CUTE

# Environment
- cuda:12.2
- memory > 50G (You can avoid such large memory requirements by reducing the file size as mentioned above.)
- device: Tesla V100 32GB/ Tesla A100 40GB

# Package
Please Check config/Environment.ipynb

# Reproduction

We provide 5-15 trained models for reproduction.

1. Change the 'split_strategy' in the following scripts to reproduce the results.

2. Saved model for davis have been uploaded in saved_pth/Davis_repr
```commandline
python scripts/davisSerial.py
```

Saved model for BindingDB:
- https://drive.google.com/file/d/1GC5f1A9CsORmLwy4oXgEjcnx0YJsYTFK/view?usp=sharing
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
