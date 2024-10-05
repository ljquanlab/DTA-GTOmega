import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from Dataloader.DatasetDefination import DrugDataset, collate_fn, ProteinFeatureManager, get_mol_edge_list_and_feat_mtx
import pandas as pd
import pickle
from rdkit import Chem
import numpy as np
import itertools
from torch.utils.data import DataLoader
'''
Create DataLoader
'''

dataset = 'Davis'
threthod = 0.5
pocket = False
split_strategy = 'cold_drug'
# split_strategy = 'cold_drug_target'
split_strategy = 'cold_target'

print('Start loading all smiles.')
with open(f'data/{dataset}/compound_to_smiles.pkl','rb') as f:
    df_drugs_smiles = pickle.load(f)

all_smiles = list(df_drugs_smiles.keys())
protein_feature_manager = ProteinFeatureManager(f'data/{dataset}', pocket=pocket, omega_pdb=True)
print('Load all smiles done.')

print('Start Transforming smiles to Mol Obeject')
import os
if os.path.exists(f'data/{dataset}/saved_pickle/all_smiles_mols.pkl'):
    with open(f'data/{dataset}/saved_pickle/all_smiles_mols.pkl', 'rb') as f:
        drug_id_mol_graph_tup = pickle.load(f)
else:
    drug_id_mol_graph_tup = [(smiles, Chem.MolFromSmiles(smiles.strip())) for smiles in all_smiles]
    with open(f'data/{dataset}/saved_pickle/all_smiles_mols.pkl', 'wb') as f:
        pickle.dump(drug_id_mol_graph_tup, f)

print('End Transforming.')

print('Start Atom Feature Construction.')
if os.path.exists(f'data/{dataset}/saved_pickle/MOL_EDGE_LIST_FEAT_MTX.pkl'):
    with open(f'data/{dataset}/saved_pickle/MOL_EDGE_LIST_FEAT_MTX.pkl', 'rb') as f:
        MOL_EDGE_LIST_FEAT_MTX = pickle.load(f)
    with open(f'data/{dataset}/saved_pickle/TOTAL_ATOM_FEATS.pkl', 'rb') as f:
        TOTAL_ATOM_FEATS = pickle.load(f)
else:
    AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
    AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
    ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])
    AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
    max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
    max_valence = max(max_valence, 9)
    AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

    MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
    MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
    MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
    MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0

    MOL_EDGE_LIST_FEAT_MTX = {smiles: get_mol_edge_list_and_feat_mtx(mol)for smiles, mol in drug_id_mol_graph_tup}
    MOL_EDGE_LIST_FEAT_MTX = {smiles: mol for smiles, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}
    
    TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])
    with open(f'data/{dataset}/saved_pickle/MOL_EDGE_LIST_FEAT_MTX.pkl', 'wb') as f:
        pickle.dump(MOL_EDGE_LIST_FEAT_MTX, f)
    with open(f'data/{dataset}/saved_pickle/TOTAL_ATOM_FEATS.pkl', 'wb') as f:
        pickle.dump(TOTAL_ATOM_FEATS, f)
print('End Cnstuction')

print('Start Loading Dataset')
dti_train = pd.read_csv(f'data/{dataset}/split_data/{split_strategy}/train.csv')
dti_val = pd.read_csv(f'data/{dataset}/split_data/{split_strategy}/valid.csv')
dti_test = pd.read_csv(f'data/{dataset}/split_data/{split_strategy}/test.csv')

d_train, t_train, y_train = dti_train['Drug'], dti_train['Target'], dti_train['Y']
d_test, t_test, y_test = dti_test['Drug'], dti_test['Target'], dti_test['Y']
d_val, t_val, y_val = dti_val['Drug'], dti_val['Target'], dti_val['Y']

stand_label = False
train_dataset = DrugDataset(d_train, t_train, y_train, MOL_EDGE_LIST_FEAT_MTX, protein_feature_manager, threthod, stand_label=stand_label)
test_dataset = DrugDataset(d_test, t_test,y_test, MOL_EDGE_LIST_FEAT_MTX, protein_feature_manager, threthod)
val_dataset = DrugDataset(d_val, t_val, y_val, MOL_EDGE_LIST_FEAT_MTX, protein_feature_manager, threthod)

import resource
import random
import torch
from torch.optim import Adam, AdamW
import torch.optim as optim
import torch.nn as nn
import json
import os
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import logging

with open('config/Train_Davis_repr.json') as f:
    params = json.load(f)
workers = params['workers']
batch_size = params['batch_size']
SEED = params['seed']

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# os.environ['PYTHONHASHSEED'] = str(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=workers)
print('Create Dataloader Done.')

if split_strategy == 'cold_drug' or split_strategy == 'cold_drug_target':
    all_drugs = list(set(list(d_train) + list(d_val))) + list(set(d_test))
    compound_to_smiles = {}
    for drug in all_drugs:
        compound_to_smiles[drug] = Chem.MolFromSmiles(drug.strip())

    print('Train Set has drugs number:',len(set(d_train)))
    print('Validation Set has drugs number:',len(set(d_val)))
    print('Test Set has drugs number:',len(set(d_test)))

    print('Trainset has unique drugs number (which are not in Testset):',len(set(d_train) - set(d_test)))
    print('Testset has unique drugs number (which are not in Trainset):',len(set(d_test) - set(d_train)))
    print(f'There are {len(all_drugs)} drugs in total.')

    print(f'There are {len(set(all_drugs))} drugs')
    print()
    
if split_strategy == 'cold_target' or split_strategy == 'cold_drug_target':
    all_proteins = list(list(set(t_train)) + list(set(t_val))) + list(set(t_test))
    print(f'There are {len(set(all_proteins))} proteins')
    print('Train Set has proteins/target number:',len(set(t_train)))
    print('Validation Set has proteins/target number:',len(set(t_val)))
    print('Test Set has proteins number:',len(set(t_test)))
    print('Trainset has unique proteins number (which are not in Testset):',len(set(t_train) - set(t_test)))
    print('Testset has unique proteins number (which are not in Trainset):',len(set(t_test) - set(t_train)))
    print()
    
print(f'Split Strategy:{split_strategy}')
print(f'The train size is {len(y_train)}')
print(f'The valid size is {len(y_val)}')
print(f'The test size is {len(y_test)}')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info('This is a start info')

# from model_ import StructDTI
from model_ import StructDTI
import torch.nn.functional as F
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
learning_rate = params['learning_rate']
dim_drug = TOTAL_ATOM_FEATS
print(f'dim_drug: {dim_drug}')
dim_protein = params['dim_protein']
hidd_dim = params['hidd_dim']
edge_dim = params['edge_dim']

head = params['head']
out_features = params['out_features']
kge_dim = head * out_features
head_out_features = [out_features, out_features, out_features, out_features]
heads = [head, head, head, head]
epo_num = params['epo_num']
patient = params['patient']

if edge_dim is not None:
    model = StructDTI(dim_drug, dim_protein, hidd_dim, kge_dim, heads_out_feat_params=head_out_features, blocks_params=heads, edge_dim=edge_dim)
else:
    logger.info('No edge attr.')
    model = StructDTI(dim_drug, dim_protein, hidd_dim, kge_dim, heads_out_feat_params=head_out_features, blocks_params=heads, edge_dim=None)
'''
Multiple-GPU if needed:
    device_ids = [0, 1] 
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to('cuda')
'''
model = model.to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.98, patience=4, min_lr=1e-5, verbose=True)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
logger.info('Training Paragrams Set.')

from tqdm import tqdm
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
# from scipy.stats import spearmanr as R2
import warnings
warnings.simplefilter('ignore', category=UserWarning)
logger.info('Start Training')
val_loss_record = 100.
val_r2_record = -1000.
for epoch in range(epo_num):
    model.train()
    num_batches = len(train_loader)
    train_loss = []
    print('-'*50 + f'Epoch:{epoch} Start'+ '-'*50)
    for batch_idx, batch_data in tqdm(enumerate(train_loader)):
        drug_graphs = batch_data['drug_graphs'].to(device)
        protein_graphs = batch_data['protein_graphs'].to(device)
        labels = batch_data['labels'].to(device).float()
        optimizer.zero_grad()

        outputs = model(drug_graphs, protein_graphs, bi_graphs=None, task='repr')  # Adjust depending on model's forward method
        loss = criterion(outputs.unsqueeze(1), labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss.append(loss.cpu().item())
    print(f'Train Loss: {np.mean(train_loss)}')
    
    model.eval()
    with torch.no_grad():
        val_loss = []
        val_labels = []
        val_pred = []
        for batch_idx, batch_data in tqdm(enumerate(val_loader)):
            drug_graphs = batch_data['drug_graphs'].to(device)
            protein_graphs = batch_data['protein_graphs'].to(device)
            valid_labels = batch_data['labels'].to(device).float()  # 注意标签类型改为float
            
            # batch_size, 1
            outputs = model(drug_graphs, protein_graphs, bi_graphs=None, task='repr')
            # batch_size,
            if stand_label:
                outputs = outputs * train_dataset.std + train_dataset.mean_
            valid_loss = criterion(outputs.unsqueeze(1), valid_labels.unsqueeze(1))
            val_loss.append(valid_loss.cpu().item())
            
            val_labels.extend(valid_labels.cpu().numpy())
            val_pred.extend(outputs.cpu().numpy())
            
        r2 = R2(np.array(val_pred), np.array(val_labels))
        if type(r2) == tuple:
            r2 = r2[0]
        scheduler.step(np.mean(val_loss))
        print(f'Valid Loss: {np.mean(val_loss)}, valid r2:{r2}')
        if np.mean(val_loss) < val_loss_record:
            patient = params['patient']
            val_loss_record = np.mean(val_loss)
            val_r2_record = r2
            print('This epoch save model')
            torch.save(model.state_dict(), f'saved_pth/Davis_repr/{split_strategy}_model.pth')
        else:
            patient -= 1
            if patient == 0:break
        print('-'*50 + f'Epoch:{epoch} End'+ '-'*52)
        print()
logger.info('End Training')


from tqdm import tqdm

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2

# with open('config/Train_Davis_repr.json') as f:
#     params = json.load(f)
    
# dim_drug = params['dim_drug']
# dim_protein = params['dim_protein']
# hidd_dim = params['hidd_dim']
# kge_dim = params['kge_dim']
# head_out_features = params['head_out_features']
# heads = params['heads']
# edge_dim = params['edge_dim']
# device = 'cuda:0'

# model = StructDTI(dim_drug, 
#                  dim_protein,
#                  hidd_dim, 
#                  kge_dim, 
#                  heads_out_feat_params=head_out_features,
#                  blocks_params=heads,
#                  edge_dim=edge_dim
#                 )

serial_check = -1

if serial_check == -1:
    model.load_state_dict(torch.load(f'saved_pth/Davis_repr/{split_strategy}_model.pth',map_location='cpu'))
else:
    model.load_state_dict(torch.load(f'saved_pth/Davis_repr_upload/{split_strategy}_model_{serial_check}.pth',map_location='cpu'))
    
# model = model.to('cuda:0')

model.eval()
with torch.no_grad():
    
    all_labels = []
    all_predictions = []
    
    for batch_idx, batch_data in tqdm(enumerate(test_loader)):
        drug_graphs = batch_data['drug_graphs'].to(device)
        protein_graphs = batch_data['protein_graphs'].to(device)
        # bi_graphs = batch_data['bipartite_graphs'].to(device)
        # batch_size,
        test_labels = batch_data['labels'].to(device).float()

        # batch_size,
        outputs = model(drug_graphs, protein_graphs, bi_graphs=None, task='repr')
        if stand_label:
            outputs = outputs * train_dataset.std + train_dataset.mean_

        all_labels.extend(test_labels.cpu().numpy().tolist())
        all_predictions.extend(outputs.cpu().numpy().tolist())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    Y = all_labels
    P = all_predictions
    print('Test shape:', Y.shape, P.shape)
    
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as R2
from scipy.stats import pearsonr as pear
from scipy.stats import spearmanr as spear
from Dataloader.metrics import *
import numpy as np
m = mean_squared_error(Y, P)
r2 = R2(Y, P)
# Accuracy_dev = CI(Y, P)
MSE = mse(Y, P)
RMSE = rmse(Y, P)
CI = ci(Y, P)
RM2 = rm2(Y, P)
pcc = pear(Y, P)
# r_squared_error(Y, P)
sp = spear(Y, P)
print(f'Davis {split_strategy}, label shape is {Y.shape}, pred shape is {P.shape}')
print('MSE:', m, MSE)
print('R2:', r2, RM2)
print('CI:', CI)

print('PCC:', pcc[0])
print('Spear:', sp[0])

S = P
P = [0 if i<5 else 1 for i in S]
Y = [0 if i<5 else 1 for i in Y]

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import matthews_corrcoef, roc_curve
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, average_precision_score

# S = P
# P = [0 if i<5 else 1 for i in S]
# Y = [0 if i<5 else 1 for i in Y]
AUC_dev = roc_auc_score(Y, S)
fpr_2, tpr_2, thresholds = roc_curve(Y, S)
ROC_AUC = auc(fpr_2, tpr_2)

PRC_dev = average_precision_score(Y, S)
tpr, fpr, _ = precision_recall_curve(Y, S)
PRC_AUC = auc(fpr, tpr)

f1 = f1_score(Y, P)
mcc = matthews_corrcoef(Y, P)
Precision_dev = precision_score(Y, P)
Reacll_dev = recall_score(Y, P)
Accuracy_dev = accuracy_score(Y, P)

print('Precision:', Precision_dev)
print('Reacll:', Reacll_dev)
print('Accuracy:', Accuracy_dev)
print('F1 score:', f1)
print('AUC:', AUC_dev, ROC_AUC)
print('PRC:', PRC_dev, PRC_AUC)
print('MCC:', mcc)
