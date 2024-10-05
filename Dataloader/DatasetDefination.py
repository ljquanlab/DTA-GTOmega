import pandas as pd
from rdkit import Chem
import itertools
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from Bio.PDB import PDBParser
import time
import os
from collections import defaultdict
from Dataloader.AtomEncode import *
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    # if x not in allowable_set:
    #     raise Exception("input {0} not in allowable set{1}:".format(
    #         x, allowable_set))
    return [x == s for s in allowable_set]


def atom_features(atom,
                  explicit_H=True,
                  use_chirality=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
         'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
         'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
         ]) + [atom.GetDegree() / 10, atom.GetImplicitValence(),
               atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other'
              ]) + [atom.GetIsAromatic()]
    
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results) # 56

def atomFeatures_From_dl4chem(a, ri_a):

    def _ringSize_a(a, rings):
        onehot = np.zeros(6)
        aid = a.GetIdx()
        for ring in rings:
            if aid in ring and len(ring) <= 8:
                onehot[len(ring) - 3] += 1

        return onehot
    
    def to_onehot(val, cat, etc=0):
        onehot=np.zeros(len(cat))
        for ci, c in enumerate(cat):
            if val == c:
                onehot[ci]=1

        if etc==1 and np.sum(onehot)==0:
            print(val)

        return onehot

    v1 = to_onehot(a.GetSymbol(), ['C','N','O','F','Cl','Br','I','S','B','Si','P','Te','Se','Ge','As'], 1)
    v2 = to_onehot(str(a.GetHybridization()), ['SP','SP2','SP3','SP3D','SP3D2'], 1)

    v3 = [a.GetAtomicNum(), a.GetDegree() / 10, a.GetFormalCharge(), a.GetTotalNumHs(), a.GetImplicitValence(), a.GetNumRadicalElectrons(), int(a.GetIsAromatic())]
    v4 = _ringSize_a(a, ri_a)

    v5 = np.zeros(3)
    try:
        tmp = to_onehot(a.GetProp('_CIPCode'), ['R','S'], 1)
        v5[0] = tmp[0]
        v5[1] = tmp[1]
    except:
        v5[2]=1

    v5 = v5[:2]

    return torch.from_numpy(np.concatenate([v1,v2,v3,v4,v5], axis=0))

def atom_feature_78(atom):
    features =  np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
                     + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7, 8, 9, 10])
                     + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10])
                     + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) 
                     + [atom.GetIsAromatic()]
                        )
    
    return torch.from_numpy(features)

def dgllite_atom_features(atom):
    '''
    ConcatFeaturizer(
        [atom_type_one_hot,
         atom_degree_one_hot,
         atom_implicit_valence_one_hot,
         atom_formal_charge,
         atom_num_radical_electrons,
         atom_hybridization_one_hot,
         atom_is_aromatic,
         atom_total_num_H_one_hot]
    )
    
    [atom_type_one_hot,
      atomic_number,
      atom_explicit_valence_one_hot,
      atom_total_num_H_one_hot,
      atom_hybridization_one_hot,
      atom_is_aromatic_one_hot,
      atom_is_in_ring_one_hot,
      atom_chirality_type_one_hot,
      atom_is_chiral_center]
              
    [partial(atom_type_one_hot, allowable_set=[
                    'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S',
                    'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'], encode_unknown=True),
                 partial(atom_degree_one_hot, allowable_set=list(range(6))),
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 partial(atom_hybridization_one_hot, encode_unknown=True),
                 atom_is_aromatic,
                 atom_total_num_H_one_hot,
                 atom_is_chiral_center,
                 atom_chirality_type_one_hot]
                 
    [partial(atom_type_one_hot,allowable_set=["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"]),
                                     atomic_number,
                                     atom_explicit_valence_one_hot,
                                     atom_total_num_H_one_hot,
                                     atom_hybridization_one_hot,
                                     atom_is_aromatic_one_hot,
                                     atom_is_in_ring_one_hot,
                                     atom_chirality_type_one_hot,
                                     atom_is_chiral_center]
         
    '''
    from functools import partial
    func_list = [atom_type_one_hot,
      atomic_number,
      atom_explicit_valence_one_hot,
      atom_total_num_H_one_hot,
      atom_hybridization_one_hot,
      atom_is_aromatic_one_hot,
      atom_is_in_ring_one_hot,
      atom_chirality_type_one_hot,
      atom_is_chiral_center]
    
    features = [func(atom) for func in func_list]
    features = np.concatenate(features)
    return torch.from_numpy(features)

def bond_features(bond, use_chirality=True):
    """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)

def get_mol_edge_list_and_feat_mtx(mol_graph, dgllife=True):
    try:

        ri = mol_graph.GetRingInfo()
        ri_a = ri.AtomRings()

        a_features = [(atom.GetIdx(), atom_feature_78(atom)) for atom in mol_graph.GetAtoms()]
        a_features.sort()  # to make sure that the feature matrix is aligned according to the idx of the atom
        _, a_features = zip(*a_features)
        a_features = torch.stack(a_features)

        edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
        undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
        
        e_features = torch.Tensor([bond_features(bond) for bond in mol_graph.GetBonds()]).float()
        e_features = torch.cat([e_features, e_features], 0)
        return undirected_edge_list.T, a_features, e_features
    except Exception as e:
        print(f"Error processing molecule: {e}")
        return None


def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_residue_centroids(structure):
    centroids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('CA'):  # 以α碳为代表，适用于标准氨基酸
                    centroid = residue['CA'].get_coord()
                    centroids.append(centroid)
    return np.array(centroids)

def calculate_edges_3d(centroids, threshold=5.0):
    edges_forward = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            if distance < threshold:
                edges_forward.append([i, j])  # 添加边 i 到 j

    # 将边复制并反转，以创建反向边
    edges_backward = [[j, i] for i, j in edges_forward]

    # 合并边并转换为NumPy数组
    edges = np.array(edges_forward + edges_backward).T
    return edges

# start_time = time.time()
# print('start_time:', start_time)

class ProteinFeatureManager():
    def __init__(self, data_path, pocket=True, threthod=0.5, omega_pdb=False):
        dataset = data_path.split('/')[-1]
        if 'Davis' in data_path:
            dataset = 'Davis'
        elif 'KIBA' in data_path:
            dataset = 'KIBA'
        elif 'BindingDB' in data_path:
            dataset = 'BindingDB'
        elif 'Case' in data_path:
            dataset = 'Case'
        elif 'DrugBank' in data_path:
            dataset = 'DrugBank'
        sequence_to_id = {}
        id_to_sequence = {}
        
        sequence_to_name = {}
        name_to_omega = {}
        name_to_contact = {}
        name_to_edges = {}
        name_to_edges_features = {}
        self.threthod = threthod
        name_to_omega_features = {}
        self.name_to_seq_features = {}
        self.dic = defaultdict(lambda:len(self.dic))
        # Repr
        if 'Davis' in data_path and 'Davis_logic_dataset' not in data_path:
            with open(os.path.join(data_path, 'seq_to_name.pkl'), 'rb') as f:
                sequence_to_name = pickle.load(f)
            with open(os.path.join(data_path, 'name_to_seq.pkl'), 'rb') as f:
                name_to_sequence = pickle.load(f)
            map_data = pd.read_csv(os.path.join(data_path, 'mapping.csv'))
            for seq, uniprot in zip(list(map_data['sequences']), list(map_data['uniprot'])):
                sequence_to_id[seq] = uniprot
                id_to_sequence[uniprot] = seq
                
        elif 'GalaxyDB' in data_path:
            map_data = pd.read_csv(os.path.join(data_path, 'mapping.csv'))
            for seq, uniprot, name in zip(list(map_data['sequences']),list(map_data['uniprot']),list(map_data['name'])):
                sequence_to_id[seq] = uniprot
                id_to_sequence[uniprot] = seq
                if seq in sequence_to_name.keys():
                    continue
                else:
                    sequence_to_name[seq] = name
                    
        elif 'Davis_logic_dataset' in data_path:
            with open(os.path.join(data_path, 'seq_to_name.pkl'), 'rb') as f:
                sequence_to_name = pickle.load(f)
            with open(os.path.join(data_path, 'name_to_seq.pkl'), 'rb') as f:
                name_to_sequence = pickle.load(f)
            map_data = pd.read_csv(os.path.join(data_path, 'mapping.csv'))
            for seq, uniprot in zip(list(map_data['sequences']), list(map_data['uniprot'])):
                sequence_to_id[seq] = uniprot
                id_to_sequence[uniprot] = seq
        
        elif 'KIBA' in data_path or 'BindingDB' in data_path or 'Case' in data_path or 'Test' in data_path or 'DrugBank' in data_path:
            with open(os.path.join(data_path, 'seq_to_name.pkl'), 'rb') as f:
                sequence_to_name = pickle.load(f)
            with open(os.path.join(data_path, 'name_to_seq.pkl'), 'rb') as f:
                name_to_sequence = pickle.load(f)

        else:
            assert False,'No such dataset'
            
        '''
        Process Omega and contact map
        '''
        # if 'Davis' in data_path:
        if True:
            if omega_pdb:
                files = os.listdir(f'data/{dataset}_omega/omega_2/')
            else:
                files = os.listdir(f'data/{dataset}_omega/omega/')
            if pocket:
                files = os.listdir(f'data/{dataset}_omega/pocket_omega/')
            files = [f for f in files if f[-3:] == 'pkl' and 'ipynb' not in f]
            for f in files:
                name = f[:-4]
                # max_len = False
                if name in name_to_omega:
                    continue
                if '_' in name:
                    name = name[:-2]
                    # CutOff to max_len: 800
                    f = name + '_0.pkl'
                    if omega_pdb:
                        with open(f'data/{dataset}_omega/omega_2/{f}','rb') as ff:
                            o = pickle.load(ff)
                    else:
                        with open(f'data/{dataset}_omega/omega/{f}','rb') as ff:
                            o = pickle.load(ff)
                    if pocket:
                        with open(f'data/{dataset}_omega/pocket_omega/{f}','rb') as ff:
                            o = pickle.load(ff)
                else:
                    if omega_pdb:
                        with open(f'data/{dataset}_omega/omega_2/{f}','rb') as ff:
                            o = pickle.load(ff)
                    else:
                        with open(f'data/{dataset}_omega/omega/{f}','rb') as ff:
                            o = pickle.load(ff)
                    if pocket:
                        with open(f'data/{dataset}_omega/pocket_omega/{f}','rb') as ff:
                            o = pickle.load(ff)
                if dataset == 'DrugBank' or dataset=='BindingDB':
                    # del edges for memory
                    o['struct_edge'] = 0.
                name_to_omega[name] = o
            if omega_pdb:
                print('Use Omega pdb contact')
                if pocket:
                    contact_map_files = os.listdir(f'data/{dataset}_omega/pocket_contact/')
                else:
                    contact_map_files = os.listdir(f'data/{dataset}_omega/pdb_contact_map/')
                contact_map_files = [f for f in contact_map_files if f[-3:] == 'npy']
                for f in contact_map_files:
                    name = f[:-4]
                    ff = f
                    if '_' in name:
                        name = name[:-2]
                        ff = name + '_0.npy'
                    if name in name_to_contact.keys():
                        continue
                    if pocket:
                        name_to_contact[name] = np.load(f'data/{dataset}_omega/pocket_contact/{ff}')
                    else:
                        name_to_contact[name] = np.load(f'data/{dataset}_omega/pdb_contact_map/{ff}')
                    if dataset != 'DrugBank' and dataset != 'BindingDB':
                        assert name_to_contact[name].shape[0] == name_to_contact[name].shape[1] == name_to_omega[name]['struct_node'].shape[0] == name_to_omega[name]['struct_edge'].shape[0] == name_to_omega[name]['struct_edge'].shape[1], (name, f'data/{dataset}_omega/pdb_contact_map/{f}', name_to_contact[name].shape,name_to_omega[name]['struct_node'].shape, name_to_omega[name]['struct_edge'].shape)
                    else:
                        assert name_to_contact[name].shape[0] == name_to_contact[name].shape[1] == name_to_omega[name]['struct_node'].shape[0], (name, f'data/{dataset}_omega/pdb_contact_map/{f}', name_to_contact[name].shape, name_to_omega[name]['struct_node'].shape)
            else:
                contact_map_files = os.listdir(data_path+'/protein_contact_map/')
                contact_map_files = [f for f in contact_map_files if f[-3:] == 'npy']
                for f in contact_map_files:
                    uni = f[:-4]
                    seq = id_to_sequence[uni]
                    name = sequence_to_name[seq]
                    name_to_contact[name] = np.load(data_path+f'/protein_contact_map/{uni}.npy')
                    size = min(name_to_omega[name]['struct_node'].shape[0], name_to_contact[name].shape[0])
                    name_to_contact[name] = name_to_contact[name][:size, :size]
                    name_to_omega[name]['struct_node'] = name_to_omega[name]['struct_node'][:size, :]
                    name_to_omega[name]['struct_edge'] = name_to_omega[name]['struct_edge'][:size, :size, :]
                    assert name_to_contact[name].shape[0] == name_to_contact[name].shape[1] == name_to_omega[name]['struct_edge'].shape[1]
        
        for name in name_to_contact.keys():
            # print(name)
            if name not in name_to_omega:continue
            omega = name_to_omega[name]
            contact = name_to_contact[name]

            contact_map_dig = (contact - np.eye(contact.shape[0]))
            # print(contact_map_dig.shape)
            protein_omega_features, protein_omega_edge_features = omega['struct_node'], omega['struct_edge']
            protein_omega_features = protein_omega_features[:contact_map_dig.shape[0]]
            protein_omega_edges = np.argwhere(contact_map_dig > self.threthod).T
            if dataset == 'DrugBank' or dataset == 'BindingDB':
                # del edge features for memory
                protein_omega_edge_features = 0.
            else:
                protein_omega_edge_features = protein_omega_edge_features[:contact_map_dig.shape[0]]
                protein_omega_edge_features = protein_omega_edge_features[:, :contact_map_dig.shape[0]]
            # assert protein_omega_features.shape[0] == contact_map_dig.shape[0]
                protein_omega_edge_features = protein_omega_edge_features[protein_omega_edges.T[:,0], protein_omega_edges.T[:,1]]
            name_to_edges[name] = protein_omega_edges
            name_to_edges_features[name] = protein_omega_edge_features
            name_to_omega_features[name] = protein_omega_features
            
            sequence = name_to_sequence[name]
            seq_features = np.array([self.dic[res] for res in sequence[:800]])
            self.name_to_seq_features[name] = seq_features
            assert len(seq_features) == protein_omega_features.shape[0]
            
        self.name_to_contact = name_to_contact
        self.name_to_omega = name_to_omega
        self.sequence_to_id = sequence_to_id
        self.sequence_to_name = sequence_to_name
        self.name_to_sequence = name_to_sequence
        # with open(os.path.join(data_path, 'protein_embedding/bert_embedding_Nongram.pkl'), 'rb') as f:
        #     self.bert_embed_dict = pickle.load(f)
        self.data_path = data_path
        self.pocket = pocket
        
        self.name_to_edges = name_to_edges
        self.name_to_edges_features = name_to_edges_features
        self.name_to_omega_features = name_to_omega_features
        
        # self.name_to_seq_features = seq_features

    def get_node_features(self, sequence):
        return np.load(os.path.join(self.data_path, 'protein_node_features', self.sequence_to_id[sequence] + '.npy'))

    def get_contact_map(self, sequence):
        # contact_map = np.load(os.path.join(self.data_path, 'protein_contact_map', self.sequence_to_id[sequence] + '.npy'))
        # assert len(sequence) == contact_map.shape[0] == contact_map.shape[1]
        if self.pocket:
            name = self.sequence_to_name[sequence]
        else:
            name = self.sequence_to_id[sequence]
        
        return self.name_to_contact[name]

    def get_pdb_map(self, sequence):
        return np.load(os.path.join(self.data_path, 'new_npy_files_old_new', self.sequence_to_id[sequence] + '.npy'))

    def get_omega_feature(self, sequence):
        name = self.sequence_to_name[sequence]
        # omega_feature = self.name_to_omega[name]
        # exceed_len = 800
        # slices = len(sequence) // 512 + 1
        # struct_nodes = []
        # struct_edges = []
        # if len(sequence) >= exceed_len:
        #     # for sli in range(slices):
        #     # 截断
        #     for sli in range(1):
        #         if 'Davis' in self.data_path:
        #             path = 'data/Davis_omega/omega/' + name + f'_{sli}.pdb_.pkl'
        #         with open(path, 'rb') as f:
        #             omega_feature = pickle.load(f)
        #         struct_node = omega_feature['struct_node']
        #         struct_edge = omega_feature['struct_edge']
        #         if struct_nodes == []:
        #             struct_nodes = struct_node
        #             struct_edges = struct_edge
        #         else:
        #             struct_nodes = np.concatenate([struct_nodes, struct_node], 0)
        #             new_pairs = np.zeros((struct_nodes.shape[0], struct_nodes.shape[0], 128))
        #             # last slice
        #             new_pairs[:struct_edges.shape[0], :struct_edges.shape[1], :] = struct_edges
        #             # new slice
        #             new_pairs[struct_edges.shape[0]:, struct_edges.shape[1]:, :] = struct_edge
        #             struct_edges = new_pairs
        # else:
        #     if 'Davis' in self.data_path:
        #         cut_path = self.data_path.split('_')[0]
        #         path = f'{cut_path}_omega/omega/' + name + '.pdb_.pkl'
        #     elif 'GalaxyDB' in self.data_path:
        #         path = f'data/{self.data_path}_omega/omega/' + name + '.pkl'
        #     else:
        #         assert False,'No such dataset'
        #     with open(path, 'rb') as f:
        #         omega_feature = pickle.load(f)
        
        # struct_nodes = omega_feature['struct_node']
        # struct_edges = omega_feature['struct_edge']
        # print(name)
        struct_nodes = self.name_to_omega_features[name]
        edges = self.name_to_edges[name]
        edges_features = self.name_to_edges_features[name]
        
        seq_features = self.name_to_seq_features[name]
        if type(edges_features) != float:
            assert edges.shape[1] == edges_features.shape[0]

        return struct_nodes, edges, edges_features, seq_features.astype(int)
    
    def get_pretrained_embedding(self, sequence):
        bert_embed = self.bert_embed_dict[sequence]
        return bert_embed

def get_bipartite_graph(num_drug_atoms, num_protein_residues):
    # 创建二部图的边列表，不需要添加偏移量
    x1 = np.arange(0, num_drug_atoms)
    x2 = np.arange(0, num_protein_residues)
    meshgrid = np.array(np.meshgrid(x1, x2))
    edge_list = torch.LongTensor(meshgrid)
    edge_list = torch.stack([edge_list[0].reshape(-1), edge_list[1].reshape(-1)], dim=0)

    return edge_list

class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, num_nodes=0):
        super().__init__(num_nodes=num_nodes)
        # edge_index就是他们原子之间连接后的组合[4, 5, 4, 5, 4, 5, 4, 5],[0, 0, 1, 1, 2, 2, 3, 3]，（表示4,5节点和0,1,2,3节点的组合）
        # x_s, x_t分别指的是他们的特征[28, 55]， [22, 55] 如果按照(4,5)节点和(0,1,2,3)节点的组合来看，那么就是[2, 55], [4, 55]
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

class DrugDataset(Dataset):
    """药物-蛋白质互作的图表示数据集"""

    def __init__(self, compound_smiles, protein_sequences, labels,
                 mol_edge_list_feat_mtx, protein_feature_manager, threshold, stand_label=False):
        """
        初始化数据集。
        :param compound_smiles: 化合物的SMILES表示列表
        :param protein_sequences: 蛋白质序列的列表
        :param labels: 标签列表
        :param mol_edge_list_feat_mtx: 化合物的边和节点特征矩阵字典
        :param protein_feature_manager: 管理蛋白质特征的对象
        :param threshold: 决定蛋白质接触图边的阈值
        """
        self.compound_smiles = compound_smiles
        self.protein_sequences = protein_sequences
        self.labels = labels
        self.mol_edge_list_feat_mtx = mol_edge_list_feat_mtx
        self.protein_feature_manager = protein_feature_manager
        self.threshold = threshold
        if stand_label:
            self.labels = np.array([float(l) for l in self.labels])
            self.mean_ = np.mean(self.labels)
            self.std = np.std(self.labels)
            self.labels = (self.labels - self.mean_) / self.std
            

    def __len__(self):
        """返回数据集中的样本总数"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        返回索引idx处的样本和标签。
        """
        compound_smile = self.compound_smiles[idx]
        protein_sequence = self.protein_sequences[idx]
        label = self.labels[idx]

        # 获取药物的边和节点特征
        drug_edges, drug_nodes, drug_edges_features = self.mol_edge_list_feat_mtx[compound_smile]
        assert drug_edges.shape[0] == 2
        assert drug_edges.shape[1] == drug_edges_features.shape[0]
        # 蛋白质特征
        # node_features = self.protein_feature_manager.get_node_features(protein_sequence)
        # contact_map = self.protein_feature_manager.get_contact_map(protein_sequence)
        # contact_map_dig = (contact_map - np.eye(contact_map.shape[0]))
        # Protein Omega Features
        protein_omega_features, protein_omega_edges, protein_omega_edge_features, seq_features = self.protein_feature_manager.get_omega_feature(protein_sequence)
        
        # contact_map_dig = contact_map_dig[:protein_omega_features.shape[0], :protein_omega_features.shape[0]]
        # protein_omega_edges = np.argwhere(contact_map_dig > self.threshold).T
        # assert protein_omega_features.shape[0] == contact_map_dig.shape[0]
        # protein_omega_edge_features = protein_omega_edge[protein_omega_edges.T[:,0], protein_omega_edges.T[:,1]]
        
        
        # print(protein_omega_edge_features.shape, protein_omega_edge.shape)
        if type(protein_omega_edge_features) != float:
            assert protein_omega_edge_features.shape[0] == protein_omega_edges.shape[1]
        # 确定蛋白质的边
#         edges = []
#         for i in range(contact_map.shape[0]):
#             for j in range(i + 1, contact_map.shape[1]):
#                 if contact_map[i, j] >= self.threshold:
#                     edges.append([i, j])

#         # 将边列表转换为张量，并添加双向边
#         edges_tensor = torch.tensor(edges).T
#         undirected_edges_tensor = torch.cat([edges_tensor, edges_tensor[:, [1, 0]]], dim=1) if len(edges) else edges_tensor

        # pdb_graph_edge = self.protein_feature_manager.get_pdb_map(protein_sequence)

        # 构建样本字典
        sample = {
            # 'compound_smile': compound_smile,
            'drug_edges': drug_edges,
            'drug_nodes': drug_nodes,
            'drug_edges_features':drug_edges_features,
            # 'protein_sequence': protein_sequence,
            # 'node_features': torch.tensor(node_features, dtype=torch.float),
            # 'pdb_graph_edge': torch.tensor(protein_omega_edges, dtype=torch.float),
            'protein_omega_features': torch.tensor(protein_omega_features, dtype=torch.float),
            'protein_omega_edge': torch.tensor(protein_omega_edges, dtype=torch.float),
            'protein_omega_edge_features': torch.tensor(protein_omega_edge_features, dtype=torch.float),
            'protein_sequence': torch.LongTensor(seq_features),
            # 'edges': undirected_edges_tensor,
            'label': torch.tensor(label, dtype=torch.float)
        }

        return sample


def collate_fn(batch):
        drug_graphs = []
        protein_graphs = []
        bipartite_graphs = []  # 用于存储二部图
        labels_batch = []

        for sample in batch:
            # 药物和蛋白质图
            
            drug_graph = Data(x=sample['drug_nodes'].float(), 
                              edge_index=sample['drug_edges'].long(), 
                              edge_attr=sample['drug_edges_features'].float())
            # protein_graph = Data(x=sample['node_features'].float(), edge_index=sample['pdb_graph_edge'].long())
            # x = torch.cat([sample['protein_omega_features'], sample['protein_sequence'].float()[:,None]], 1)
            protein_graph = Data(x = sample['protein_omega_features'], 
                                 edge_index = sample['protein_omega_edge'].long(), 
                                 edge_attr = sample['protein_omega_edge_features'].float())
            
            drug_graphs.append(drug_graph)
            protein_graphs.append(protein_graph)

#             drug_features = sample['drug_nodes'].float()
#             protein_features = sample['protein_omega_features'].float()

#             # 使用修改后的函数创建二部图的边
#             num_drug_atoms = drug_features.size(0)
#             num_protein_residues = protein_features.size(0)

            # 创建二部图数据实例
            # num_nodes = drug_features.shape[0] + protein_features.shape[0]
            # bipartite_edge_index = get_bipartite_graph(num_drug_atoms, num_protein_residues)
            # bipartite_data = BipartiteData(edge_index=bipartite_edge_index.long(), x_s=drug_features.float(), x_t=protein_features.float(), num_nodes = num_nodes)
            # bipartite_graphs.append(bipartite_data)

            # 标签
            labels_batch.append(sample['label'])

        # 将图数据汇总成批次
        drug_batch = Batch.from_data_list(drug_graphs)
        protein_batch = Batch.from_data_list(protein_graphs)
        # bipartite_batch = Batch.from_data_list(bipartite_graphs)  # 汇总二部图

        # 创建包含所有批处理数据的字典
        batch_data = {
            'drug_graphs': drug_batch,
            'protein_graphs': protein_batch,
            # 'bipartite_graphs': bipartite_batch,  # 添加二部图数据
            'labels': torch.stack(labels_batch)
        }
        return batch_data