from torch import nn
from torch_geometric.nn import (GATConv,
                                GATv2Conv,
                                SAGPooling,
                                LayerNorm,
                                global_mean_pool,
                                max_pool_neighbor_x,
                                global_add_pool)
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.norm import GraphNorm
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
import torch
import numpy as np
import math

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim, out_dim=32, n_heads=2, edge_dim=None):
        super().__init__()
        self.input_dim = input_dim
        if edge_dim is not None:
            self.intra = TransformerConv(input_dim, out_dim, n_heads)
        else:
            self.intra = TransformerConv(input_dim, out_dim, n_heads)

    def forward(self, data, edge_dim=None):
        input_feature, edge_index = data.x, data.edge_index
        input_feature = F.elu(input_feature)
        intra_rep = self.intra(input_feature, edge_index)
        return intra_rep

class InterGraphAttention(nn.Module):
    def __init__(self, input_dim, out_dim=32, n_heads=2):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim, input_dim), 32, 2)

    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)

        t_rep = self.inter((h_input, t_input), edge_index)
        h_rep = self.inter((t_input, h_input), edge_index[[1, 0]])
        return h_rep, t_rep

class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores

        return attentions


class RESCAL(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
    def forward(self, heads, tails, alpha_scores):
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        # bacth_size, 4, 4
        scores = heads @ tails.transpose(-2, -1)
        scores2 = tails @ heads.transpose(-2, -1)
        
        if alpha_scores is not None:
            scores = alpha_scores * scores
            scores2 = alpha_scores * scores2
        return scores, scores2

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features_drug, in_features_protein, head_out_feats, edge_dim=None):
        super().__init__()
        self.n_heads = n_heads
        self.in_features_drug = in_features_drug
        self.in_features_protein = in_features_protein
        self.out_features = head_out_feats
        self.edge_dim = edge_dim

        if edge_dim is not None:
            # print('Bias')
            self.feature_conv_protein = TransformerConv(in_features_protein, head_out_feats, n_heads)
            self.feature_conv_drug = TransformerConv(in_features_drug, head_out_feats, n_heads, edge_dim=10)
        else:
            self.feature_conv_protein = TransformerConv(in_features_protein, head_out_feats, n_heads)
            self.feature_conv_drug = TransformerConv(in_features_drug, head_out_feats, n_heads)
        
        self.intraAtt = IntraGraphAttention(head_out_feats * n_heads, out_dim=head_out_feats, n_heads=n_heads)
        self.interAtt = InterGraphAttention(head_out_feats * n_heads, out_dim=head_out_feats, n_heads=n_heads)
        self.readout_1 = SAGPooling(head_out_feats*n_heads, min_score=-1)
        self.readout_2 = SAGPooling(head_out_feats*n_heads, min_score=-1)

    def forward(self, h_data, t_data, b_graph=None):
        if self.edge_dim is not None:
            t_data.x = self.feature_conv_protein(t_data.x, t_data.edge_index)
            h_data.x = self.feature_conv_drug(h_data.x, h_data.edge_index, h_data.edge_attr)
        else:
            t_data.x = self.feature_conv_protein(t_data.x, t_data.edge_index)
            h_data.x = self.feature_conv_drug(h_data.x, h_data.edge_index)
        # b, head_out_feats * n_heads
        
        # b, 64 * 1
        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)
        
        # if b_graph != None:
        #     h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)
        #     h_rep = torch.cat([h_intraRep, h_interRep], 1)
        #     t_rep = torch.cat([t_intraRep, t_interRep], 1)
        # else:
        #     h_rep = torch.cat([h_intraRep, h_intraRep], 1)
        #     t_rep = torch.cat([t_intraRep, t_intraRep], 1)
        
        # b, 128
        h_data.x = h_intraRep
        t_data.x = t_intraRep

        # readout
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores = self.readout_1(h_data.x, h_data.edge_index, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores = self.readout_2(t_data.x, t_data.edge_index, batch=t_data.batch)
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb

class StructDTI(nn.Module):
    def __init__(self, in_features_drug, in_features_protein, hidd_dim, kge_dim, heads_out_feat_params, blocks_params, edge_dim=None):
        super(StructDTI, self).__init__()

        self.in_features_drug = in_features_drug
        self.in_features_protein = in_features_protein
        self.hidd_dim = hidd_dim
        self.n_blocks = len(blocks_params)
        self.kge_dim = kge_dim

        self.initial_norm_drug = LayerNorm(self.in_features_drug)
        self.initial_norm_protein = LayerNorm(self.in_features_protein)

        self.blocks = []
        self.net_norms = ModuleList()
        self.batchnorm = ModuleList()
        self.edge_dim = edge_dim

        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(n_heads, in_features_drug, in_features_protein, head_out_feats, edge_dim=edge_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            # self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            self.net_norms.append(GraphNorm(head_out_feats * n_heads))
            in_features_drug = head_out_feats * n_heads
            in_features_protein = head_out_feats * n_heads

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.kge_dim)
        self.out = nn.Sequential(
                    nn.ReLU(),
                    # nn.Dropout(0.2),
                    nn.Linear(head_out_feats * n_heads * 2, 1)
        )

    def forward(self, drug_graphs, protein_graphs, bi_graphs, task='repr'):
        h_data, t_data, b_graph = drug_graphs, protein_graphs, bi_graphs
        drug_init = self.initial_norm_drug(h_data.x)
        h_data.x = drug_init
        omega_init = self.initial_norm_protein(t_data.x)
        t_data.x = omega_init

        repr_h = []
        repr_t = []

        # Separate processing for drugs and proteins
        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        kge_drug = repr_h
        kge_target = repr_t

        attentions = self.co_attention(kge_drug, kge_target)
        
        # batch_size, 4, 4
        scores, scores2 = self.KGE(kge_drug, kge_target, attentions)
        
        # kge_target: batch_size, 4, head_out_feats * n_heads
        # -> batch_size, head_out_feats * n_heads -> batch_size, 1
        out_features = torch.cat((scores @ kge_target, scores2 @ kge_drug), -1)

        out_features = out_features.sum(dim=-2)
        out_features = self.out(out_features)
        # print(out_features.shape)
        if task == 'repr':
            out = out_features.sum(-1)
        else:
            out = out_features
        return out