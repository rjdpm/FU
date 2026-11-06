#!/usr/bin/env python3
import pickle, inspect

import torch
import torch.nn as nn
from torchinfo import summary

import sys, os, copy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_module_fu import make_mlp, RGCNConv, normalize_adjacency, activation_func 
from data_utils_fu import auto_repr

__all__ = ['CG_GML']

source_file = os.path.abspath(__file__)
with open(source_file, 'rb') as fp:
    file_ = fp.read()
    network_pyfile = pickle.dumps(file_)
    
class ProportionTo(nn.Module):
    """
    Computes:
        output = exp(scale_head(x)) * y_head(x)

    Returns:
        y      : base prediction
        scaled : proportional prediction
    """

    def __init__(self, mlp_args, add_bias=False):
        super().__init__()

        mlp_args = copy.deepcopy(mlp_args)
        mlp_args['list_dims'][-1] = 3
        self.add_bias = add_bias
        self.y_head = make_mlp(**mlp_args)

    def forward(self, x):

        y_all = self.y_head(x)
        y = y_all[:, 0]
        scale = torch.exp(y_all[:, 1])
        bias = y_all[:, 2]

        prop_pred = scale * y + bias

        return y, prop_pred

class MRGCN(nn.Module):

    def __init__(self,
                 list_dims_gcn,
                 list_dims_fc,
                 max_num_atom,
                 dropout=0.2,
                 num_relations=4,
                 act_func='relu',
                 act_func_gcn='relu',
                 norm_type='layer',
                 alpha=1.0,
                 negative_slope=1e-2,
                 use_attention=False,
                 attn_type='self attention',
                 attn_dim=None,
                 num_heads=1,
                 use_gin_readout = False,
                 ):
        """
        Multi-Relational Graph Convolutional Network with optional attention-based pooling and dense prediction head.

        Args:
            list_dims_gcn (list[int]): Dimensions for GCN layers.
            list_dims_fc (list[int]): Dimensions for fully connected layers.
            dropout (float): Dropout rate.
            max_num_atom (int): Max number of atoms (used for pooling logic externally).
            num_relations (int): Number of edge types in the graph.
            act_func (str): Activation function for FC layers.
            act_func_gcn (str): Activation function for GCN layers.
            norm_type (str): Type of normalization ('layer', 'batch', etc.).
            alpha (float): Parameter for certain activation functions (e.g., ELU).
            negative_slope (float): Slope for LeakyReLU.
            use_attention (bool): Whether to use attention-based pooling.
            attn_type (str): Type of attention pooling ('self attention' or 'attention pool').
            attn_dim (int): Attention dimension.
            num_heads (int): Number of attention heads (for multi-head attention).
            use_gin_readout (bool): To specify whether to use GIN-like readout representation or not.

        Outputs:
            - Tensor: Final prediction or representation [batch_size, output_dim]
        """
        super().__init__()

        # ----------- Store configs -----------
        self.max_num_atom = max_num_atom
        self.list_dims_gcn = list_dims_gcn
        self.list_dims_fc = list_dims_fc
        self.num_relations = num_relations
        self.act_func = act_func
        self.act_func_gcn = act_func_gcn
        self.norm_type = norm_type
        self.alpha = alpha
        self.negative_slope = negative_slope
        self.use_gin_readout = use_gin_readout

        self.use_attention = use_attention
        self.attn_type = attn_type
        self.attn_dim = attn_dim
        self.num_heads = num_heads

        self.class_def = inspect.getsource(self.__class__)
        self.network_pyfile = network_pyfile

        # ----------- Build modules -----------
        self._build_gcn_layers()
        self._build_attention_layers()
        self._build_readout_layers()
        self._build_fc_layers(dropout)

    def __repr__(self):
        return auto_repr(self)

    # BUILDING BLOCKS
    def _build_gcn_layers(self):
        
        """Construct GCN layers, norms, and residual projections."""
        
        self.gcn_layers = nn.ModuleList()
        self.gcn_layer_norms = nn.ModuleList()

        for i in range(len(self.list_dims_gcn) - 1):
            in_dim = self.list_dims_gcn[i]
            out_dim = self.list_dims_gcn[i + 1]

            self.gcn_layers.append(RGCNConv(in_dim, out_dim, num_relations=self.num_relations))
            if i < len(self.list_dims_gcn) - 2:
                self.gcn_layer_norms.append(nn.LayerNorm(out_dim))
        self.activation_func_gcn = activation_func(self.act_func_gcn, alpha=self.alpha, negative_slope=self.negative_slope)

    def _build_attention_layers(self):
        
        """Construct attention pooling layers if enabled."""
        readout_dim = self.list_dims_gcn[-1]
        attn_dim = self.attn_dim or readout_dim

        if not self.use_attention:
            return

        if self.attn_type == 'self attention':
            self.W_Q = nn.Linear(readout_dim, attn_dim * self.num_heads)
            self.W_K = nn.Linear(readout_dim, attn_dim * self.num_heads)
            self.W_V = nn.Linear(readout_dim, attn_dim * self.num_heads)
            self.W_out = nn.Linear(attn_dim * self.num_heads, readout_dim)

        elif self.attn_type == 'attention pool':
            self.att_pool = nn.Sequential(
                nn.Linear(readout_dim, 2*readout_dim // 3),
                nn.Tanh(),
                nn.Linear(2*readout_dim // 3, readout_dim // 3),
                nn.Tanh(),
                nn.Linear(readout_dim // 3, 1)
            )

            # self.att_pool = nn.Sequential(
            #     nn.Linear(readout_dim, 3*readout_dim // 4),
            #     nn.Tanh(),
            #     nn.Linear(3*readout_dim // 4, 2*readout_dim // 4),
            #     nn.Tanh(),
            #     nn.Linear(2*readout_dim // 4, readout_dim // 4),
            #     nn.Tanh(),
            #     nn.Linear(readout_dim // 4, 1)
            # )
            
    def _build_readout_layers(self):
        
        if not self.use_gin_readout:
            return
        else:
            readout_dim = sum(self.list_dims_gcn[1:])
            step = int((readout_dim - self.list_dims_gcn[-1])/2)
            self.readout_fn = nn.Sequential(
                nn.Linear(readout_dim, readout_dim-step),
                nn.Tanh(),
                nn.Linear(readout_dim-step, self.list_dims_gcn[-1])
            )

    def _build_fc_layers(self, dropout):
        
        """Construct fully connected head."""
        self.dropout = nn.Dropout(dropout)

        if len(self.list_dims_fc) > 0:
            self.fc_conn = nn.Linear(self.list_dims_gcn[-1], self.list_dims_fc[0])
            self.input2repr = make_mlp(
                list_dims=self.list_dims_fc,
                dropout=dropout,
                act_func=self.act_func,
                norm_type=self.norm_type
            )
        else:
            self.fc_conn = nn.Identity()
            self.input2repr = nn.Identity()

    # CORE PIPELINE
    def _apply_masks(self, feature_vector, adjacency_tensor, degree_tensor,
                    feature_mask, adjacency_mask):
        """Apply feature and adjacency masks."""
        
        feature_vector = feature_vector * feature_mask
        adjacency_tensor = adjacency_tensor * adjacency_mask
        degree_tensor = degree_tensor * adjacency_mask
        
        return feature_vector, adjacency_tensor, degree_tensor

    def _normalize_adj(self, adjacency_tensor, degree_tensor):
        """Normalize adjacency matrix."""
        
        return normalize_adjacency(adjacency=adjacency_tensor,
                                   degree=degree_tensor)

    def _gcn_forward(self, x, adjacency_tensor):
        """Forward pass through stacked GCN layers."""
        
        self.layer_outputs = []
        for i, gcn_layer in enumerate(self.gcn_layers):

            x = gcn_layer(x, adjacency_tensor)

            if i < len(self.gcn_layers) - 1:
                x = self.activation_func_gcn(x)
                x = self.gcn_layer_norms[i](x)
                x = self.dropout(x)
            self.layer_outputs.append(x)
            
        return x

    # POOLING METHODS
    def _self_attention_pool(self, x):
        
        """Multi-head self-attention pooling."""
        
        B, N, D = x.shape
        H = self.num_heads
        d_k = D // H

        Q = self.W_Q(x).view(B, N, H, -1).transpose(1, 2)
        K = self.W_K(x).view(B, N, H, -1).transpose(1, 2)
        V = self.W_V(x).view(B, N, H, -1).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        x = torch.matmul(attn_weights, V)
        x = x.mean(dim=2)

        x = x.transpose(1, 2).contiguous().view(B, -1)
        x = self.W_out(x)

        return x

    def _attention_pool(self, x):
        
        """Simple attention pooling."""
        att_weights = torch.softmax(self.att_pool(x), dim=1)
        
        return torch.sum(att_weights * x, dim=1)

    def _mean_pool(self, x):
        
        """Mean pooling."""
        
        return x.mean(dim=1)
    
    def _gin_pool(self, layer_outputs):
        """
        GIN-style readout (canonical):
        Concatenate SUM-pooled representations from all layers.
        
        Input:
            layer_outputs: list of tensors [B, N, D]
        Output:
            Tensor [B, D * num_layers]
        """
        pooled_list = []

        for x in layer_outputs:
            pooled = x.sum(dim=1)  # [B, D]
            pooled_list.append(pooled)

        x = torch.cat(pooled_list, dim=-1)  # [B, D * num_layers]
        x = self.readout_fn(x)

        return x

    def _pool(self, x):
        
        """Unified pooling interface."""
        if self.use_gin_readout:
            return self._gin_pool(self.layer_outputs)
        if self.use_attention:
            if self.attn_type == 'self attention':
                return self._self_attention_pool(x)
            elif self.attn_type == 'attention pool':
                return self._attention_pool(x)

        return self._mean_pool(x)

    # PUBLIC METHODS
    def get_features(self, feature_vector, adjacency_tensor, degree_tensor,
                     feature_mask=1.0, adjacency_mask=1.0):
        """
        Extract graph-level features.
        Task:
            Extract graph-level features from multi-relational input.

        Inputs:
            - x (Tensor): Node features
            - adjacency_tensor (Tensor): Edge-type-specific adjacency matrices
            - degree_tensor (Tensor): Node degree matrices

        Outputs:
            - Tensor: Aggregated graph features [batch_size, gcn_out_dim]
        """

        feature_vector, adjacency_tensor, degree_tensor = self._apply_masks(feature_vector, adjacency_tensor, degree_tensor,
                                                                            feature_mask, adjacency_mask)
        adjacency_tensor = self._normalize_adj(adjacency_tensor, degree_tensor)
        x = self._gcn_forward(feature_vector, adjacency_tensor)
        x = self._pool(x)

        self.feature_dim = x.shape[-1]
        return x

    def forward(self, feature_vector, adjacency_tensor, degree_tensor,
                feature_mask=1.0, adjacency_mask=1.0):
        """
        Full forward pass.
        Task:
            Forward pass from graph input to final prediction via GCN and FC layers.

        Inputs:
            - x: Node features
            - adjacency_tensor: Multi-relational adjacency
            - degree_tensor: Node degree tensor
            - feature_mask (float or Tensor): Mask to apply on features
            - adjacency_mask (float or Tensor): Mask to apply on adjacency

        Outputs:layer_outputs
            - Tensor: Final prediction/representation
        """

        x = self.get_features(feature_vector, adjacency_tensor, degree_tensor,
                              feature_mask, adjacency_mask)

        x = self.fc_conn(x)
        x = self.input2repr(x)

        return x
    
    
    
class CG_GML(nn.Module):

    def __init__(self,
                 gcn_args,
                 mlp_args,
                 use_sigmoid=False,
                 add_bias=False,
                 desc_list_ka=['TPSA', 'MolWt', 'NumAromaticRings'],
                 desc_list_fu=['MolLogP', 'NumHDonors', 'NumHAcceptors'],
                 n_coeffs=3,
                 ):
        super().__init__()

        self.gcn_args = gcn_args
        self.add_bias = add_bias
        self.mlp_args = mlp_args
        self.n_coeffs = n_coeffs
        self.use_sigmoid = use_sigmoid
        self.desc_list_fu = desc_list_fu
        self.desc_list_ka = desc_list_ka

        self.loss_weights = nn.Parameter(torch.ones(n_coeffs), requires_grad=True)

        # GNN encoders
        self.fu_gnn = MRGCN(**gcn_args)

        # Primary heads
        self.fu_mlp = make_mlp(**mlp_args)
        self.ka_mlp = make_mlp(**mlp_args)

        # Ka dependent properties
        for desc in desc_list_ka:
            setattr(self, f'{desc}_head', ProportionTo(mlp_args, add_bias=add_bias))

        # Fu dependent properties
        for desc in desc_list_fu:
            setattr(self, f'{desc}_head', ProportionTo(mlp_args, add_bias=add_bias))

    def __repr__(self):
        return auto_repr(self)

    def get_features(self, data):

        fu_rep = self.fu_gnn(**data)
        self.feature_dim = fu_rep.shape[-1]

        return fu_rep

    def latent2Fu(self, latent):
        return self.fu_mlp(latent)

    def pred_fu(self, data):
        return self.latent2Fu(self.get_features(data))

    def forward(self, data):

        fu_rep = self.get_features(data)

        # Primary predictions
        Fu = self.fu_mlp(fu_rep)
        if self.use_sigmoid:
            Fu = torch.sigmoid(Fu)
        logKa = self.ka_mlp(fu_rep)

        # Ka properties
        for desc in self.desc_list_ka:
            head = getattr(self, f'{desc}_head')
            desc_val, Ka_desc = head(fu_rep)
            setattr(self, f'Ka_{desc}', Ka_desc)
            setattr(self, f'{desc}', desc_val)

        # Fu properties        
        for desc in self.desc_list_fu:
            head = getattr(self, f'{desc}_head')
            desc_val, Fu_desc = head(fu_rep)
            if self.use_sigmoid:
                Fu_desc = torch.sigmoid(Fu_desc)
            setattr(self, f'Fu_{desc}', Fu_desc)
            setattr(self, f'{desc}', desc_val)
        
        results = {
            "logKa": logKa,
            "Fu": Fu,
            
            **{f"Ka_{desc}": getattr(self, f'Ka_{desc}') for desc in self.desc_list_ka},
            **{desc: getattr(self, desc) for desc in self.desc_list_ka},
            **{f"Fu_{desc}": getattr(self, f'Fu_{desc}') for desc in self.desc_list_fu},
            **{desc: getattr(self, desc) for desc in self.desc_list_fu},
        }

        return results
    
    
if __name__ == "__main__":
    gcn_args = {
        'list_dims_gcn': [17, 64, 256, 512],
        'list_dims_fc': [512, 256, 120],
        'dropout': 0,
        'max_num_atom': 65,
        'act_func': 'relu',
        'act_func_gcn': 'relu',
        'norm_type': 'layer',
    }
    
    mlp_args = {
        'list_dims': [120, 256, 256, 1],
        'dropout': 0,
        'act_func': 'relu',
        'norm_type': 'layer',
        'alpha': 1,
        'negative_slope': 0.01
    }
    
    model = CG_GML(gcn_args, mlp_args)
    data = {'feature_vector': torch.ones(10, 65, 17),
            'adjacency_tensor': torch.ones(10, 4, 65, 65),
            'degree_tensor': torch.ones(10, 4, 65, 65)}
    
    summary(model,
            input_data=(data,),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=5,
            )
    print(model(data))
        