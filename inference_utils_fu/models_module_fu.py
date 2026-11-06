#!/usr/bin/env python3
import inspect
import os, pickle, sys
# from typing import List, Tuple, Optional, Dict, Set

import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils_fu import auto_repr

__all__ = [
    'network_pyfile',
    'activation_func',
    'SMILESLinearNet',
    'make_mlp',
    'normalize_adjacency',
    'RGCNConv',
]

source_file = os.path.abspath(__file__)
with open(source_file, 'rb') as fp:
    file_ = fp.read()
    network_pyfile = pickle.dumps(file_)
    
def activation_func(name='relu', alpha=1.0, negative_slope=1e-2):
    """
    Return a PyTorch activation module based on the provided name.

    Args:
        name (str): Name of the activation function.
        alpha (float): Parameter for ELU/CELU.
        negative_slope (float): Parameter for LeakyReLU.

    Returns:
        nn.Module: Corresponding activation function.
    """
    name = name.lower()
    if name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'selu':
        return nn.SELU()
    elif name == 'elu':
        return nn.ELU(alpha)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope)
    elif name == 'celu':
        return nn.CELU(alpha)
    elif name == 'gelu':
        return nn.GELU()
    elif name is None:
        return nn.Identity()
    else:
        raise ValueError(f'Unsupported activation function: "{name}"')


class SMILESLinearNet(nn.Module):
    """
    Task:
        Feedforward network for regression or classification on SMILES-based vector inputs.

    Inputs:
        - list_dims (List[int]): List of dimensions for each linear layer.
        - dropout (float): Dropout probability applied after each layer except the last.

    Outputs:
        - Tensor of shape [batch_size, output_dim] after passing through linear layers.
    """
    def __init__(self, list_dims, dropout=0.2, act_func='relu', norm_type='layer'):
        super().__init__()
        
        self.class_def = inspect.getsource(self.__class__)
        self.list_dims = list_dims
        self.dropout = dropout
        self.act_func = act_func
        self.norm_type = norm_type
        self.network_pyfile = network_pyfile
        self.MLP = make_mlp(list_dims=list_dims,
                            dropout=dropout,
                            act_func=act_func,
                            norm_type=norm_type
                            )
        
        # Weight Initialization
        for layer in self.MLP:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def __repr__(self):
        return auto_repr(self)
    
    def forward(self, x):
        """
        Task:
            Forward pass through MLP with ReLU, BatchNorm, and Dropout.

        Inputs:
            - x (Tensor): Input tensor of shape [batch_size, list_dims[0]]

        Outputs:
            - Tensor: Output tensor after final linear layer.
        """
        return self.MLP(x)

def make_mlp(list_dims, dropout=0.0, act_func='relu', norm_type='layer', alpha=1.0, negative_slope=1e-2):
    
    """
    Constructs a multi-layer perceptron (MLP) using a sequence of linear layers, each optionally followed by 
    normalization, activation, and dropout layers.

    Args:
        list_dims (List[int]): A list specifying the input, hidden, and output dimensions. 
                               For example, [128, 256, 64, 1] creates 3 layers.
        dropout (float): Dropout rate to apply after each activation (except the final layer). Default is 0.0.
        act_func (Union[str, List[str], None]): Activation function(s) to use. 
            - If a single string (e.g., 'relu', 'elu', 'leakyrelu'), it is used for all hidden layers. 
              If list_dims=[128,256,64,1], then act_func list must have length 2 (for the two hidden layers [256,64]).
            - If a list of strings, it must have `len(list_dims) - 2` entries (for hidden layers only).
            - If None, no activation is applied.
        norm_type (str): Type of normalization to apply after each linear layer:
            - 'batch' for BatchNorm1d
            - 'layer' for LayerNorm
            - 'NA' for No Normalization
            - None for no normalization.
        alpha (float): The α parameter used for ELU/CELU activations. Default is 1.0.
        negative_slope (float): The negative slope parameter used for LeakyReLU. Default is 1e-2.

    Returns:
        nn.Sequential: A fully constructed MLP as a `torch.nn.Sequential` block.
    """
    layers = nn.ModuleList([])

    # Standardize act_func to a list of activation layers
    num_layers = len(list_dims) - 1
    if act_func is None or not act_func:
        act_funcs = [nn.Identity()] * (num_layers - 1)
    elif isinstance(act_func, str):
        act_funcs = [activation_func(act_func, alpha=alpha, negative_slope=negative_slope)] * (num_layers - 1)
    elif isinstance(act_func, list):
        assert (len(act_func) == num_layers - 1), f"Length of act_func list {len(act_func)} must match the number of hidden layers {(len(list_dims) - 2)}."
        act_funcs = [activation_func(name, alpha=alpha, negative_slope=negative_slope) for name in act_func]
    else:
        raise TypeError("act_func must be a string or a list of strings.")

    if num_layers > 0:
        for i in range(num_layers):
            in_dim = list_dims[i]
            out_dim = list_dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                # Normalization
                if norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(out_dim))
                elif norm_type == 'layer':
                    layers.append(nn.LayerNorm(out_dim))
                elif (norm_type == 'NA') or (norm_type is None):
                    layers.append(nn.Identity())
                elif norm_type is not None:
                    raise ValueError(f"Unsupported normalization type: {norm_type}")
                
                # Activation and Dropout
                layers.append(act_funcs[i])
                layers.append(nn.Dropout(dropout))
    else:
        layers.append(nn.Identity())

    return nn.Sequential(*layers)

    
def normalize_adjacency(
    adjacency,
    degree,
    laplacian=False
):
    """
    Normalize adjacency (or Laplacian) matrices for both single and multi-relational cases.

    Args:
        adjacency (Tensor): Shape [B, N, N] or [B, R, N, N].
        degree (Tensor):    Same shape as adjacency.
        laplacian (bool):   Whether to compute normalized Laplacian instead of adjacency.

    Returns:
        Tensor: Normalized adjacency or Laplacian matrix of same shape as input.
    """
    if adjacency.dim() == 4:  # [B, R, N, N] multi-relational case
        B, R, N, _ = adjacency.shape
        normalized = torch.zeros_like(adjacency)
        
        for rel in range(R):
            adj = adjacency[:, rel]       # [B, N, N]
            deg = degree[:, rel]   if degree.dim() == 4  else degree    # [B, N, N]

            deg_inv_sqrt = deg.clone()
            mask = deg_inv_sqrt != 0
            deg_inv_sqrt[mask] = deg_inv_sqrt[mask].pow(-0.5)

            if laplacian:
                lap = deg - adj
                norm = torch.bmm(torch.bmm(deg_inv_sqrt, lap), deg_inv_sqrt)
            else:
                norm = torch.bmm(torch.bmm(deg_inv_sqrt, adj), deg_inv_sqrt)

            normalized[:, rel] = norm

        return normalized

    elif adjacency.dim() == 3:  # [B, N, N] single-relation case
        deg_inv_sqrt = degree.clone()
        mask = deg_inv_sqrt != 0
        deg_inv_sqrt[mask] = deg_inv_sqrt[mask].pow(-0.5)

        if laplacian:
            lap = degree - adjacency
            normalized = torch.bmm(torch.bmm(deg_inv_sqrt, lap), deg_inv_sqrt)
        else:
            normalized = torch.bmm(torch.bmm(deg_inv_sqrt, adjacency), deg_inv_sqrt)

        return normalized

    else:
        raise ValueError("Adjacency tensor must be 3D or 4D")

    
class Identity(nn.Module):
    """
    Identity layer used to pass input without modification.

    Task:
        Acts as a placeholder in models when conditional operations like dropout 
        or batch norm are toggled off.

    Input:
        Tensor of any shape.

    Output:
        Same tensor, unchanged.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    
class Flatten(nn.Module):
    """
    Flattens all dimensions of a tensor except the batch dimension.

    Task:
        Prepares convolutional outputs for linear layers.

    Input:
        Tensor of shape (N, C, H, W)

    Output:
        Tensor of shape (N, C*H*W)
    """
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

    
class View_(nn.Module):
    """
    Reshapes tensor to a specified shape using `.view`.

    Task:
        Useful in pipelines like nn.Sequential where dynamic reshaping is needed.

    Args:
        size (tuple): Target shape (can include -1 for inferred dimensions)

    Input:
        Tensor of any shape.

    Output:
        Tensor reshaped to specified dimensions.
    """
    def __init__(self, size):
        super(View_, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class SMILESLinearNet(nn.Module):
    """
    Task:
        Feedforward network for regression or classification on SMILES-based vector inputs.

    Inputs:
        - list_dims (List[int]): List of dimensions for each linear layer.
        - dropout (float): Dropout probability applied after each layer except the last.

    Outputs:
        - Tensor of shape [batch_size, output_dim] after passing through linear layers.
    """
    def __init__(self, list_dims, dropout=0.2, act_func='relu', norm_type='layer'):
        super().__init__()
        
        self.class_def = inspect.getsource(self.__class__)
        self.list_dims = list_dims
        self.dropout = dropout
        self.act_func = act_func
        self.norm_type = norm_type
        self.network_pyfile = network_pyfile
        self.MLP = make_mlp(list_dims=list_dims,
                            dropout=dropout,
                            act_func=act_func,
                            norm_type=norm_type
                            )
        
        # Weight Initialization
        for layer in self.MLP:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def __repr__(self):
        return auto_repr(self)
    
    def forward(self, x):
        """
        Task:
            Forward pass through MLP with ReLU, BatchNorm, and Dropout.

        Inputs:
            - x (Tensor): Input tensor of shape [batch_size, list_dims[0]]

        Outputs:
            - Tensor: Output tensor after final linear layer.
        """
        return self.MLP(x)


## Multi Layer GCN Network
class RGCNConv(nn.Module):
    """
    Task:
        Relation-aware Graph Convolution Layer for multi-relational graphs.

    Inputs:
        - in_channels (int): Input node feature dim.
        - out_channels (int): Output node feature dim.
        - num_relations (int): Number of bond types or edge relations.

    Outputs:
        - Tensor: Transformed node features with all relation contributions.
    """
    def __init__(self, in_channels, out_channels, num_relations=4, bias=True):
        super(RGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weights = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.self_loop_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()
        
    def __repr__(self):
        return auto_repr(self)

    def reset_parameters(self):
        """Task: Initialize parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.self_loop_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adjacency_tensor):
        """
        Task:
            Perform multi-relational message passing.

        Inputs:
            - x (Tensor): Node features [batch_size, num_nodes, in_channels]
            - adjacency_tensor (Tensor): [batch_size, num_relations, num_nodes, num_nodes]

        Outputs:
            - Tensor: Updated node features [batch_size, num_nodes, out_channels]
        """
        batch_size, num_relations, num_nodes, _ = adjacency_tensor.size()
        out = torch.zeros(batch_size, num_nodes, self.out_channels, device=x.device)

        # Option - A (for loop)
        # ********************************************************************************
        for rel in range(num_relations):
            adj = adjacency_tensor[:, rel]  # Shape: [batch_size, num_nodes, num_nodes]

            h_rel = torch.matmul(x, self.weights[rel])  # Shape: [batch_size, num_nodes, out_channels]
            out += torch.matmul(adj, h_rel)
        # ********************************************************************************
            
        # # Option - B (vectorised) 
        # # ********************************************************************************
        # h = torch.matmul(x.unsqueeze(1),          # [B,1,N,Fin]
        #                  self.weights.unsqueeze(0) # [1,R,Fin,Fout]
        #                  )  # -> [B,R,N,Fout]
        # out = torch.matmul(adjacency_tensor, h)  # [B,R,N,Fout]
        # out = out.sum(dim=1)
        # # ********************************************************************************
            
        out += torch.matmul(x, self.self_loop_weight)
        # out = out/num_relations

        if self.bias is not None:
            out += self.bias

        return out
