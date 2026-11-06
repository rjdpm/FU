import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_fu import CG_GML
from data_utils_fu import load_project_root
from models_module_fu import SMILESLinearNet

project_root = load_project_root(1)

__all__ = ['desc_list',
           'LOAD_MODEL_REGISTRY',
           'all_model_paths_fdim'
           ]

gcn_args={'list_dims_gcn': [53, 167, 282, 397, 512], 
        'list_dims_fc': [512], 
        'dropout': 0.3, 
        'max_num_atom': 72, 
        'act_func': 'relu', 
        'act_func_gcn': 'relu', 
        'norm_type': 'layer', 
        'use_attention': True, 
        'attn_dim': 16, 
        'num_heads': 4, 
        'attn_type': 'attention pool', 
        'use_gin_readout': False
        }

mlp_args={'list_dims': [512, 256, 1], 
        'dropout': 0, 
        'act_func': 'relu', 
        'norm_type': 'layer', 
        'alpha': 1, 
        'negative_slope': 0.01
        }

regr_mlp_args = {'list_dims':[512, 512, 384, 256, 128, 1],
                'dropout':0.2,
                'act_func':'relu',
                'norm_type':'layer'
                }
use_sigmoid=False
add_bias=True
n_coeffs=3


desc_list_fu = {'t_0p25': ['FilterItLogS', 'GATS1i', 'AATS1i', 'HybRatio', 'FpDensityMorgan1', 'BalabanJ', 'HallKierAlpha'],}
desc_list_ka = {'t_0p25': ['MolLogP', 'SMR_VSA7', 'piPC5', 'AUTOCORR2D_42', 'AATS1v', 
                           'AUTOCORR2D_25', 'fMF', 'ZMIC1', 'PEOE_VSA6', 'AATS3d', 'ATSC0i']    
                }

def get_CG_GML(tau):

    tau = f"{tau:.2f}"
    tau = 't_'+tau.replace('.', 'p')
    CG_GML_model = CG_GML(gcn_args=gcn_args,
                    mlp_args=mlp_args,
                    use_sigmoid=use_sigmoid,
                    add_bias=add_bias,
                    desc_list_ka=desc_list_ka[tau],
                    desc_list_fu=desc_list_fu[tau],
                    n_coeffs=n_coeffs
                    )
    
    return CG_GML_model

def get_MLP_model(tau):
    return SMILESLinearNet(**regr_mlp_args)

def get_repr_model_path(root=project_root, fold=1):
    model_name = 'gcn_repr_best_model_t0p25.pt'
    filepath = f'{root}/best_models_fu/F{fold}/{model_name}'
    return filepath

def get_regr_model_path(root=project_root, regressor='rf', fold=1):

    nn_model_name = 'gcn_nn_best_model.pt'
    rf_model_name = 'gcn_rf_best_model.pkl'
    svm_model_name = 'gcn_svm_best_model.pkl'
    xgb_model_name = 'gcn_xgb_best_model.pkl'
    model_names = {'rf':rf_model_name, 'nn':nn_model_name, 
                   'svm':svm_model_name, 'xgb':xgb_model_name
                   }
    model_name = model_names[regressor]
    filepath = f'{root}/best_models_fu/F{fold}/{model_name}'

    return filepath

repr_model_path_t0p25= {f'F{k}':get_repr_model_path(fold=k) for k in range(1, 11)}
regr_model_path_t0p25_rf = {f'F{k}':get_regr_model_path(regressor='rf', fold=k) for k in range(1, 11)}
regr_model_path_t0p25_nn = {f'F{k}':get_regr_model_path(regressor='nn', fold=k) for k in range(1, 11)}
regr_model_path_t0p25_svm = {f'F{k}':get_regr_model_path(regressor='svm', fold=k) for k in range(1, 11)}
regr_model_path_t0p25_xgb = {f'F{k}':get_regr_model_path(regressor='xgb', fold=k) for k in range(1, 11)}

all_model_paths = {'repr':repr_model_path_t0p25,
                   'rf':regr_model_path_t0p25_rf,
                   'nn':regr_model_path_t0p25_nn,
                   'svm':regr_model_path_t0p25_svm,
                   'xgb':regr_model_path_t0p25_xgb,
                   }
desc_list = {'fu': desc_list_fu, 'ka': desc_list_ka,}

def LOAD_MODEL_REGISTRY():

    ## Creating model registry
    MODEL_REGISTRY = {}

    tau = 0.25
    tau_name = f"{tau:.2f}"
    tau_name = 't_'+tau_name.replace('.', 'p')
    for fold in range(1, 11):

        MODEL_REGISTRY[f"F{fold}"] = {
            "tau": tau,
            "repr_model": all_model_paths["repr"][f"F{fold}"],
            "model_fn": lambda tau=tau: get_CG_GML(tau),
            "mlp_model_fn": lambda tau=tau: get_MLP_model(tau),

            "desc_fu": desc_list["fu"][tau_name],
            "desc_ka": desc_list["ka"][tau_name],

            "rf_model": all_model_paths["rf"][f"F{fold}"],
            "nn_model": all_model_paths["nn"][f"F{fold}"],
            "svm_model": all_model_paths["svm"][f"F{fold}"],
            "xgb_model": all_model_paths["xgb"][f"F{fold}"],
        }

    return MODEL_REGISTRY
