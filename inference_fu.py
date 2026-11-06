import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import torch, pickle
from pathlib import Path

import torch
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference_utils_fu.graph_utils_smiles_fu import smiles_to_graph
from inference_utils_fu.data_utils_fu import load_from_pickle, load_json, load_project_root, _round
from inference_utils_fu.train_module_fu import Trainer_functions
from inference_utils_fu.model_classes_fu import LOAD_MODEL_REGISTRY
from inference_utils_fu.data_utils_smiles_fu import plot_smiles_grid
from inference_utils_fu.config.inference_config import FUConfig

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

__all__ = ['Predict_FU']

project_root = load_project_root(1)
fold = 7
num_workers = 8
metadata_path = Path(f"{project_root}/metadata")
config = FUConfig(fold=fold, num_workers=num_workers, metadata_path=metadata_path)

class GraphBuilder:

    def __init__(self, config):

        self.config = config

        with open(config.dataset_mean_path, "rb") as f:
            self.dataset_mean = pickle.load(f)

        with open(config.dataset_std_path, "rb") as f:
            self.dataset_std = pickle.load(f)

        metadata = load_json(config.graph_metadata_path)
        self.hybridization_list = metadata["hybridization_list"]
        self.chiraltypes = metadata["chiraltypes"]
        self.features_list = load_json(config.node_feature_path)

    def build(self, smiles):

        graph =  smiles_to_graph(smiles=smiles,
                                 dataset_mean=self.dataset_mean,
                                 dataset_std=self.dataset_std,
                                 features_list=self.features_list,
                                 hybridization_list=self.hybridization_list,
                                 chiraltypes=self.chiraltypes,
                                 max_num_atoms=self.config.max_num_atoms,
                                 device="cpu",
                                 )

        return graph


class FUModelManager:

    def __init__(self, config):

        self.config = config
        self.registry = LOAD_MODEL_REGISTRY()
        self.fold_registry = self.registry[f"F{config.fold}"]
        self.gcn_model = self._load_gcn()

    def _load_gcn(self):

        model = self.fold_registry["model_fn"]()
        model_path = self.fold_registry["repr_model"]
        Trainer_functions.load_test_network(network=model, temp_network_path=model_path, verbose=False)
        model.to(self.config.device)
        model.eval()

        return model

    def load_mlp(self):

        model = self.fold_registry["mlp_model_fn"]()
        mlp_path = self.fold_registry["nn_model"]
        Trainer_functions.load_test_network(network=model, temp_network_path=mlp_path, verbose=False)
        model.to(self.config.device)
        model.eval()

        return model

    def load_nonNN_regressor(self, name):

        path = self.fold_registry[f"{name}_model"]

        return load_from_pickle(path)


class FULatent:

    def __init__(self, smiles, config, graph_builder=None, model_manager=None):

        self.smiles = [smiles] if isinstance(smiles, str) else list(smiles)
        self.config = config
        self.graph_builder = graph_builder if graph_builder is not None else GraphBuilder(config)
        self.model_manager = model_manager if model_manager is not None else FUModelManager(config)
        self.latent_vec = self._extract_latent_vectors()

    def _build_graphs(self):

        workers = self.config.num_workers
        with Pool(workers) as pool:
            graphs = pool.map(self.graph_builder.build, self.smiles)

        return graphs

    def _extract_latent_vectors(self):

        graphs = self._build_graphs()
        latent_vectors = []
        model = self.model_manager.gcn_model
        device = self.config.device

        with torch.inference_mode():
            for start in range(0, len(graphs), self.config.batch_size):

                batch = graphs[start:start + self.config.batch_size]
                fv = torch.stack([g[0] for g in batch]).to(device)
                adj = torch.stack([g[1] for g in batch]).to(device)
                deg = torch.stack([g[2] for g in batch]).to(device)
                fmask = torch.stack([g[3] for g in batch]).to(device)
                amask = torch.stack([g[4] for g in batch]).to(device)

                graph_input = {"feature_vector": fv,
                               "adjacency_tensor": adj,
                               "degree_tensor": deg,
                               "feature_mask": fmask,
                               "adjacency_mask": amask,
                               }

                latent = model.get_features(graph_input)
                latent_vectors.append(latent)

        return torch.cat(latent_vectors, dim=0)


class FURegressor:

    def __init__(self, model_manager, config):

        self.model_manager = model_manager
        self.config = config

    def predict_mlp(self, latent_vectors):

        model = self.model_manager.load_mlp()

        predictions = []
        with torch.inference_mode():
            for start in range(0, len(latent_vectors), self.config.batch_size):

                batch = latent_vectors[start:start + self.config.batch_size]
                pred = model(batch)
                predictions.extend(pred.detach().cpu().numpy().reshape(-1))

        return {"MLP": np.asarray(predictions)}

    def predict_nonNN(self, latent_vectors, regressors):

        results = {}
        if isinstance(regressors, str):
            regressors = [regressors]

        for name in regressors:

            model = self.model_manager.load_nonNN_regressor(name)
            predictions = []
            for start in range(0, len(latent_vectors), self.config.batch_size):
                batch = latent_vectors[start:start + self.config.batch_size]
                if (name in ['rf', 'svm']):
                    batch = batch.detach().cpu().numpy()
                    pred = model.predict(batch)
                elif (name == 'xgb'):
                    batch = batch.to(self.config.device)
                    model.set_params(tree_method='hist', device='cuda')
                    pred = model.predict(batch)
                predictions.extend(pred)

            results[name.upper()] = np.asarray(predictions)

        return results

    def predict_gcn(self, latent_vectors):
    
        predictions = []
        gcn_model = self.model_manager._load_gcn()
        gcn_model.eval()

        with torch.no_grad():
            for start in range(0, len(latent_vectors), self.config.batch_size):
                batch = latent_vectors[start:start + self.config.batch_size]
                pred = gcn_model.latent2Fu(batch)
                pred = pred.detach().cpu().numpy().reshape(-1)
                predictions.extend(pred)

        out = {'GCN': np.round(np.asarray(predictions), 4)}

        return out

def Predict_FU(smiles, regressor='all'):

    if regressor == 'all':
        regressor = ['rf', 'svm', 'mlp', 'xgb']

    graph_builder = GraphBuilder(config)
    model_manager = FUModelManager(config)
    fu_latent = FULatent(smiles=smiles, 
                         config=config, 
                         graph_builder=graph_builder, 
                         model_manager=model_manager
                         )
    predictor = FURegressor(config=config, model_manager=model_manager)

    latent_vector = fu_latent.latent_vec
    smiles = fu_latent.smiles

    if isinstance(regressor, str):
        regressor = regressor.lower()
        if regressor in [None, 'gcn']:
            out = predictor.predict_gcn(latent_vector)
        elif regressor in ['rf', 'svm', 'xgb']:
            out = predictor.predict_nonNN(regressors=regressor, latent_vectors=latent_vector)
        elif regressor == 'mlp':
            out = predictor.predict_mlp(latent_vector)

    elif isinstance(regressor, (list, tuple)):
        regressor = [x.lower() for x in regressor]
        _non_NN = list(set(regressor).intersection({'rf', 'svm', 'xgb'}))
        _NN = set(regressor).intersection({'mlp'})
        _GCN = set(regressor).intersection({'gcn'})
        out = {}
        if _non_NN:
            out_non_NN = predictor.predict_nonNN(regressors=_non_NN, latent_vectors=latent_vector)
            out = {**out_non_NN}
        if _NN:
            out_NN = predictor.predict_mlp(latent_vector)
            out = {**out, **out_NN}
        if _GCN:
            out_NN = predictor.predict_gcn(latent_vector)
            out = {**out, **out_NN}

    if isinstance(smiles, (list, tuple)):
        img = [plot_smiles_grid(smiles_list=[smi], cols=1, image_size=(600, 600), 
                                figsize=(12, 8), atom_font_size=25, fmt='svg', show=False) for smi in smiles]
    elif isinstance(smiles, str):
        img = [plot_smiles_grid(smiles_list=[smiles], cols=1, image_size=(600, 600), 
                                figsize=(12, 8), atom_font_size=25, fmt='svg', show=False)]
    
    output = format_predictions(smiles=smiles, predictions=out, images=img)

    return output

def format_predictions(smiles, predictions, images=None):

    output = {"smiles": smiles, **predictions}
    if images is not None:
        output["image"] = images

    df = pd.DataFrame(output)
    prediction_columns = [c for c in df.columns if c not in {"smiles", "image"}]
    for column in prediction_columns:
        df[column] = (df[column].map(lambda x: np.clip(_round(x, 4),1e-5,1.0)))

    return df.to_dict(orient="records")