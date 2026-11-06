import torch
import numpy as np
import pickle, gzip
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from ordered_set import OrderedSet
from pathlib import Path

# from logP_values import *
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from compute_molecular_desc_fu import AtomicDescriptorCalculator
from data_utils_fu import load_project_root

__all__ = [
    'list_to_onehot',
    'list_to_onehot_oov',
    'Multirelational_GraphDataset',
    'SmilesDataset_graph_gen',
    'GraphData_from_pickle',
    'smiles_to_graph',
    'feature_representation'
]

PROJECT_ROOT = load_project_root(1)

class Multirelational_GraphDataset(Dataset):
    
    def __init__(self,
                 smi_list,
                 labels,
                 max_num_atoms = 100,
                 hybridization_list=['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP2D', 'SP3D', 'SP3D2', 'OTHER'],
                 chiraltypes=['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'],
                 **kwargs
                 ):
        
        self.smi_list = smi_list
        self.labels = labels
        self.max_num_atoms = max_num_atoms
        self.hybridization_list = hybridization_list
        self.chiraltypes = chiraltypes
    
    def __len__(self):
        return len(self.smi_list)

    def _get_one(self, idx):
        
        smi = self.smi_list[idx]
        atom_features, bond_matrix = self.smi2feature(smi)
        y  = self.labels[idx]
        
        return smi, atom_features, bond_matrix, y
    
    def __getitem__(self, idx):
        if isinstance(idx, (slice, list, tuple)):  
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
                print(len(self))
            else:
                indices = idx
            items = [self._get_one(i) for i in indices]
            return self.collate_fn(items)
        return self._get_one(idx)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate a list of (smi, atom_feature_dict, adjacency_matrix, degree_matrix, y)
        into numpy arrays (safe for multiprocessing).
        """
        smi_list, atom_features, bond_matrix, ys = zip(*batch)

        smi_list = list(smi_list)

        # collate atom feature dicts
        feature_keys = atom_features[0].keys()
        atom_feature_vectors = {}
        for key in feature_keys:
            try:
                atom_feature_vectors[key] = np.stack([afd[key] for afd in atom_features], axis=0)#.astype(np.float32)
            except:
                atom_feature_vectors[key] = [afd[key] for afd in atom_features]
                
        bond_matrix_batch = {}
        for key in bond_matrix[0].keys():
            try:
                bond_matrix_batch[key] = np.stack([bm[key] for bm in bond_matrix], axis=0)#.astype(np.float32)
            except ValueError:
                bond_matrix_batch[key] = [bm[key] for bm in bond_matrix]

        ys                = np.array(ys, dtype=np.float32)
        return_vals = {'smiles':smi_list,
                       'atom_feature_vectors':atom_feature_vectors,
                       'bond_matrix':bond_matrix_batch,
                       'labels':ys,
                       }
        
        return return_vals
    
    def smi2feature(self, smi):
        
        descriptor_calculator = AtomicDescriptorCalculator(smi=smi,
                                                           hybridization_list=self.hybridization_list,
                                                           chiraltypes=self.chiraltypes
                                                           )
        atom_features, bond_features = descriptor_calculator.get_all_molecule_properties()
        
        bond_matrix_temp = bond_features['bond_matrix']
        bond_type = bond_features['bond_type_str']
        bond_matrix = {'bond_matrix':bond_matrix_temp, 'bond_type':bond_type}
        
        return atom_features, bond_matrix

            
def process_single_sample(data_point):
        
        smi, atom_feature_vector, bond_matrix, y = data_point
        if atom_feature_vector is None:
            return None
        try:
            return {
                'SMILES': smi,
                'atom_feature_vector': atom_feature_vector,
                'bond_matrix': bond_matrix,
                'label': y
            }     
        except:
            print(f'Error found at: {data_point}')
        
def SmilesDataset_graph_gen(split, dataset, out_path = 'SmilesDataset_graph',
                            mean_std_flag = False, num_workers=int(cpu_count()/4),
                            chunksize=1000, filetype='zip', save_per_item=10000):
    
    total_len = len(dataset)
    num_batches = (total_len // save_per_item) + 1
    SmilesDataset_graph = []

    with Pool(num_workers) as pool:
        for batch_idx in tqdm(range(num_batches), desc=f"{split}-batches"):

            start = batch_idx * save_per_item
            end   = min((batch_idx + 1) * save_per_item, total_len)
            if start >= total_len:
                break

            print(f"Processing samples from {start} to {end}...")
            batch_items = [dataset[idx] for idx in range(start, end)]
            temp_results = list(
                                tqdm(
                                    pool.imap(process_single_sample, batch_items, chunksize=chunksize),
                                    total=len(batch_items),
                                    desc=f"{split}-batch-{batch_idx}"
                                )
                            )
            temp_results = [x for x in temp_results if x is not None]
            SmilesDataset_graph.extend(temp_results)

            # Save checkpoints
            if filetype == 'zip':
                filename = f"{out_path}_{split}.pkl.gz"
                with gzip.open(filename, "wb") as f:
                    pickle.dump(SmilesDataset_graph, f)

            elif filetype == 'pkl':
                filename = f"{out_path}_{split}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(SmilesDataset_graph, f)

            print(f"Saved {len(SmilesDataset_graph)} items to: {filename}")
            
    if mean_std_flag:
        _, atom_feature_vector, _, _ = dataset[0]
        features_keys = list(atom_feature_vector.keys())
        
        ## Subsampling for large datasets to reduce required time
        subset_size = min(50000, len(SmilesDataset_graph))
        sampled_idxs = torch.randperm(len(SmilesDataset_graph))[:subset_size]
        SmilesDataset_graph = [SmilesDataset_graph[i] for i in sampled_idxs]
        
        mean_dict, std_dict = dict({}), dict({})
        for key in OrderedSet(features_keys) - OrderedSet({'symbol', 'block', 'sh_struct', 'donor_atoms', 'acceptor_atoms', 'atom_idx'}):
            temp = [np.squeeze(data['atom_feature_vector'][key]) for data in SmilesDataset_graph]
            concatenated_features = np.concatenate(temp, axis=0)
            concatenated_features[concatenated_features == AtomicDescriptorCalculator.MASKING_VALUE] =  0
            mean = concatenated_features.mean(axis=0)
            std = concatenated_features.std(axis=0)
            if np.isscalar(std):
                if std == 0.:
                    std = 1.0
                mean_dict[key] = np.array([mean])
                std_dict[key] = np.array([std])
            else:
                std[std == 0] = 1.0
                mean_dict[key] = mean
                std_dict[key] = std
            
        filename = '/'.join(out_path.split('/')[:-1]+['dataset_mean.pkl'])
        with open(filename, "wb") as outfile:
            pickle.dump(mean_dict, outfile)
        filename = '/'.join(out_path.split('/')[:-1]+['dataset_std.pkl'])
        with open(filename, "wb") as outfile:
            pickle.dump(std_dict, outfile)
            
class GraphData_from_pickle(Dataset):
    
    def __init__(self,
                 dataset_path,
                 features_list,
                 max_num_atoms=65,
                 padding=True
                 ):
        self.features_list = features_list
        self.max_num_atoms = max_num_atoms
        self.padding = padding
        if dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as fp:
                self.dataset = pickle.load(fp)
        if dataset_path.endswith('.pkl.gz'):
            with gzip.open(dataset_path, 'rb') as fp:
                self.dataset = pickle.load(fp)
        
        filename = '/'.join(dataset_path.split('/')[:-1]+['dataset_mean.pkl'])
        with open(filename, 'rb') as fp:
            self.dataset_mean = pickle.load(fp)
        filename = '/'.join(dataset_path.split('/')[:-1]+['dataset_std.pkl'])
        with open(filename, 'rb') as fp:
            self.dataset_std = pickle.load(fp)
            
        self.mean = np.concatenate([self.dataset_mean[k] for k in self.features_list],axis=0)
        self.std = np.concatenate([self.dataset_std[k] for k in self.features_list],axis=0)
        
    def __len__(self):
        return len(self.dataset)

    def _get_one(self, idx):
        
        data=self.dataset[idx]
        feature_vector, adjacency_tensor, degree_tensor, feat_mask, adj_mask, y = self.smi2feature(data)
        
        return feature_vector, adjacency_tensor, degree_tensor, feat_mask, adj_mask, y
    
    def __getitem__(self, idx):
        if isinstance(idx, (slice, list, tuple)):  
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
            else:
                indices = idx
            items = [self._get_one(i) for i in indices]
            return self.collate_fn(items)
        return self._get_one(idx)
    
    def collate_fn(self, batch):
        """
        Collate a list of (feature_vector, adjacency_tensor, degree_tensor, feat_mask, adj_mask, y)
        into batched tensors.
        """
        feature_vectors, adjacency_matrices, degree_matrices, feat_masks, adj_masks, ys = zip(*batch)

        # Convert to torch tensors
        feature_vectors   = [torch.as_tensor(fv, dtype=torch.float32) for fv in feature_vectors]
        adjacency_matrices = [torch.as_tensor(adj, dtype=torch.float32) for adj in adjacency_matrices]
        degree_matrices    = [torch.as_tensor(deg, dtype=torch.float32) for deg in degree_matrices]
        feat_masks         = [torch.as_tensor(fm, dtype=torch.bool) for fm in feat_masks]
        adj_masks          = [torch.as_tensor(am, dtype=torch.bool) for am in adj_masks]
        ys                 = [torch.as_tensor(y, dtype=torch.float32) for y in ys]

        try:
            feature_vectors = torch.stack(feature_vectors, dim=0)
            feat_masks = torch.stack(feat_masks, dim=0)
        except RuntimeError:
            pass

        try:
            adjacency_matrices = torch.stack(adjacency_matrices, dim=0)
            adj_masks = torch.stack(adj_masks, dim=0)
        except RuntimeError:
            pass

        try:
            degree_matrices = torch.stack(degree_matrices, dim=0)
        except RuntimeError:
            pass

        ys = torch.stack(ys, dim=0)

        return feature_vectors, adjacency_matrices, degree_matrices, feat_masks, adj_masks, ys
    
    @staticmethod
    def build_3d_adjacency(bond_matrix, bond_type, num_atoms):
        
        bond_map = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        
        # Initialize adjacency tensor: [N, N, bond_features]
        A = np.zeros((4, num_atoms, num_atoms), dtype=np.float32)

        # number of bonds
        num_bonds = bond_matrix.shape[1]

        for b in range(num_bonds):
            i = int(bond_matrix[0, b])
            j = int(bond_matrix[1, b])

            btype = bond_type[b]
            channel = bond_map.get(btype, None)

            if channel is not None:
                A[channel, i, j] = 1.0
                A[channel, j, i] = 1.0
                
        for k in range(num_atoms):
            for l in range(4):
                A[l, k, k] = 1.0

        return torch.tensor(A)
    
    def smi2feature(self, data):
        
        atom_feature_vector = data['atom_feature_vector']
        try:
            features = [atom_feature_vector[k] for k in self.features_list]
        except:
            raise AttributeError(f'Features should be in: {list(atom_feature_vector.keys())}')
        
        try:
            atom_feature_vector = np.concatenate(features, axis=1, dtype=np.float32)
            atom_feature_vector[atom_feature_vector == AtomicDescriptorCalculator.MASKING_VALUE] =  0
            n_atoms = atom_feature_vector.shape[0]
            if self.max_num_atoms == 0:
                self.max_num_atoms = n_atoms
            feat_mask = torch.zeros((self.max_num_atoms, atom_feature_vector.shape[1]), dtype=torch.bool)
            feat_mask[:n_atoms] = True
            feat_mask[:n_atoms] &= torch.tensor(atom_feature_vector != -99999.0)
        except:
            raise ValueError(f'Error found in: SMILES - {data["SMILES"]}, Features - {atom_feature_vector}')
        
         # Convert to tensor
        atom_feature_vector = torch.from_numpy(atom_feature_vector).float()
        bond_matrix = data['bond_matrix']
        
        adjacency_tensor = GraphData_from_pickle.build_3d_adjacency(bond_matrix['bond_matrix'], bond_matrix['bond_type'], self.max_num_atoms)
        adj_mask = torch.zeros((4, self.max_num_atoms, self.max_num_atoms), dtype=torch.bool)
        adj_mask[:, :n_atoms, :n_atoms] = True
        degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)])
        
        # normalize and pad if needed
        if self.padding:
            feature_vector = torch.zeros((self.max_num_atoms, atom_feature_vector.shape[1]), dtype=torch.float32)
            feature_vector[:atom_feature_vector.shape[0], :] = (atom_feature_vector - self.mean) / self.std
        else:
            feature_vector = (atom_feature_vector - self.mean) / self.std
        
        # get adjacency, degree, label
        y                = torch.tensor(data['label'], dtype=torch.float32)
        
        return feature_vector, adjacency_tensor, degree_tensor, feat_mask, adj_mask, y
    
    def get_smiles(self, idx):
        smi = self.dataset[idx]['SMILES']
        return smi
    
    def get_atom_info_vector(self, idx):
        
        data=self.dataset[idx]
        atom_feature_vector = data['atom_feature_vector']
        
        return atom_feature_vector
    
    @staticmethod
    def GcnDictData2GcnInput(data,
                            dataset_mean,
                            dataset_std,
                            max_num_atoms,
                            features_list=['symbol', 'atom_properties', 'hybridization', 'aromaticity', 'ring', 'chirality', 'neighbour_info'], #'3d_descriptors'
                            padding=True
                            ):
        mean = np.concatenate([dataset_mean[k] for k in features_list],axis=0)
        std = np.concatenate([dataset_std[k] for k in features_list],axis=0)
        
        atom_feature_vector = data['atom_feature_vector']
        try:
            features = [atom_feature_vector[k] for k in features_list]
        except:
            raise AttributeError(f'Features should be in: {list(atom_feature_vector.keys())}')
        
        try:
            atom_feature_vector = np.concatenate(features, axis=1, dtype=np.float32)
            atom_feature_vector[atom_feature_vector == AtomicDescriptorCalculator.MASKING_VALUE] =  0
            n_atoms = atom_feature_vector.shape[0]
            if max_num_atoms == 0:
                max_num_atoms = n_atoms
            feat_mask = torch.zeros((max_num_atoms, atom_feature_vector.shape[1]), dtype=torch.bool)
            feat_mask[:n_atoms] = True
            feat_mask[:n_atoms] &= torch.tensor(atom_feature_vector != -99999.0)
        except:
            raise ValueError(f'Error found in: SMILES - {data["SMILES"]}, Features - {atom_feature_vector}')
        
         # Convert to tensor
        atom_feature_vector = torch.from_numpy(atom_feature_vector).float()
        bond_matrix = data['bond_matrix']
        
        adjacency_tensor = GraphData_from_pickle.build_3d_adjacency(bond_matrix['bond_matrix'], bond_matrix['bond_type'], max_num_atoms)
        adj_mask = torch.zeros((4, max_num_atoms, max_num_atoms), dtype=torch.bool)
        adj_mask[:, :n_atoms, :n_atoms] = True
        degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)])
        
        # normalize and pad if needed
        if padding:
            feature_vector = torch.zeros((max_num_atoms, atom_feature_vector.shape[1]), dtype=torch.float32)
            feature_vector[:atom_feature_vector.shape[0], :] = (atom_feature_vector - mean) / std
        else:
            feature_vector = (atom_feature_vector - mean) / std
        
        # get adjacency, degree, label
        y                = torch.tensor(data['label'], dtype=torch.float32)
        
        return feature_vector, adjacency_tensor, degree_tensor, feat_mask, adj_mask, y
    

def smiles_to_graph(smiles,
                    dataset_mean, 
                    dataset_std, 
                    features_list, 
                    max_num_atoms=65, 
                    hybridization_list=['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 
                                        'SP2D', 'SP3D', 'SP3D2', 'OTHER'],
                    chiraltypes=['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'],
                    padding=True,
                    label=None,
                    device = 'cuda:0' if torch.cuda.is_available() else 'cpu',
                    ):
    """
    Convert a SMILES into the graph representation expected by the model.

    Parameters
    ----------
    smiles : str
    dataset_mean : dict
    dataset_std : dict
    features_list : list
    max_num_atoms : int
    padding : bool
    label : float or None

    Returns
    -------
    feature_vector : torch.FloatTensor
        (max_num_atoms, feature_dim)

    adjacency_tensor : torch.FloatTensor
        (4, max_num_atoms, max_num_atoms)

    degree_tensor : torch.FloatTensor
        (4, max_num_atoms, max_num_atoms)

    feat_mask : torch.BoolTensor
        (max_num_atoms, feature_dim)

    adj_mask : torch.BoolTensor
        (4, max_num_atoms, max_num_atoms)

    y : torch.FloatTensor
    """

    mean = np.concatenate([dataset_mean[k] for k in features_list], axis=0)
    std = np.concatenate([dataset_std[k] for k in features_list], axis=0)
    descriptor_calculator = AtomicDescriptorCalculator(smi=smiles,
                                                       hybridization_list=hybridization_list,
                                                       chiraltypes=chiraltypes,
                                                       )
    atom_features, bond_features = descriptor_calculator.get_all_molecule_properties()
    features = [atom_features[k] for k in features_list]
    atom_feature_vector = np.concatenate(features,
                                         axis=1,
                                         dtype=np.float32,
                                         )

    atom_feature_vector[atom_feature_vector == AtomicDescriptorCalculator.MASKING_VALUE] = 0

    n_atoms = atom_feature_vector.shape[0]

    if max_num_atoms == 0:
        max_num_atoms = n_atoms

    # Feature mask
    feat_mask = torch.zeros((max_num_atoms, atom_feature_vector.shape[1]), dtype=torch.bool,)
    feat_mask[:n_atoms] = True
    feat_mask[:n_atoms] &= torch.tensor(atom_feature_vector != -99999.0)

    # Normalize features
    atom_feature_vector = torch.from_numpy(atom_feature_vector).float()

    normalized = (atom_feature_vector - mean) / std

    if padding:
        feature_vector = torch.zeros((max_num_atoms, normalized.shape[1]), dtype=torch.float32,)
        feature_vector[:n_atoms] = normalized
    else:
        feature_vector = normalized

    # Build adjacency tensor
    adjacency_tensor = GraphData_from_pickle.build_3d_adjacency(bond_features["bond_matrix"],
                                                                bond_features["bond_type_str"],
                                                                max_num_atoms,
                                                                )

    # Degree tensor and Adjacency mask
    degree_tensor = torch.stack([torch.diag(channel.sum(dim=1)) for channel in adjacency_tensor])
    adj_mask = torch.zeros((4, max_num_atoms, max_num_atoms), dtype=torch.bool,)
    adj_mask[:, :n_atoms, :n_atoms] = True

    # Label
    if label is None:
        y = torch.tensor(float("nan"), dtype=torch.float32)
    else:
        y = torch.tensor(label, dtype=torch.float32)

    feature_vector = feature_vector.to(device)
    adjacency_tensor = adjacency_tensor.to(device)
    degree_tensor = degree_tensor.to(device)
    feat_mask = feat_mask.to(device)
    adj_mask = adj_mask.to(device)
    y = y.to(device)

    return feature_vector, adjacency_tensor, degree_tensor, feat_mask, adj_mask, y

    
def feature_representation(dataset,
                           model,
                           device='cuda:0' if torch.cuda.is_available() else 'cpu',
                           input_params = ["feature_vector", "adjacency_tensor","degree_tensor",
                                           "feature_mask", "adjacency_mask", "labels"]
                           ):
    
    '''
    dataset: A dataset with first entry of every datapoint is a SMILES
    '''
    
    feature_dim = model.feature_dim
    num_samples = len(dataset)

    features = np.zeros((num_samples, feature_dim))
    labels = np.zeros((num_samples))
    smi_list = [None]*num_samples

    model.eval()
    with torch.no_grad():  
        for i, data in tqdm(enumerate(dataset), total=len(dataset), desc=f'Feature Extraction: '):
            batch = {k: torch.tensor(v).to(device).unsqueeze(0) for k, v in zip(input_params, data)}
            targets = batch.pop('labels').squeeze()
            outputs = model.get_features(**batch)

            outputs_np = outputs.cpu().numpy()
            features[i] = outputs_np
            labels[i] = targets.cpu().numpy()
            smi_list[i] = dataset.get_smiles(i)

    return features, labels, smi_list