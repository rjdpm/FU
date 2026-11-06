from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass(frozen=True)
class FUConfig:

    fold: int = 7
    tau: float = 0.25

    max_num_atoms: int = 72
    batch_size: int = 512
    num_workers: int = 16

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    metadata_path: str = Path("../../metadata")

    @property
    def node_feature_path(self) -> Path:
        return self.metadata_path/"atom_props_keys_numeric.json"

    @property
    def graph_metadata_path(self) -> Path:
        return self.metadata_path/"Dataset_atoms_maxatoms_hybds_chityp_bndlst.json"
    
    @property
    def fold_dir(self) -> Path:
        return (self.metadata_path/"fold_datasets_metadata"/f"F{self.fold}")

    @property
    def scaler_path(self) -> Path:
        return self.fold_dir / "StandardScaler_labels.pkl"

    @property
    def dataset_mean_path(self) -> Path:
        return self.fold_dir / "dataset_mean.pkl"

    @property
    def dataset_std_path(self) -> Path:
        return self.fold_dir / "dataset_std.pkl"