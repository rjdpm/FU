import math, sys
import numpy as np
import pandas as pd
import networkx as nx

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from collections import defaultdict
from typing import Tuple

from rdkit.Chem import ChemicalFeatures, EState
from rdkit import RDConfig
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils_fu import fix_seed, load_project_root, list_to_onehot, list_to_onehot_oov
fix_seed(np_seed=20, torch_seed=30, random_seed=10)

project_root = load_project_root(N=1)

__all__ = ['AtomicDescriptorCalculator']

class AtomicDescriptorCalculator:
    
    ACIDIC_SMARTS = [
        "[C,S](=O)[O;H1]",          # Protonated carboxyl, sulfinic acids
        "P(=O)(O)O",                # Phosphoric acids
        "P(=O)(O)[O;H1]",           # Organophosphates
        "[S;D2](O)(=O)=O",          # Sulfonic acids
        "S(=O)(=O)[O;H1]",          # Sulfonic acids (strong)
        "c[OH]",                    # phenol
        "[nH]",                     # heterocyclic N–H (pyrrolic)
        "[O;H1][C,N,S]",            # alcohol adjacent to electron-withdrawing atom
        "[C;X3](=O)[O;H1]",         # generic COOH
        "[$([O-]),$([O;H1][#6]=O)]",# oxyanion or protonated oxygen near carbonyl
    ]

    BASIC_SMARTS = [
        "[N;H2,H1;!$(NC=O)]",          # primary/secondary amines, excluding amides
        "[N;H0;X3;!$(N=*)]",           # tertiary amines
        "[n;H0;X2]",                   # hetero aromatic N (pyridine)
        "N=C(N)N",                     # guanidine
        "C(=N)N",                      # amidine
        "[C-]",                        # carbanions
    ]
    CHIRAL_DICT = {'CHI_UNSPECIFIED':0, 'CHI_TETRAHEDRAL_CW':1, 'CHI_TETRAHEDRAL_CCW':-1}
    ACIDIC_PATTERNS = [Chem.MolFromSmarts(p) for p in ACIDIC_SMARTS]
    BASIC_PATTERNS  = [Chem.MolFromSmarts(p) for p in BASIC_SMARTS]
    MASKING_VALUE = -999999.0
    
    def __init__(self,
                 smi,
                 hybridization_list=['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP2D', 'SP3D', 'SP3D2', 'OTHER'],
                 chiraltypes=['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'],
                 bonds_list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
                 get_3d=False,
                 AddHs = False,
                 ):
        
        if isinstance(smi, str):
            self.mol = Chem.MolFromSmiles(smi)
            self.smi = smi
        else:
            self.mol=smi
            self.smi = Chem.MolToSmiles(smi)
            
        self.hybridization_list = hybridization_list
        self.chiraltypes = chiraltypes
        self.bonds_list = bonds_list
            
        self.graph = self.mol_to_nx(self.mol)
        if AddHs:
            self.mol = Chem.AddHs(self.mol)
            
        if self.mol is None:
            raise ValueError(f"Invalid SMILES string: {smi}")
        Chem.AssignStereochemistry(self.mol, force=True, cleanIt=True)

        # 3D coordinates (optional)
        if get_3d:
            try:
                AllChem.EmbedMolecule(self.mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(self.mol)
                self.has_3d = True
            except:
                self.has_3d = False
        self.logp_mr_contrib()
        self.estate_indices()
        
        self.df = pd.read_csv(f'{project_root}/metadata/Atom_properties_webelements.csv')
        self.df = self.df.to_dict()
        self.df['Atom'] = {v:k for k, v in self.df['Atom'].items()}
            
    def logp_mr_contrib(self):
    
        crippen = rdMolDescriptors._CalcCrippenContribs(self.mol)
        crippen_logp = [c[0] for c in crippen]
        crippen_mr = [c[1] for c in crippen]
        
        hydrophobic_centers = np.array([float(v > 0.0) for v in crippen_logp])
        sa_list = list(rdMolDescriptors._CalcLabuteASAContribs(self.mol)[0])
        sa_logp = [s * lp for s,lp in zip(sa_list, crippen_logp)]
        self.logp_mr = {'logP':crippen_logp, 'MR':crippen_mr, 'hypb_cntr':hydrophobic_centers, 'at_sasa':sa_list, 'sa_logp':sa_logp}
        
        return self.logp_mr
    
    def estate_indices(self):
        self.estates = {'estates':list(EState.EStateIndices(self.mol))}
        return self.estates
    
    def atom_tpsa(self):
        tp_list = list(rdMolDescriptors._CalcTPSAContribs(self.mol))
        return {'at_tpsa':tp_list}
    
    def donor_or_acceptor(self):
    
        fdef = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef)
        feats = factory.GetFeaturesForMol(self.mol)
        donor_atoms = set()
        acceptor_atoms = set()
        results = np.zeros((self.mol.GetNumAtoms(), 2), dtype=np.bool_)
        for ft in feats:
            ftype = ft.GetType()
            if ftype == "SingleAtomDonor":
                donor_atoms.update(ft.GetAtomIds())

            elif ftype == "SingleAtomAcceptor":
                acceptor_atoms.update(ft.GetAtomIds())
                
        donor_atoms = np.array(sorted(list(donor_atoms)), dtype=int)
        acceptor_atoms = np.array(sorted(list(acceptor_atoms)), dtype=int)
        results[donor_atoms, 0] = True
        results[acceptor_atoms, 1] = True
        
        return {'donor_atoms':donor_atoms, 'acceptor_atoms':acceptor_atoms, 'DA_flag':results}
    
    def detect_ionizable_atoms(self):
        n = self.mol.GetNumAtoms()

        acidic_set = set()
        basic_set = set()

        # acidic matches
        for patt in self.ACIDIC_PATTERNS:
            for match in self.mol.GetSubstructMatches(patt):
                acidic_set.update(match)

        # basic matches
        for patt in self.BASIC_PATTERNS:
            for match in self.mol.GetSubstructMatches(patt):
                basic_set.update(match)

        # convert to lists of 0/1 flags
        acidic = [i in acidic_set for i in range(n)]
        basic  = [i in basic_set for i in range(n)]

        return {"in_acidic_grp": acidic, "in_basic_grp": basic}
    
    def get_element_properties(self):
    
        df = self.df.copy()
        atom_properties = {}
        all_keys = ['symbol', 'atomic_num', 'degree', 'total_degree', 'formal_charge',
                    'total_num_Hs', 'explicit_valence', 'implicit_valence', 'total_valence',
                    'num_explicit_Hs', 'num_implicit_Hs', 'mass', 'num_pi_electrons',
                    'num_radical_electrons', 'GasteigerCharge', 'GasteigerHCharge',
                    'block', 'sh_struct', 'block_vec', 'VE', 'molar_vol', 'T_cond',
                    # 'RI',
                    'EA',  'IE1', 'IE2', 'IE3', 'At_rad', 'Cov_rad', 'SB_CR',
                    'DB_CR', 'TB_CR', 'vdW_rad', 'E_pauli', 'E_sand', 'E_alrd',
                    'E_muliken', 'polarizability', 'MP', 'BP', 'Etlp_f', 'Etlp_v', 'valence_contrib',
                    'Etlp_a', 'is_inring', 'is_aromatic', 'nring_mmbr', 'chirality','hybridization']
        for key in all_keys:
            atom_properties[key] = [None] * self.mol.GetNumAtoms()
        
        self.more_details = defaultdict(list)
        Chem.rdPartialCharges.ComputeGasteigerCharges(self.mol)
        for i, atom in enumerate(self.mol.GetAtoms()):
            symbol = atom.GetSymbol()
            atom_properties['symbol'][i] = symbol
            
            atom_properties['atomic_num'][i] = float(atom.GetAtomicNum())
            atom_properties['degree'][i] = float(atom.GetDegree())
            atom_properties['total_degree'][i] = float(atom.GetTotalDegree())
            atom_properties['formal_charge'][i] = float(atom.GetFormalCharge())
            atom_properties['total_num_Hs'][i] = float(atom.GetTotalNumHs())
            atom_properties['explicit_valence'][i] = float(atom.GetExplicitValence())
            atom_properties['implicit_valence'][i] = float(atom.GetImplicitValence())
            atom_properties['total_valence'][i] = float(atom.GetTotalValence())
            atom_properties['num_explicit_Hs'][i] = float(atom.GetNumExplicitHs())
            atom_properties['num_implicit_Hs'][i] = float(atom.GetNumImplicitHs())
            atom_properties['mass'][i] = float(atom.GetMass())
            atom_properties['num_pi_electrons'][i] = float(Chem.rdchem.GetNumPiElectrons(atom))
            atom_properties['num_radical_electrons'][i] = float(atom.GetNumRadicalElectrons())
            atom_properties['GasteigerCharge'][i] = float(atom.GetDoubleProp('_GasteigerCharge'))
            atom_properties['GasteigerHCharge'][i] = float(atom.GetDoubleProp('_GasteigerHCharge'))
            
            atom_properties['block'][i] = str(df['Block'][df['Atom'][symbol]])
            atom_properties['sh_struct'][i] = str(df['Shell structure'][df['Atom'][symbol]])
            atom_properties['block_vec'][i] = self.orbit_vector(block=str(df['Block'][df['Atom'][symbol]]), s_struct = str(df['Shell structure'][df['Atom'][symbol]]))
            
            atom_properties['VE'][i] = float(df['VE'][df['Atom'][symbol]])
            atom_properties['molar_vol'][i] = float(df['Molar Volume (cm^3)'][df['Atom'][symbol]])
            atom_properties['T_cond'][i] = float(df['Thermal Conductivity (W m^-1 K^-1)'][df['Atom'][symbol]])
            # atom_properties['RI'].append(float(df['Refractive Index'][df['Atom'][symbol]]))
            atom_properties['EA'][i] = float(df['E - Affinity (kJ/mol)'][df['Atom'][symbol]])
            atom_properties['IE1'][i] = float(df['IE - 1 (kJ/mol)'][df['Atom'][symbol]])
            val = df['IE - 2 (kJ/mol)'][df['Atom'][symbol]]
            atom_properties['IE2'][i] = float(val) if pd.notna(val)  else self.MASKING_VALUE
            val = df['IE - 3 (kJ/mol)'][df['Atom'][symbol]]
            atom_properties['IE3'][i] = float(val) if pd.notna(val)  else self.MASKING_VALUE
            atom_properties['At_rad'][i] = float(df['Atomic radius (pm)'][df['Atom'][symbol]])
            atom_properties['Cov_rad'][i] = float(df['Covalent radius (CR)'][df['Atom'][symbol]])
            atom_properties['SB_CR'][i] = float(df['Single Bond CR'][df['Atom'][symbol]])
            val = df['Double Bond CR'][df['Atom'][symbol]]
            atom_properties['DB_CR'][i] = float(val) if pd.notna(val)  else self.MASKING_VALUE
            val = df['Triple Bond CR'][df['Atom'][symbol]]
            atom_properties['TB_CR'][i] = float(val) if pd.notna(val)  else self.MASKING_VALUE
            atom_properties['vdW_rad'][i] = float(df['vdW radius (ppm)'][df['Atom'][symbol]])
            atom_properties['E_pauli'][i] = float(df['E - Pauling'][df['Atom'][symbol]])
            atom_properties['E_sand'][i] = float(df['E - Sanderson'][df['Atom'][symbol]])
            atom_properties['E_alrd'][i] = float(df['E - Allred'][df['Atom'][symbol]])
            val = df['E - Muliken'][df['Atom'][symbol]]
            atom_properties['E_muliken'][i] = float(val) if pd.notna(val)  else self.MASKING_VALUE
            atom_properties['polarizability'][i] = float(df['Polarizability'][df['Atom'][symbol]])
            atom_properties['MP'][i] = float(df['Melting Point (K)'][df['Atom'][symbol]])
            atom_properties['BP'][i] = float(df['Boiling Point (K)'][df['Atom'][symbol]])
            atom_properties['Etlp_f'][i] = float(df['Enthalpy fusion (kJ/mol)'][df['Atom'][symbol]])
            atom_properties['Etlp_v'][i] = float(df['Enthalpy vapour (kJ/mol)'][df['Atom'][symbol]])
            atom_properties['Etlp_a'][i] = float(df['Enthalpy atomisation (kJ/mol)'][df['Atom'][symbol]])
            
            atom_properties['is_inring'][i] = atom.IsInRing()
            atom_properties['is_aromatic'][i] = atom.GetIsAromatic()
            atom_properties['nring_mmbr'][i] = [atom.IsInRingSize(i) for i in range(3,9)]
            atom_properties['chirality'][i] = self.get_chirality(atom.GetChiralTag())
            atom_properties['hybridization'][i] = self.get_hybridization(atom.GetHybridization())
            atom_properties['valence_contrib'][i] = self.atom_valence_contrib(atom)
            
        return dict(atom_properties)
    
    def mol_to_nx(self, mol):
        G = nx.Graph()

        # Add atoms as nodes
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            G.add_node(idx)

        # Add bonds as edges
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            G.add_edge(a1, a2)

        return G
    
    def compute_all_centralities(self):
        
        n_atoms = self.mol.GetNumAtoms()
        
        if n_atoms <= 2:
            return {
                "atom_idx": np.arange(n_atoms),
                "eigenvector": [1e-5]*n_atoms,
                "betweenness": [1e-5]*n_atoms,
                "closeness": [1e-5]*n_atoms,
                "clustering": [1e-5]*n_atoms
            }

        eig = nx.eigenvector_centrality_numpy(self.graph)
        bet = nx.betweenness_centrality(self.graph)
        clo = nx.closeness_centrality(self.graph)
        clu = nx.clustering(self.graph)

        results = {
            "atom_idx": list(eig.keys())[:n_atoms],
            "eigenvector": list(eig.values())[:n_atoms],
            "betweenness": list(bet.values())[:n_atoms],
            "closeness": list(clo.values())[:n_atoms],
            "clustering": list(clu.values())[:n_atoms]
        }
        return results
    
    def katz_matrix(self, G, beta=0.01):
        A = nx.to_numpy_array(G)
        I = np.eye(A.shape[0])
        K = np.linalg.inv(I - beta*A) - I
        return K

    
    def compute_graph_indices(self):
        # adjacency matrix
        A = nx.to_numpy_array(self.graph)
        
        # degree vector
        deg = A.sum(axis=1)
        
        # intersection of neighbors for all pairs: A^2 > 0 means common neighbors
        common = A @ A  # each (i,j) = |Γ(i) ∩ Γ(j)|

        # union = deg(i) + deg(j) - |intersection|
        deg_i = deg.reshape(-1, 1)
        deg_j = deg.reshape(1, -1)
        union = deg_i + deg_j - common

        # Jaccard
        jaccard = np.divide(common, union, out=np.zeros_like(common), where=union != 0)

        # Salton (Cosine similarity)
        salton = np.divide(common, np.sqrt(deg_i * deg_j), out=np.zeros_like(common), where=(deg_i * deg_j) != 0)
        
        # Sorensen
        sorensen = np.divide(2 * common, deg_i + deg_j, out=np.zeros_like(common), where=(deg_i + deg_j) != 0)

        # Resource Allocation (RA)
        deg_inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg != 0)
        RA = A @ np.diag(deg_inv) @ A

        # Adamic–Adar (AA)
        log_deg_inv = np.divide(1.0, np.log(deg), out=np.zeros_like(deg), where=(deg > 1))
        AA = A @ np.diag(log_deg_inv) @ A

        # zero diagonals
        for M in [jaccard, salton, sorensen, RA, AA]:
            np.fill_diagonal(M, 0)

        return {
            "Jaccard": jaccard.sum(axis=1),
            "Salton": salton.sum(axis=1),
            "Sorensen": sorensen.sum(axis=1),
            "RA": RA.sum(axis=1),
            "AA": AA.sum(axis=1),
            "Katz": self.katz_matrix(self.graph).sum(axis=1)
        }
    
    def orbital_feature_vector(self, n: int, l: int) -> Tuple[float, ...]:
        """
        Return a physics-based feature vector for an atomic orbital:
        (n, l, 2l+1, n-l-1, l, sqrt(l(l+1))).

        Parameters
        ----------
        n : int
            Principal quantum number (1, 2, 3, ...).
        l : int
            Azimuthal quantum number (0=s, 1=p, 2=d, 3=f, ...).

        Returns
        -------
        tuple of floats
            Feature vector: (n, l, m, lh).
        """
        m = 2 * l + 1   
        L_over_hbar = float(math.sqrt(l * (l + 1)))  # |L| / ħ
        n, l, m, lh = (float(n), float(l), float(m), L_over_hbar)

        return n, l, m, lh

    def orbital_feature_from_symbol(self, n: int, symbol: str) -> Tuple[float, ...]:
        """
        Convenience wrapper: n and spectroscopic symbol ('s','p','d','f').
        """
        SYMBOL_TO_L = { "s": 0, "p": 1, "d": 2, "f": 3}
        l = SYMBOL_TO_L[symbol.lower()]
        return self.orbital_feature_vector(n, l)
    
    def orbit_vector(self, block, s_struct):
        
        config = s_struct.split('.')
        v = self.orbital_feature_from_symbol(n=len(config) if block != 'd' else len(config) - 1,
                                             symbol=block)
        if block=='s':
            n_elec = int(config[-1])
        elif block == 'p':
            n_elec = int(config[-1]) - 2
        elif block == 'd':
            n_elec = int(config[-2]) - 8 
        return [*v, float(n_elec)]

    def get_chirality(self, chiral_tag):
        
        if chiral_tag == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            return self.MASKING_VALUE
        elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            return 1
        elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            return -1
        else:
            return 0
        
    def get_hybridization(self, hybridization_tag):
        
        tag = str(hybridization_tag)
        if tag == 'UNSPECIFIED':
            v = list_to_onehot(str(hybridization_tag), self.hybridization_list, val=self.MASKING_VALUE)
        else:
            v = list_to_onehot(str(hybridization_tag), self.hybridization_list)
        
        return np.array(v, dtype=np.bool_)
    
    def atom_valence_contrib(self, atom):
        
        bond_value_dict = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 4}
        val = [0.0, 0.0, 0.0, 0.0, 0.0]

        for bond in atom.GetBonds():
            btype = str(bond.GetBondType())
            idx = bond_value_dict.get(btype, 0)
            contrib = bond.GetValenceContrib(atom)
            val[idx] += contrib

        implicit_H = atom.GetNumImplicitHs()
        if implicit_H > 0:
            val[bond_value_dict.get('SINGLE', 0)] += implicit_H * 1.0

        return val
    
    
    def get_bond_features(self):
        """
        Extracts all relevant bond features for every bond in the molecule.
        Returns a list of feature dictionaries, one per bond.
        """
        all_bonds = ['SINGLE', 'AROMATIC', 'DOUBLE', 'TRIPLE']
        stereo_list = ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOATROPCW', 'STEREOATROPCCW']
        bond_features = {}
        all_keys = ['begin_atom_idx', 'end_atom_idx', 'begin_atom_symbol', 'end_atom_symbol', 'begin_atom_num', 'end_atom_num',
                'bond_type', 'bond_type_str', 'bond_order', 'is_in_ring', 'is_conjugated', 'is_aromatic', 'stereo', 'stereo_vec',
                'stereo_atoms', 'bond_dir', 'bond_is_unsaturated', 'nring_mmbr', 'begin_hyb', 'end_hyb', 'polarity', 'sa_bond',
                'EA_init', 'EA_end', 'E_pauli_init', 'E_pauli_end', 'bond_centrality']
        for key in all_keys:
            bond_features[key] = [None] * self.mol.GetNumBonds()

        bond_centrality = nx.edge_betweenness_centrality(self.graph, normalized=True)
        for i, bond in enumerate(self.mol.GetBonds()):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            init_idx = a1.GetIdx()
            end_idx = a2.GetIdx()
            sym1 = a1.GetSymbol()
            sym2 = a2.GetSymbol()

            bond_features['begin_atom_idx'][i] = init_idx
            bond_features['end_atom_idx'][i] = end_idx
            bond_features['begin_atom_symbol'][i] = sym1
            bond_features['end_atom_symbol'][i] = sym2
            bond_features['begin_atom_num'][i] = a1.GetAtomicNum()
            bond_features['end_atom_num'][i] = a2.GetAtomicNum()
            bond_features['bond_type_str'][i] = str(bond.GetBondType())
            bond_features['bond_type'][i] = np.array(list_to_onehot_oov(bond_features['bond_type_str'][i], all_bonds))
            bond_features['bond_order'][i] = bond.GetBondTypeAsDouble()
            bond_features['is_in_ring'][i] = bond.IsInRing()
            bond_features['is_conjugated'][i] = bond.GetIsConjugated()
            bond_features['is_aromatic'][i] = bond.GetIsAromatic()
            bond_features['stereo_vec'][i] = list_to_onehot_oov(str(bond.GetStereo()), stereo_list)
            bond_features['stereo'][i] = str(bond.GetStereo())
            bond_features['stereo_atoms'][i] = list(bond.GetStereoAtoms())
            bond_features['bond_dir'][i] = str(bond.GetBondDir())
            bond_features['bond_is_unsaturated'][i] = (bond.GetIsConjugated() or bond.GetBondType() in [Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC])
            bond_features['nring_mmbr'][i] = [bond.IsInRingSize(j) for j in range(3,9)]
            bond_features['begin_hyb'][i] = self.get_hybridization(a1.GetHybridization())
            bond_features['end_hyb'][i] = self.get_hybridization(a2.GetHybridization())
            bond_features['polarity'][i] = abs(self.estates['estates'][init_idx] - self.estates['estates'][end_idx])
            bond_features['sa_bond'][i] = abs(self.logp_mr['at_sasa'][init_idx] - self.logp_mr['at_sasa'][end_idx])
            bond_features['EA_init'][i] = float(self.df['E - Affinity (kJ/mol)'][self.df['Atom'][sym1]])
            bond_features['EA_end'][i] = float(self.df['E - Affinity (kJ/mol)'][self.df['Atom'][sym2]])
            bond_features['E_pauli_init'][i] = float(self.df['E - Pauling'][self.df['Atom'][sym1]])
            bond_features['E_pauli_end'][i] = float(self.df['E - Pauling'][self.df['Atom'][sym2]])
            bond_features['bond_centrality'][i] = float(bond_centrality.get((init_idx, end_idx), 0))
            
        bond_features['bond_matrix'] = np.array([bond_features['begin_atom_idx'], bond_features['end_atom_idx']])

        for k,v in bond_features.items():
            if k not in ['stereo', 'stereo_atoms']:
                bond_features[k] = np.array(v)
                
        return dict(bond_features)
    
    def get_all_molecule_properties(self):
        
        atom_properties = self.get_element_properties()
        atom_properties.update(self.logp_mr)
        atom_properties.update(self.estates)
        atom_properties.update(self.atom_tpsa())
        atom_properties.update(self.donor_or_acceptor())
        atom_properties.update(self.detect_ionizable_atoms())
        atom_properties.update(self.compute_all_centralities())
        atom_properties.update(self.compute_graph_indices())
        
        for k,v in atom_properties.items():
            v = np.array(v)
            atom_properties[k] = v.reshape(-1,1) if v.ndim==1 else v
            
        bond_properties = self.get_bond_features()
            
        return dict(atom_properties), bond_properties
    #----------------------------------------------------------------------------------------------------------------------------
    #============================================================================================================================
