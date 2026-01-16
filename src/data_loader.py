# src/data_loader.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import mplhep

# ---- Existing DataLoader adapted for convenience ----
class DataLoader:
    """Handles loading ROOT files and accessing their trees via uproot."""
    
    def __init__(self, file_path, tree_name):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        self.file = uproot.open(file_path)
        if tree_name not in self.file:
            raise ValueError(f"The tree {tree_name} does not exist in {file_path}.")

        self.tree = self.file[tree_name]

    def get_tree(self):
        return self.tree

    def get_file(self):
        return self.file

# ---- File and tree configuration ----
BASE_DIR = Path(__file__).resolve().parent.parent

EXP_FILE = BASE_DIR / "data" / "data_Xib2XicPi_2016_MU.addVar.wMVA.root"
MC_FILE  = BASE_DIR / "data" / "MC_Xib2XicPi_2016MC_MU.pid.addVar.wMVA.root"

EXP_TREE = "mytree;13"
MC_TREE  = "mytree;3"


# ---- Data class for impact parameters ----
@dataclass
class ImpactParameterDataset:
    # Variable names exactly as requested
    var_names: List[str] = field(default_factory=lambda: [
        # --- te z kartki ---
        "Xb_IP_OWNPV",
        "Xc_IP_OWNPV",
        "Xc_P",
        "Xb_P",
        "k_P",  
        "p_P",
        # to mi mowil ostatnio prowadzacy
        "p_IPCHI2_OWNPV",
        "pi_PT",
        "pi_P",
        "Xb_M",
        "Xc_M",

        
        # --- dodatkowe/ polecone przez chata --- 
        "Xb_IPCHI2_OWNPV",   # Normalized displacement of the Xb particle
        "Xc_IPCHI2_OWNPV",   # Normalized displacement of the Xc particle
        "k_IPCHI2_OWNPV",    # Helps filter kaons originating from the PV (noise)
        "p_IPCHI2_OWNPV",    # Helps filter protons originating from the PV (noise)

        #         "Xb_FDCHI2_OWNPV",   # Significance of how far Xb traveled before decaying
        "Xb_DIRA_OWNPV",     # Angle between momentum and flight path (Signal should be ~1)
        "Xb_ENDVERTEX_CHI2"  # Quality of vertex fit; high values usually indicate noise
    ])

    # Internal storage
    exp_loader: DataLoader = None
    mc_loader: DataLoader = None

    def load_experimental(self):
        """Load experimental data ROOT tree."""
        self.exp_loader = DataLoader(EXP_FILE, EXP_TREE)
        print(f"Loaded experimental tree: {self.exp_loader.get_tree()}")

    def load_mc(self):
        """Load Monte Carlo data ROOT tree."""
        self.mc_loader = DataLoader(MC_FILE, MC_TREE)
        print(f"Loaded MC tree: {self.mc_loader.get_tree()}")

    def to_dataframe(self, dataset: str = "exp") -> pd.DataFrame:
        """Return a pandas DataFrame containing the selected variables."""
        if dataset == "exp":
            if self.exp_loader is None:
                raise ValueError("Experimental data not loaded yet.")
            tree = self.exp_loader.get_tree()
        elif dataset == "mc":
            if self.mc_loader is None:
                raise ValueError("MC data not loaded yet.")
            tree = self.mc_loader.get_tree()
        else:
            raise ValueError("dataset must be 'exp' or 'mc'")

        # Convert ROOT tree to DataFrame using the selected variables
        data_dict = {var: tree[var].array() for var in self.var_names}
        return pd.DataFrame(data_dict)