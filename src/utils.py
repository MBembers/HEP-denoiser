# src/utils.py
from pathlib import Path
import uproot
# Change .dataloader to .data_loader
from .data_loader import DataLoader

# ---- ROOT files ----
BASE_DIR = Path(__file__).resolve().parent.parent
FILES = [
    BASE_DIR / "data" / "data_Xib2XicPi_2016_MU.addVar.wMVA.root",
    BASE_DIR / "data" / "MC_Xib2XicPi_2016MC_MU.pid.addVar.wMVA.root"
]

TREES = ["mytree"] 


def print_tree_variables(file_path, tree_name):
    """Print all variables/branches inside a ROOT tree."""
    loader = DataLoader(file_path, tree_name)
    tree = loader.get_tree()
    print(f"\nFile: {file_path}")
    print(f"Tree: {tree_name}")
    print("Branches / Variables:")
    for branch in tree.keys():
        print(f" - {branch}")


if __name__ == "__main__":
    for f in FILES:
        for t in TREES:
            try:
                print_tree_variables(f, t)
            except Exception as e:
                print(f"Could not access {t} in {f}: {e}")

