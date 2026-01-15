import uproot
from pathlib import Path

# Project root = parent of src/
BASE_DIR = Path(__file__).resolve().parent.parent

EXP_FILE = BASE_DIR / "data" / "data_Xib2XicPi_2016_MU.addVar.wMVA.root"
MC_FILE  = BASE_DIR / "data" / "MC_Xib2XicPi_2016MC_MU.pid.addVar.wMVA.root"

EXP_TREE = "mytree;13"
MC_TREE = "mytree;3"


class DataLoader:
    # This class handles loading of ROOT files and accessing their trees.
    def __init__(self, file_path, tree_name):
        # ROOT version
        # if not os.path.exists(file_path):
        #     raise FileNotFoundError(f"The file {file_path} does not exist.")

        # self.file = ROOT.TFile.Open(file_path)
        # if not self.file or self.file.IsZombie():
        #     raise IOError(f"Could not open the file {file_path}.")

        # self.tree = self.file.Get(tree_name)
        # if not self.tree:
        #     raise ValueError(
        #         f"The tree {tree_name} does not exist in the file {file_path}.")

        # Uproot version

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        self.file = uproot.open(file_path)

        if tree_name not in self.file:
            raise ValueError(
                f"The tree {tree_name} does not exist in the file {file_path}."
            )

        self.tree = self.file[tree_name]
    def get_tree(self):
        return self.tree

    def get_file(self):
        return self.file


if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader(EXP_FILE, EXP_TREE)
    print(f"Successfully loaded tree: {data_loader.tree.GetName()}")
