import data_loader as dl
from matplotlib import pyplot as plt
import mplhep
import numpy as np
import ROOT


def main():
    # Load experimental data
    exp_data_loader = dl.DataLoader(dl.EXP_FILE, dl.EXP_TREE)
    exp_tree = exp_data_loader.get_tree()
    print(f"Loaded experimental tree: {exp_tree}")

    # Load Monte Carlo data
    mc_data_loader = dl.DataLoader(dl.MC_FILE, dl.MC_TREE)
    mc_tree = mc_data_loader.get_tree()
    print(f"Loaded MC tree: {mc_tree}")

    # Example: Plotting a histogram of a variable from the experimental data
    exp_Xb_M = exp_tree["Xb_M"].array()
    mc_Xb_M = mc_tree["Xb_M"].array()

    plt.style.use(mplhep.style.ATLAS)
    plt.hist(exp_Xb_M, bins=70, alpha=0.5,
             label="Experimental Data", color='blue', density=True)
    plt.hist(mc_Xb_M, bins=70, alpha=0.5, label="MC Data",
             color='orange', density=True)
    plt.xlabel("Xb_M")
    plt.ylabel("Density")
    plt.title("Histogram of Xb_M")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
