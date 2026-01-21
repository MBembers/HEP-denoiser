import argparse
import src.data_loader as dl
import pandas as pd
import src.visualisation as vis
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import src.utils as utils
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

def main(argv=None):
    parser = argparse.ArgumentParser(description="HEP denoiser entry point")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print variable count table")
    parser.add_argument("-p", "--plot", action="store_true", help="Show all plots")
    args = parser.parse_args(argv)

    dataset = dl.ImpactParameterDataset()
    
    try:
        dataset.load_experimental()
        dataset.load_mc()
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Convert ROOT trees to DataFrames
    exp_df = dataset.to_dataframe("exp")
    mc_df = dataset.to_dataframe("mc")

    if args.verbose:
        # Table Header with column formatting
        print(f"\n{'variable':<25} {'MC Count':<12} {'EXP Count':<12}")
        print("-" * 55)

        for var in dataset.var_names:
            # Get counts for each variable in both datasets
            # If variable is missing, count is 0
            mc_count = mc_df[var].count() if var in mc_df.columns else 0
            exp_count = exp_df[var].count() if var in exp_df.columns else 0
            
            print(f"{var:<25} {mc_count:<12} {exp_count:<12}")

    
    # try denoising Xb_M
    curr_var = "Xb_M"
    # Apply cut for Xb_M to define background region
    bkg_df = exp_df.query(f"~(5750 < {curr_var} < 5850)")

    if args.plot:
        vis.plot_var_comparison(curr_var, exp_df, mc_df)
    if args.plot:
        vis.plot_var_comparison(curr_var, bkg_df, mc_df)

    # utils.print_tree_variables(dl.EXP_FILE, dl.EXP_TREE)
    # define classifier
    bdt = XGBClassifier(n_estimators=20)
    # prepare training data
    bkg_df = bkg_df.copy()
    bkg_df['category'] = 0  # Use 0 for background
    mc_df['category'] = 1  # Use 1 for signal
    # Now merge the data together
    training_data = pd.concat([bkg_df, mc_df], copy=True, ignore_index=True)
    # Define columns to use for training
    training_columns = ["Xb_M", "Xb_IPCHI2_OWNPV", "Xb_DIRA_OWNPV", "Xb_ENDVERTEX_CHI2"]
    bdt.fit(training_data[training_columns], training_data['category'])
    # We can now use slicing to select column 1 in the array from for all rows
    # probabilities = bdt.predict_proba(exp_df[training_columns])[:,1]
    # print(probabilities)

    mc_df['BDT'] = bdt.predict_proba(mc_df[training_columns])[:,1]
    bkg_df['BDT'] = bdt.predict_proba(bkg_df[training_columns])[:,1]
    exp_df['BDT'] = bdt.predict_proba(exp_df[training_columns])[:,1]
    training_data['BDT'] = bdt.predict_proba(training_data[training_columns])[:,1]
    # filter by BDT score
    bdt_cut = 0.99
    bkg_filt_df = bkg_df.query(f"BDT < {1 - bdt_cut}")
    mc_filt_df = mc_df.query(f"BDT > {bdt_cut}")
    exp_filt_df = exp_df.query(f"BDT > {bdt_cut}")

    plt.figure(figsize=(6,6))
    vis.add_var_to_plot(curr_var, bkg_df, bins=50, color='blue', label='Background')
    vis.add_var_to_plot(curr_var, mc_df, bins=50, color='orange', label='MC')
    vis.add_var_to_plot(curr_var, exp_df, bins=50, color='green', label='Experimental')
    plt.legend()
    plt.savefig(f"{curr_var}_no_filter_comp.png")
    plt.close()

    plt.figure(figsize=(6,6))
    vis.add_var_to_plot(curr_var, mc_df, bins=50, color='orange', label='MC')
    vis.add_var_to_plot(curr_var, exp_filt_df, bins=50, color='green', label='Experimental')
    plt.legend()
    plt.savefig(f"{curr_var}_filtered_comp.png")
    plt.close()

    plt.figure(figsize=(6,6))
    vis.add_var_to_plot(curr_var, exp_filt_df , bins=50, color='blue', label='Filtered Experimental')
    vis.add_var_to_plot(curr_var, exp_df , bins=50, color='green', label='Experimental')
    plt.legend()
    plt.savefig(f"{curr_var}_exp_before_after.png")

    plt.figure(figsize=(6,6))
    vis.add_var_to_plot("BDT", bkg_df, bins=50, color='blue', label='Background')
    vis.add_var_to_plot("BDT", mc_df, bins=50, color='orange', label='MC')
    vis.add_var_to_plot("BDT", exp_df, bins=50, color='green', label='Experimental')
    plt.legend()
    plt.savefig("BDT_distribution.png")
    plt.close()


        


if __name__ == "__main__":
    main()
