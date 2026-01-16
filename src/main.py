import argparse
import src.data_loader as dl
import pandas as pd
import src.visualisation as vis
import numpy as np
import matplotlib.pyplot as plt
import mplhep
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
    if args.plot:
        vis.plot_var_comparison(curr_var, exp_df, mc_df)
    bkg_df = exp_df.query(f"~(5700 < {curr_var} < 5900)")
    if args.plot:
        vis.plot_var_comparison(curr_var, bkg_df, mc_df)

    # define classifier
    bdt = XGBClassifier(n_estimators=20)
    # prepare training data
    bkg_df = bkg_df.copy()
    bkg_df['category'] = 0  # Use 0 for background
    mc_df['category'] = 1  # Use 1 for signal
    # Now merge the data together
    training_data = pd.concat([bkg_df, mc_df], copy=True, ignore_index=True)
    bdt.fit(training_data[[curr_var]], training_data['category'])
    # We can now use slicing to select column 1 in the array from for all rows
    probabilities = bdt.predict_proba(exp_df[[curr_var]])[:,1]
    print(probabilities)

    mc_df['BDT'] = bdt.predict_proba(mc_df[[curr_var]])[:,1]
    bkg_df['BDT'] = bdt.predict_proba(bkg_df[[curr_var]])[:,1]
    exp_df['BDT'] = bdt.predict_proba(exp_df[[curr_var]])[:,1]
    training_data['BDT'] = bdt.predict_proba(training_data[[curr_var]])[:,1]
    if args.plot:
        vis.plot_var_comparison("BDT", exp_df, mc_df)
        vis.plot_var_comparison("BDT", bkg_df, mc_df)
    data_with_cuts_df = exp_df.query('BDT > 0.95')
    vis.plot_var_comparison(curr_var, data_with_cuts_df, mc_df)
        


if __name__ == "__main__":
    main()
