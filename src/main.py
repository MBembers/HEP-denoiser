import argparse
import src.data_loader as dl
import pandas as pd
import src.visualisation as vis

def main(argv=None):
    parser = argparse.ArgumentParser(description="HEP denoiser entry point")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print variable count table")
    args = parser.parse_args(argv)

    dataset = dl.ImpactParameterDataset()
    
    try:
        dataset.load_experimental()
        dataset.load_mc()
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Convert ROOT trees to DataFrames
    df_exp = dataset.to_dataframe("exp")
    df_mc = dataset.to_dataframe("mc")

    if args.verbose:
        # Table Header with column formatting
        print(f"\n{'variable':<25} {'MC Count':<12} {'EXP Count':<12}")
        print("-" * 55)

        for var in dataset.var_names:
            # Get counts for each variable in both datasets
            # If variable is missing, count is 0
            mc_count = df_mc[var].count() if var in df_mc.columns else 0
            exp_count = df_exp[var].count() if var in df_exp.columns else 0
            
            print(f"{var:<25} {mc_count:<12} {exp_count:<12}")


if __name__ == "__main__":
    main()
