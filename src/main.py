import argparse
from pathlib import Path

import pandas as pd

import src.visualisation as vis
from src.train import train_bdt_denoiser

def main(argv=None):
    parser = argparse.ArgumentParser(description="HEP denoiser entry point")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print variable count table")
    parser.add_argument("-p", "--plot", action="store_true", help="Show all plots")
    parser.add_argument("--plot-dir", type=str, default=None, help="Directory to save plots")
    parser.add_argument("--threshold", type=float, default=0.95, help="BDT threshold for filtering")
    parser.add_argument(
        "--background-window",
        type=str,
        default="5700,5900",
        help="Background mass window to exclude (low,high)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained BDT model",
    )
    args = parser.parse_args(argv)

    low, high = (float(x.strip()) for x in args.background_window.split(","))
    result = train_bdt_denoiser(
        background_window=(low, high),
        threshold=args.threshold,
        model_output_dir=Path(args.model_dir),
    )

    exp_df = result.exp_df.copy()
    mc_df = result.mc_df.copy()
    bkg_df = result.background_df.copy()

    if args.verbose:
        # Table Header with column formatting
        print(f"\n{'variable':<25} {'MC Count':<12} {'EXP Count':<12}")
        print("-" * 55)

        for var in result.denoiser.features or []:
            # Get counts for each variable in both datasets
            # If variable is missing, count is 0
            mc_count = mc_df[var].count() if var in mc_df.columns else 0
            exp_count = exp_df[var].count() if var in exp_df.columns else 0
            
            print(f"{var:<25} {mc_count:<12} {exp_count:<12}")

    exp_df["BDT"] = result.prob_exp
    mc_df["BDT"] = result.prob_mc
    bkg_df["BDT"] = result.prob_bkg

    print("\nBDT classification summary (MC vs background)")
    print("-------------------------------------------")
    print(
        f"precision={result.metrics['precision']:.3f}, "
        f"recall={result.metrics['recall']:.3f}, "
        f"accuracy={result.metrics['accuracy']:.3f}, "
        f"f1={result.metrics['f1']:.3f}"
    )
    print(
        f"tp={int(result.metrics['tp'])}, fp={int(result.metrics['fp'])}, "
        f"tn={int(result.metrics['tn'])}, fn={int(result.metrics['fn'])}"
    )
    print(
        f"exp kept @ {args.threshold:.2f}: {(exp_df['BDT'] >= args.threshold).mean():.3f}"
    )

    if args.plot:
        vis.plot_var_comparison("BDT", exp_df, mc_df, output_dir=args.plot_dir)
        vis.plot_var_comparison("BDT", bkg_df, mc_df, output_dir=args.plot_dir)
        for var in result.denoiser.features or []:
            vis.plot_var_comparison(var, exp_df, mc_df, output_dir=args.plot_dir)

    data_with_cuts_df = exp_df.query("BDT > @args.threshold")
    if args.plot:
        for var in result.denoiser.features or []:
            vis.plot_var_comparison(
                var,
                exp_df,
                mc_df,
                output_dir=args.plot_dir,
                filtered_df=data_with_cuts_df,
            )
        if args.plot_dir:
            print(f"Saved plots to {Path(args.plot_dir).resolve()}")

    print(f"\nSaved BDT model to {Path(args.model_dir).resolve()}")
        


if __name__ == "__main__":
    main()
