import argparse
from pathlib import Path

import pandas as pd

import src.visualisation as vis
from src.evaluate import compute_roc, compute_significance_curve
from src.logging_utils import format_classification_metrics, print_variable_table, summarize_variable_stats
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
    print(format_classification_metrics(result.metrics))
    print(
        f"exp kept @ {args.threshold:.2f}: {(exp_df['BDT'] >= args.threshold).mean():.3f}"
    )

    if args.plot:
        pass

    data_with_cuts_df = exp_df.query("BDT > @args.threshold")
    if args.plot:
        summary = {}
        for var in result.denoiser.features or []:
            zoom = (5, 95) if var == "Xb_IPCHI2_OWNPV" else (0.5, 99.5)
            vis.plot_var_combined(
                var,
                exp_df,
                mc_df,
                data_with_cuts_df,
                output_dir=args.plot_dir,
                zoom_percentiles=zoom,
            )
            summary[var] = summarize_variable_stats(var, mc_df, exp_df, data_with_cuts_df)

        y_true = pd.concat([
            pd.Series(1, index=mc_df.index),
            pd.Series(0, index=bkg_df.index),
        ])
        y_score = pd.concat([mc_df["BDT"], bkg_df["BDT"]])
        roc = compute_roc(y_true.to_numpy(), y_score.to_numpy())
        vis.plot_roc_curve(roc.fpr, roc.tpr, roc.auc_score, output_dir=args.plot_dir)

        significance_df = compute_significance_curve(
            y_true=y_true.to_numpy(),
            y_score=y_score.to_numpy(),
            n_sig=float(len(mc_df)),
            n_bkg=float(len(bkg_df)),
        )
        vis.plot_significance_curve(
            significance_df["threshold"].to_numpy(),
            significance_df["significance"].to_numpy(),
            output_dir=args.plot_dir,
        )

        if args.plot_dir:
            print(f"Saved plots to {Path(args.plot_dir).resolve()}")

        if summary:
            print_variable_table(summary)

    print(f"\nSaved BDT model to {Path(args.model_dir).resolve()}")
        


if __name__ == "__main__":
    main()
