from __future__ import annotations

import argparse
from pathlib import Path

from .evaluate import benchmark_datasets, score_separation, sweep_retention_vs_ks
from .train import train_classifier_denoiser, train_denoiser
from .visualisation import (
    plot_correlation_heatmap,
    plot_feature_histograms,
    plot_feature_ks_improvement,
    plot_score_distributions,
    plot_retention_vs_ks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train denoiser and benchmark results")
    parser.add_argument(
        "--mode",
        choices=["classifier", "gaussian"],
        default="classifier",
        help="Denoiser type to run",
    )
    parser.add_argument("--save-filtered", type=str, default=None, help="Optional CSV output for filtered EXP data")
    parser.add_argument("--plot-dir", type=str, default=None, help="Optional directory to save plots")

    # Gaussian options
    parser.add_argument("--quantile", type=float, default=0.95, help="MC score quantile for threshold")
    parser.add_argument("--shrinkage", type=float, default=1e-3, help="Covariance shrinkage amount")

    # Classifier options
    parser.add_argument("--retain-quantile", type=float, default=0.95, help="MC probability retention quantile")
    parser.add_argument(
        "--target-retention",
        type=float,
        default=None,
        help="Target EXP retention rate (overrides retain-quantile)",
    )
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (e.g. cpu, cuda)")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional per-class subsample size")
    parser.add_argument(
        "--sweep-quantiles",
        type=str,
        default=None,
        help="Comma-separated EXP score quantiles for retention-vs-KS sweep",
    )
    args = parser.parse_args()

    if args.mode == "classifier":
        result = train_classifier_denoiser(
            retain_quantile=args.retain_quantile,
            target_retention=args.target_retention,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            device=args.device,
        )
        benchmark = benchmark_datasets(
            df_mc=result.df_mc,
            df_exp=result.df_exp,
            df_filtered=result.df_filtered,
            scores_mc=result.probs_mc,
            scores_exp=result.probs_exp,
            scores_filtered=result.probs_filtered,
            features=result.denoiser.features or [],
        )
        separation = score_separation(result.probs_mc, result.probs_exp)
        if args.target_retention is not None:
            threshold_label = f"Probability threshold (target retention {args.target_retention:.2f})"
        else:
            threshold_label = f"Probability threshold (retain {args.retain_quantile:.2f})"
    else:
        result = train_denoiser(quantile=args.quantile, shrinkage=args.shrinkage)
        benchmark = benchmark_datasets(
            df_mc=result.df_mc,
            df_exp=result.df_exp,
            df_filtered=result.df_filtered,
            scores_mc=result.scores_mc,
            scores_exp=result.scores_exp,
            scores_filtered=result.scores_filtered,
            features=result.denoiser.features or [],
        )
        separation = score_separation(result.scores_mc, result.scores_exp)
        threshold_label = f"Distance threshold (quantile {args.quantile:.2f})"

    print("\nDenoiser summary")
    print("================")
    print(f"{threshold_label}: {result.threshold:.4f}")
    print(f"Retention rate: {benchmark.retention_rate:.3f}")

    print("\nScore benchmark (bootstrap mean CI)")
    print("---------------------------------")
    for label, stats in benchmark.score_summary.items():
        print(
            f"{label:>8}: mean={stats['estimate']:.4f}, "
            f"CI=[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]"
        )

    print("\nScore separation (MC vs EXP)")
    print("-----------------------------")
    print(
        f"mc_mean={separation['mc_mean']:.4f}, exp_mean={separation['exp_mean']:.4f}, "
        f"delta={separation['mean_delta']:.4f}"
    )

    print("\nFeature KS summary (lower is closer to MC)")
    print("-----------------------------------------")
    for feat, stats in benchmark.per_feature.items():
        print(
            f"{feat:>20}: ks(exp,mc)={stats['ks_exp_vs_mc']:.4f}, "
            f"ks(filt,mc)={stats['ks_filt_vs_mc']:.4f}"
        )

    if args.plot_dir is not None:
        plot_correlation_heatmap(result.df_mc, title="MC correlation", output_dir=args.plot_dir, filename="corr_mc.png")
        plot_correlation_heatmap(result.df_exp, title="EXP correlation", output_dir=args.plot_dir, filename="corr_exp.png")
        plot_correlation_heatmap(
            result.df_filtered,
            title="Filtered EXP correlation",
            output_dir=args.plot_dir,
            filename="corr_filtered.png",
        )
        plot_score_distributions(
            scores_mc=result.probs_mc if args.mode == "classifier" else result.scores_mc,
            scores_exp=result.probs_exp if args.mode == "classifier" else result.scores_exp,
            scores_filtered=result.probs_filtered if args.mode == "classifier" else result.scores_filtered,
            title="Score distributions",
            output_dir=args.plot_dir,
        )
        plot_feature_ks_improvement(benchmark.per_feature, output_dir=args.plot_dir)
        plot_feature_histograms(
            result.df_mc,
            result.df_exp,
            result.df_filtered,
            features=result.denoiser.features or [],
            output_dir=args.plot_dir,
        )
        if args.sweep_quantiles and args.mode == "classifier":
            quantiles = [float(q.strip()) for q in args.sweep_quantiles.split(",") if q.strip()]
            sweep_df = sweep_retention_vs_ks(
                denoiser=result.denoiser,
                df_mc=result.df_mc,
                df_exp=result.df_exp,
                probs_exp=result.probs_exp,
                features=result.denoiser.features or [],
                quantiles=quantiles,
            )
            if not sweep_df.empty:
                plot_retention_vs_ks(sweep_df, output_dir=args.plot_dir)
                sweep_df.to_csv(Path(args.plot_dir) / "retention_vs_ks.csv", index=False)
        print(f"\nSaved plots to {args.plot_dir}")

    if args.save_filtered:
        result.df_filtered.to_csv(args.save_filtered, index=False)
        print(f"\nSaved filtered data to {args.save_filtered}")


if __name__ == "__main__":
    main()
