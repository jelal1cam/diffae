import os
import argparse
import pandas as pd

def compute_pareto_front(data, metrics):
    """
    Identify Pareto-optimal rows in `data` based on `metrics` columns.
    Returns a boolean mask of Pareto-optimal points.
    """
    pts = data[metrics].values
    is_optimal = []
    for i, pt in enumerate(pts):
        dominated = False
        for j, other in enumerate(pts):
            if j == i:
                continue
            # other dominates if other <= pt for all metrics and < for at least one
            if all(other <= pt) and any(other < pt):
                dominated = True
                break
        is_optimal.append(not dominated)
    return pd.Series(is_optimal, index=data.index)


def main():
    parser = argparse.ArgumentParser(
        description="Explore Pareto-optimal configurations from a sweep CSV."
    )
    parser.add_argument(
        '--csv', required=True,
        help='Path to the sweep CSV file.'
    )
    parser.add_argument(
        '--metrics', default=None,
        help='Comma-separated list of metric column names to use for Pareto front. '
             'If omitted, any column ending with "_loss" or "_err" will be used.'
    )
    parser.add_argument(
        '--top_k', type=int, default=None,
        help='If set, only print the top_k Pareto configs sorted by the first metric.'
    )
    parser.add_argument(
        '--out_dir', default=None,
        help='If specified, save Pareto configs to this directory as CSV.'
    )
    args = parser.parse_args()

    # Load results
    df = pd.read_csv(args.csv)

    # Determine metric columns
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',')]
    else:
        metrics = [c for c in df.columns if c.endswith('_loss') or c.endswith('_err')]
    if not metrics:
        raise ValueError("No metric columns found. Please specify with --metrics.")

    # Identify Pareto-optimal rows
    mask = compute_pareto_front(df, metrics)
    pareto_df = df[mask].copy()

    # Sort Pareto configs by the total loss.
    if 'total_loss' in metrics:
        pareto_df = pareto_df.sort_values('total_loss')
    else:
        pareto_df = pareto_df.sort_values(metrics[0])

    # Identify hyperparameter columns as those not in metrics
    hp_cols = [c for c in df.columns if c not in metrics]

    # Print summary
    print(f"Found {len(pareto_df)} Pareto-optimal configurations out of {len(df)} total.")
    display_cols = hp_cols + metrics
    if args.top_k:
        print(pareto_df[display_cols].head(args.top_k).to_string(index=False))
    else:
        print(pareto_df[display_cols].to_string(index=False))

    # Optionally save
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, 'pareto_configs.csv')
        pareto_df.to_csv(out_path, index=False)
        print(f"Pareto configurations saved to {out_path}")


if __name__ == '__main__':
    main()
