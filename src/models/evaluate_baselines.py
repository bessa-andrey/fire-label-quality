"""
Evaluate simple baselines against manual ground truth.

Implements threshold-based and rule-based classifiers as benchmarks
for the ML model selection pipeline.

Usage:
    python src/models/evaluate_baselines.py
    python src/models/evaluate_baselines.py --gt-path data/processed/validation/validation_manual_gt.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GT = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'baselines'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def bootstrap_kappa(gt, pred, n_boot=1000):
    """Compute bootstrap 95% CI for Cohen's Kappa."""
    rng = np.random.RandomState(42)
    kappas = []
    n = len(gt)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            kappas.append(cohen_kappa_score(gt[idx], pred[idx]))
        except Exception:
            continue
    return {
        'mean': float(np.mean(kappas)),
        'ci_lo': float(np.percentile(kappas, 2.5)),
        'ci_hi': float(np.percentile(kappas, 97.5)),
    }


def threshold_sweep(df, gt_col, feature_col, thresholds, real_if_above=True):
    """Sweep thresholds and compute kappa at each.

    real_if_above: if True, classify as Real (0) when feature >= T
    Returns list of (threshold, kappa, accuracy).
    """
    gt = df[gt_col].values
    results = []
    for t in thresholds:
        if real_if_above:
            pred = np.where(df[feature_col].values >= t, 0, 1)  # 0=Real, 1=Spurious
        else:
            pred = np.where(df[feature_col].values < t, 0, 1)
        k = cohen_kappa_score(gt, pred)
        a = accuracy_score(gt, pred)
        results.append((t, k, a))
    return results


def multi_rule_baseline(df):
    """Multi-rule: confidence >= 70 AND frp >= 5 AND persistence >= 0.3 → Real."""
    real_mask = (
        (df['confidence'] >= 70) &
        (df['frp'] >= 5) &
        (df['persistence_score'] >= 0.3)
    )
    return np.where(real_mask, 0, 1)  # 0=Real, 1=Spurious


def main():
    parser = argparse.ArgumentParser(description='Evaluate simple baselines')
    parser.add_argument('--gt-path', type=str, default=None,
                        help='Path to ground truth CSV (default: validation_manual_gt.csv)')
    parser.add_argument('--gt-col', type=str, default=None,
                        help='GT column name (auto-detect: human_label > manual_label)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_path = Path(args.gt_path) if args.gt_path else DEFAULT_GT
    if not gt_path.exists():
        # Fall back to auto-dNBR validation
        gt_path = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_dnbr_merged.csv'

    print("=" * 60)
    print("EVALUATE SIMPLE BASELINES")
    print("=" * 60)

    df = pd.read_csv(gt_path)
    print(f"Loaded: {gt_path.name} ({len(df)} rows)")

    # Detect GT column
    gt_col = args.gt_col
    if gt_col is None:
        if 'human_label' in df.columns and df['human_label'].notna().sum() >= 50:
            gt_col = 'human_label'
        else:
            gt_col = 'manual_label'
    print(f"GT column: {gt_col}")

    # Filter to valid GT rows
    valid = df[df[gt_col].notna() & (df[gt_col] != -1)].copy()
    valid[gt_col] = valid[gt_col].astype(int)
    print(f"Valid GT samples: {len(valid)}")
    n_real = (valid[gt_col] == 0).sum()
    n_spur = (valid[gt_col] == 1).sum()
    print(f"  Real (0): {n_real}, Spurious (1): {n_spur}")

    gt = valid[gt_col].values
    results = {}

    # ── Baseline 1: FIRMS Confidence Threshold ───────────────────────
    print("\n--- Baseline 1: FIRMS Confidence Threshold ---")
    conf_thresholds = [30, 40, 50, 60, 70, 80, 90]
    conf_results = threshold_sweep(valid, gt_col, 'confidence', conf_thresholds)
    best_conf = max(conf_results, key=lambda x: x[1])
    print(f"  Best threshold: {best_conf[0]}, Kappa={best_conf[1]:.4f}, Acc={best_conf[2]:.4f}")

    pred_conf = np.where(valid['confidence'].values >= best_conf[0], 0, 1)
    ci_conf = bootstrap_kappa(gt, pred_conf)
    results['FIRMS Confidence'] = {
        'kappa': best_conf[1], 'accuracy': best_conf[2],
        'threshold': best_conf[0], 'bootstrap_ci': ci_conf,
        'sweep': [(t, k, a) for t, k, a in conf_results],
    }

    # ── Baseline 2: FRP Threshold ────────────────────────────────────
    print("\n--- Baseline 2: FRP Threshold ---")
    frp_thresholds = [1, 2, 5, 10, 15, 20, 30, 50]
    frp_results = threshold_sweep(valid, gt_col, 'frp', frp_thresholds)
    best_frp = max(frp_results, key=lambda x: x[1])
    print(f"  Best threshold: {best_frp[0]} MW, Kappa={best_frp[1]:.4f}, Acc={best_frp[2]:.4f}")

    pred_frp = np.where(valid['frp'].values >= best_frp[0], 0, 1)
    ci_frp = bootstrap_kappa(gt, pred_frp)
    results['FRP Threshold'] = {
        'kappa': best_frp[1], 'accuracy': best_frp[2],
        'threshold': best_frp[0], 'bootstrap_ci': ci_frp,
        'sweep': [(t, k, a) for t, k, a in frp_results],
    }

    # ── Baseline 3: Multi-rule ───────────────────────────────────────
    print("\n--- Baseline 3: Multi-rule ---")
    pred_multi = multi_rule_baseline(valid)
    k_multi = cohen_kappa_score(gt, pred_multi)
    a_multi = accuracy_score(gt, pred_multi)
    ci_multi = bootstrap_kappa(gt, pred_multi)
    print(f"  Kappa={k_multi:.4f}, Acc={a_multi:.4f}")
    results['Multi-rule'] = {
        'kappa': k_multi, 'accuracy': a_multi,
        'rule': 'confidence>=70 AND frp>=5 AND persistence>=0.3',
        'bootstrap_ci': ci_multi,
    }

    # ── Baseline 4: Weak Labels ──────────────────────────────────────
    print("\n--- Baseline 4: Weak Labels (is_reliable) ---")
    pred_weak = (1 - valid['is_reliable'].astype(int)).values  # Convert to 0=Real, 1=Spurious
    k_weak = cohen_kappa_score(gt, pred_weak)
    a_weak = accuracy_score(gt, pred_weak)
    ci_weak = bootstrap_kappa(gt, pred_weak)
    print(f"  Kappa={k_weak:.4f}, Acc={a_weak:.4f}")
    results['Weak Labels'] = {
        'kappa': k_weak, 'accuracy': a_weak,
        'bootstrap_ci': ci_weak,
    }

    # ── Baseline 5: Painel Labels ────────────────────────────────────
    if 'gt_painel' in valid.columns and valid['gt_painel'].notna().sum() > 0:
        print("\n--- Baseline 5: Painel Labels (gt_painel) ---")
        has_painel = valid['gt_painel'].notna()
        if has_painel.sum() > 0:
            sub = valid[has_painel]
            pred_painel = (1 - sub['gt_painel'].astype(int)).values
            gt_sub = sub[gt_col].values
            k_painel = cohen_kappa_score(gt_sub, pred_painel)
            a_painel = accuracy_score(gt_sub, pred_painel)
            ci_painel = bootstrap_kappa(gt_sub, pred_painel)
            print(f"  Kappa={k_painel:.4f}, Acc={a_painel:.4f} (n={len(sub)})")
            results['Painel Labels'] = {
                'kappa': k_painel, 'accuracy': a_painel,
                'n_samples': int(len(sub)),
                'bootstrap_ci': ci_painel,
            }

    # ── Summary Table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"{'Baseline':<25} {'Kappa':>8} {'Acc':>7} {'CI 95%':>20}")
    print("-" * 60)
    for name, r in sorted(results.items(), key=lambda x: x[1]['kappa'], reverse=True):
        ci_str = ""
        if 'bootstrap_ci' in r:
            ci = r['bootstrap_ci']
            ci_str = f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]"
        print(f"{name:<25} {r['kappa']:>8.4f} {r['accuracy']:>7.4f} {ci_str:>20}")

    # ── Figures ──────────────────────────────────────────────────────

    # Figure 1: Threshold sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ts_conf = [t for t, k, a in conf_results]
    ks_conf = [k for t, k, a in conf_results]
    ax1.plot(ts_conf, ks_conf, 'o-', color='#2166ac', linewidth=2, markersize=8)
    ax1.axhline(y=best_conf[1], color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('FIRMS Confidence Threshold')
    ax1.set_ylabel("Cohen's Kappa")
    ax1.set_title('Confidence Threshold Sweep')

    ts_frp = [t for t, k, a in frp_results]
    ks_frp = [k for t, k, a in frp_results]
    ax2.plot(ts_frp, ks_frp, 's-', color='#b2182b', linewidth=2, markersize=8)
    ax2.axhline(y=best_frp[1], color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('FRP Threshold (MW)')
    ax2.set_ylabel("Cohen's Kappa")
    ax2.set_title('FRP Threshold Sweep')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'threshold_sweep.png')
    plt.close()
    print(f"\n[OK] threshold_sweep.png")

    # Figure 2: Bar chart comparison
    names = list(results.keys())
    kappas = [results[n]['kappa'] for n in names]
    colors = ['#4393c3' if k > 0 else '#d6604d' for k in kappas]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), kappas, color=colors, edgecolor='white')

    # Add CI whiskers
    for i, name in enumerate(names):
        if 'bootstrap_ci' in results[name]:
            ci = results[name]['bootstrap_ci']
            ax.errorbar(kappas[i], i, xerr=[[kappas[i] - ci['ci_lo']], [ci['ci_hi'] - kappas[i]]],
                       color='black', capsize=4, linewidth=1.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Cohen's Kappa")
    ax.set_title(f'Baselines vs Ground Truth (n={len(valid)})')
    ax.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'baselines_comparison.png')
    plt.close()
    print(f"[OK] baselines_comparison.png")

    # ── Save report ──────────────────────────────────────────────────
    report = {
        'gt_path': str(gt_path),
        'gt_col': gt_col,
        'n_samples': int(len(valid)),
        'n_real': int(n_real),
        'n_spurious': int(n_spur),
        'results': {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
    }
    report_path = OUTPUT_DIR / 'baselines_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[OK] {report_path.name}")

    print(f"\n[DONE] All outputs in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
