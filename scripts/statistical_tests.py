"""
Testes estatisticos formais (McNemar pairwise + Wilcoxon signed-rank) entre
os 6 modelos do experimento decisivo, no track Humano com 22 features.

Output:
- outputs/decisive_experiment/statistical_tests_mcnemar.csv  (matriz 6x6 de p-values)
- outputs/decisive_experiment/statistical_tests_wilcoxon.csv (Wilcoxon nas Kappa per-fold)
- outputs/decisive_experiment/statistical_tests_summary.json

Usage:
    python scripts/statistical_tests.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon

import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
PAINEL_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'decisive_experiment'

FEATURES_13 = [
    'brightness', 'confidence', 'frp', 'hotspot_count', 'persistence_score',
    'temperature', 'dewpoint', 'wind_speed', 'precipitation', 'rh',
    'drying_index', 'month', 'is_dry_season',
]
FEATURES_SPECTRAL = [
    'mean_ndvi_pre', 'mean_ndvi_post', 'delta_ndvi',
    'mean_nbr_pre', 'mean_nbr_post', 'dnbr_mean',
    'dnbr_max', 'pct_burned', 'land_use_class',
]
FEATURES_22 = FEATURES_13 + FEATURES_SPECTRAL


def get_models():
    """Mesmos hyperparametros do run_decisive_experiment.py (sem class_weight)."""
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            num_leaves=31, learning_rate=0.05, max_depth=10,
            n_estimators=200, random_state=42, n_jobs=-1, verbose=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, n_jobs=-1, eval_metric='logloss',
        ),
    }


def cv_oof_predictions(model, X, y, n_splits=5, random_state=42):
    """Out-of-fold predictions + per-fold Kappa list."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = np.full_like(y, fill_value=-99)
    fold_kappas = []
    for train_idx, test_idx in skf.split(X, y):
        clf = clone(model)
        clf.fit(X[train_idx], y[train_idx])
        yp = clf.predict(X[test_idx])
        y_pred[test_idx] = yp
        fold_kappas.append(cohen_kappa_score(y[test_idx], yp))
    return y_pred, np.array(fold_kappas)


def mcnemar_pairwise(predictions: dict[str, np.ndarray], y_true: np.ndarray):
    """Matriz pairwise McNemar de p-values."""
    model_names = list(predictions.keys())
    n = len(model_names)
    pvals = pd.DataFrame(np.ones((n, n)), index=model_names, columns=model_names)
    rows = []
    for a, b in combinations(model_names, 2):
        ya = (predictions[a] == y_true).astype(int)
        yb = (predictions[b] == y_true).astype(int)
        # Tabela 2x2: ambos certos | a certo, b errado | a errado, b certo | ambos errados
        table = np.array([
            [((ya == 1) & (yb == 1)).sum(), ((ya == 1) & (yb == 0)).sum()],
            [((ya == 0) & (yb == 1)).sum(), ((ya == 0) & (yb == 0)).sum()],
        ])
        # McNemar exato (binomial) se discordancia pequena, chi-quadrado caso contrario
        disagree = table[0, 1] + table[1, 0]
        exact = disagree < 25
        result = mcnemar(table, exact=exact, correction=True)
        p = result.pvalue
        pvals.loc[a, b] = p
        pvals.loc[b, a] = p
        rows.append({
            'model_a': a, 'model_b': b,
            'a_only_correct': int(table[0, 1]),
            'b_only_correct': int(table[1, 0]),
            'both_correct': int(table[0, 0]),
            'both_wrong': int(table[1, 1]),
            'statistic': float(result.statistic),
            'p_value': float(p),
            'test': 'exact' if exact else 'chi2_corrected',
        })
    return pvals, pd.DataFrame(rows)


def holm_correction(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni correction."""
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    adjusted = np.zeros_like(pvalues)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, pvalues[idx] * (n - rank))
    # Enforce monotonicity
    cumulative_max = 0
    for idx in sorted_idx:
        cumulative_max = max(cumulative_max, adjusted[idx])
        adjusted[idx] = cumulative_max
    return adjusted


def wilcoxon_pairwise(fold_kappas: dict[str, np.ndarray]):
    """Wilcoxon signed-rank test pairwise nas per-fold Kappas."""
    model_names = list(fold_kappas.keys())
    rows = []
    for a, b in combinations(model_names, 2):
        ka, kb = fold_kappas[a], fold_kappas[b]
        try:
            stat, p = wilcoxon(ka, kb)
        except ValueError:
            stat, p = float('nan'), float('nan')
        rows.append({
            'model_a': a,
            'model_b': b,
            'mean_kappa_a': round(ka.mean(), 4),
            'mean_kappa_b': round(kb.mean(), 4),
            'mean_diff': round(ka.mean() - kb.mean(), 4),
            'wilcoxon_statistic': round(stat, 4) if not np.isnan(stat) else None,
            'p_value': round(p, 4) if not np.isnan(p) else None,
        })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('STATISTICAL TESTS: McNemar + Wilcoxon (Track Humano, 22 feat)')
    print('=' * 60)

    main_df = pd.read_csv(GT_CSV)
    df = main_df[main_df['human_label'].isin([0, 1])].copy()
    df['human_label'] = df['human_label'].astype(int)

    painel = pd.read_csv(PAINEL_CSV)
    painel['acq_datetime'] = pd.to_datetime(painel['acq_datetime'], format='mixed')
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')

    keys = ['latitude', 'longitude', 'acq_datetime']
    extra_cols = [c for c in FEATURES_SPECTRAL if c in painel.columns]
    df = df.merge(painel[keys + extra_cols].drop_duplicates(keys), on=keys, how='left')
    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    df = df.dropna(subset=FEATURES_22).copy()
    print(f'Amostras com 22 features (binary Real vs Espurio): {len(df)}')

    X = df[FEATURES_22].values
    y = df['human_label'].astype(int).values

    print('\nTreinando 5 modelos com 5-fold CV...')
    predictions = {}
    fold_kappas = {}
    for name, model in get_models().items():
        print(f'  {name}...', end='', flush=True)
        y_pred, fk = cv_oof_predictions(model, X, y, n_splits=5)
        predictions[name] = y_pred
        fold_kappas[name] = fk
        print(f' Kappa = {cohen_kappa_score(y, y_pred):.3f} (folds: {fk.round(3).tolist()})')

    # McNemar
    print('\n--- McNemar pairwise ---')
    mcnemar_matrix, mcnemar_pairs = mcnemar_pairwise(predictions, y)
    # Holm correction
    pvals = mcnemar_pairs['p_value'].values
    mcnemar_pairs['p_holm'] = np.round(holm_correction(pvals), 4)
    mcnemar_pairs['significant_holm_05'] = mcnemar_pairs['p_holm'] < 0.05
    print(mcnemar_pairs[['model_a', 'model_b', 'p_value', 'p_holm', 'significant_holm_05']].to_string(index=False))
    mcnemar_pairs.to_csv(OUT_DIR / 'statistical_tests_mcnemar.csv', index=False)
    mcnemar_matrix.to_csv(OUT_DIR / 'statistical_tests_mcnemar_matrix.csv')

    # Wilcoxon
    print('\n--- Wilcoxon pairwise (per-fold Kappa) ---')
    wilcoxon_df = wilcoxon_pairwise(fold_kappas)
    wilcoxon_df.to_csv(OUT_DIR / 'statistical_tests_wilcoxon.csv', index=False)
    print(wilcoxon_df.to_string(index=False))

    # Summary
    summary = {
        'n_samples': len(df),
        'features': '22 (13 base + 9 spectral)',
        'cv_folds': 5,
        'best_model_by_kappa': max(fold_kappas.keys(), key=lambda m: fold_kappas[m].mean()),
        'kappa_per_model': {m: round(float(fold_kappas[m].mean()), 4) for m in fold_kappas},
        'n_significant_pairs_holm_05': int(mcnemar_pairs['significant_holm_05'].sum()),
        'total_pairs_tested': len(mcnemar_pairs),
    }
    (OUT_DIR / 'statistical_tests_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )

    print(f'\nSalvo: {OUT_DIR / "statistical_tests_mcnemar.csv"}')
    print(f'Salvo: {OUT_DIR / "statistical_tests_wilcoxon.csv"}')
    print(f'Salvo: {OUT_DIR / "statistical_tests_summary.json"}')
    print(f'\nResumo: {summary["n_significant_pairs_holm_05"]}/{summary["total_pairs_tested"]} pares com diferenca significativa apos Holm correction.')
    print(f'Melhor modelo: {summary["best_model_by_kappa"]} (Kappa = {summary["kappa_per_model"][summary["best_model_by_kappa"]]:.3f})')


if __name__ == '__main__':
    main()
