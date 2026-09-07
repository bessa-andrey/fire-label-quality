"""
Computa metricas per-class (Precision/Recall/F1 por Real, Espurio, Incerto)
para o melhor modelo do experimento decisivo, usando 5-fold CV + bootstrap CI.

Output:
- outputs/decisive_experiment/per_class_metrics.csv
- outputs/decisive_experiment/per_class_metrics.png

Usage:
    python scripts/compute_per_class_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    """Retorna dict de modelos comparaveis no experimento decisivo."""
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        'LightGBM': lgb.LGBMClassifier(
            num_leaves=31, learning_rate=0.05, max_depth=10,
            n_estimators=200, random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, n_jobs=-1, eval_metric='logloss', use_label_encoder=False,
        ),
    }


def cv_predictions(model, X, y, n_splits=5, random_state=42):
    """Retorna y_pred (out-of-fold predictions, mesmo tamanho de y)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = np.full_like(y, fill_value=-99)
    for train_idx, test_idx in skf.split(X, y):
        clf = model
        # Re-fit a fresh estimator per fold via clone for safety
        from sklearn.base import clone
        clf = clone(model)
        clf.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(X[test_idx])
    return y_pred


def bootstrap_per_class_metrics(y_true, y_pred, classes, n_boot=1000, random_state=42):
    """Bootstrap CI95% para precision/recall/F1 por classe."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boots = {c: {'precision': [], 'recall': [], 'f1': []} for c in classes}
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        report = classification_report(yt, yp, labels=classes, output_dict=True, zero_division=0)
        for c in classes:
            key = str(c)
            if key in report:
                boots[c]['precision'].append(report[key]['precision'])
                boots[c]['recall'].append(report[key]['recall'])
                boots[c]['f1'].append(report[key]['f1-score'])
    cis = {}
    for c in classes:
        cis[c] = {
            'precision_lo': np.percentile(boots[c]['precision'], 2.5),
            'precision_hi': np.percentile(boots[c]['precision'], 97.5),
            'recall_lo': np.percentile(boots[c]['recall'], 2.5),
            'recall_hi': np.percentile(boots[c]['recall'], 97.5),
            'f1_lo': np.percentile(boots[c]['f1'], 2.5),
            'f1_hi': np.percentile(boots[c]['f1'], 97.5),
        }
    return cis


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('PER-CLASS METRICS (Real vs Espurio vs Incerto)')
    print('=' * 60)

    # Carregar dados
    main_df = pd.read_csv(GT_CSV)
    print(f'Total amostras com GT: {len(main_df)}')
    print(f'Distribuicao human_label: {main_df["human_label"].value_counts(dropna=False).to_dict()}')

    # Para per-class incluindo Incerto, mantemos todas as 3 classes
    df = main_df.dropna(subset=['human_label']).copy()
    df['human_label'] = df['human_label'].astype(int)

    # Mergear com painel_labeled_expanded para ter as 22 features
    painel = pd.read_csv(PAINEL_CSV)
    painel['acq_datetime'] = pd.to_datetime(painel['acq_datetime'], format='mixed')
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')

    keys = ['latitude', 'longitude', 'acq_datetime']
    # Garantir colunas de features
    extra_cols = [c for c in FEATURES_SPECTRAL if c in painel.columns]
    df = df.merge(painel[keys + extra_cols].drop_duplicates(keys), on=keys, how='left')

    # is_dry_season (derivada)
    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)

    # Filtrar apenas amostras com 22 features completas
    df_22 = df.dropna(subset=FEATURES_22).copy()
    print(f'\nAmostras com 22 features completas (incluindo Incerto): {len(df_22)}')
    print(f'Distribuicao: {df_22["human_label"].value_counts(dropna=False).to_dict()}')

    # Mas o experimento decisivo opera em 3 classes? Vou reportar 3 versões:
    # 1. Binario (Real vs Espurio) - mais comum em papers
    # 2. 3-classes (Real / Espurio / Incerto) - per-class mostra como modelo lida com Incerto

    rows = []

    for label_mode, df_mode, classes, mode_str in [
        ('binary', df_22[df_22['human_label'].isin([0, 1])], [0, 1], 'binary'),
        ('3class', df_22, [0, 1, -1], '3class'),
    ]:
        print(f'\n--- {mode_str.upper()} mode (n={len(df_mode)}) ---')
        if len(df_mode) < 100:
            print(f'  Pulando: poucos exemplos')
            continue

        X = df_mode[FEATURES_22].values
        y = df_mode['human_label'].astype(int).values

        models = get_models()
        for model_name, model in models.items():
            # XGBoost nao aceita rotulo -1; pular no modo 3class
            if mode_str == '3class' and model_name == 'XGBoost':
                print(f'  {model_name}... PULADO (nao aceita classe -1)')
                continue
            print(f'  {model_name}...', end='', flush=True)
            y_pred = cv_predictions(model, X, y, n_splits=5)
            report = classification_report(y, y_pred, labels=classes, output_dict=True, zero_division=0)
            cis = bootstrap_per_class_metrics(y, y_pred, classes, n_boot=500)

            for c in classes:
                k = str(c)
                if k not in report:
                    continue
                rows.append({
                    'mode': mode_str,
                    'model': model_name,
                    'class': c,
                    'class_name': {0: 'Real', 1: 'Espurio', -1: 'Incerto'}[c],
                    'precision': round(report[k]['precision'], 4),
                    'precision_ci_lo': round(cis[c]['precision_lo'], 4),
                    'precision_ci_hi': round(cis[c]['precision_hi'], 4),
                    'recall': round(report[k]['recall'], 4),
                    'recall_ci_lo': round(cis[c]['recall_lo'], 4),
                    'recall_ci_hi': round(cis[c]['recall_hi'], 4),
                    'f1': round(report[k]['f1-score'], 4),
                    'f1_ci_lo': round(cis[c]['f1_lo'], 4),
                    'f1_ci_hi': round(cis[c]['f1_hi'], 4),
                    'support': int(report[k]['support']),
                })
            # macro and weighted averages
            for avg_key, avg_label in [('macro avg', 'macro'), ('weighted avg', 'weighted')]:
                rows.append({
                    'mode': mode_str,
                    'model': model_name,
                    'class': avg_label,
                    'class_name': f'avg_{avg_label}',
                    'precision': round(report[avg_key]['precision'], 4),
                    'precision_ci_lo': None, 'precision_ci_hi': None,
                    'recall': round(report[avg_key]['recall'], 4),
                    'recall_ci_lo': None, 'recall_ci_hi': None,
                    'f1': round(report[avg_key]['f1-score'], 4),
                    'f1_ci_lo': None, 'f1_ci_hi': None,
                    'support': int(report[avg_key]['support']),
                })
            print(' OK')

    df_out = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'per_class_metrics.csv'
    df_out.to_csv(out_csv, index=False)
    print(f'\nSalvo: {out_csv}')

    # Plot binario (mais relevante)
    binary = df_out[(df_out['mode'] == 'binary') & (df_out['class'].isin([0, 1]))]
    if not binary.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['precision', 'recall', 'f1']
        for ax, metric in zip(axes, metrics):
            pivot = binary.pivot(index='model', columns='class_name', values=metric)
            pivot.plot(kind='bar', ax=ax, color=['#2d6a4f', '#c8362d'])
            ax.set_title(f'{metric.capitalize()} per class (binary)')
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax.legend(title='Class')
            ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'per_class_metrics.png', dpi=140, bbox_inches='tight')
        print(f'Salvo: {OUT_DIR / "per_class_metrics.png"}')

    print('\nDone.')


if __name__ == '__main__':
    main()
