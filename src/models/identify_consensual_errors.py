"""
Identifica as 39 amostras em que todos os 5 classificadores erram em validacao
cruzada 5-fold (track humano, 22 features, n=534).

Output:
  outputs/decisive_experiment/consensual_errors.csv  (com lat, lon, date, true_label)
  print de amostras por categoria (FN=Espurio classificado como Real;
                                    FP=Real classificado como Espurio)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, max_depth=10, n_estimators=200, random_state=42, n_jobs=-1, verbose=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, eval_metric='logloss'),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(GT_CSV)
    df = df[df['human_label'].isin([0, 1])].copy().reset_index(drop=True)
    df['human_label'] = df['human_label'].astype(int)
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')

    # Merge spectral features
    painel = pd.read_csv(PAINEL_CSV)
    painel['acq_datetime'] = pd.to_datetime(painel['acq_datetime'], format='mixed')
    keys = ['latitude', 'longitude', 'acq_datetime']
    extra_cols = [c for c in FEATURES_SPECTRAL if c in painel.columns]
    df = df.merge(painel[keys + extra_cols].drop_duplicates(keys), on=keys, how='left')

    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    df = df.dropna(subset=FEATURES_22).copy().reset_index(drop=True)
    print(f'Data after filtering: n={len(df)}')

    X = df[FEATURES_22].values
    y = df['human_label'].values

    # 5-fold CV, accumulate out-of-fold predictions per model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = {m: np.zeros(len(df), dtype=int) for m in get_models()}
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr = y[tr]
        for name, model in get_models().items():
            clf = model
            try:
                clf.fit(X_tr, y_tr)
                preds[name][te] = clf.predict(X_te)
            except Exception as e:
                print(f'  [{name}] fold {fold_idx} error: {e}')

    # Stack predictions
    pred_matrix = np.column_stack([preds[m] for m in preds])

    # Consensual error: all 5 wrong
    all_same = (pred_matrix == pred_matrix[:, 0:1]).all(axis=1)
    all_wrong = all_same & (pred_matrix[:, 0] != y)
    n_consensual = int(all_wrong.sum())
    print(f'\nConsensual errors (all 5 models wrong): n = {n_consensual}')

    fn = all_wrong & (y == 1)  # true Spurious, predicted Real
    fp = all_wrong & (y == 0)  # true Real, predicted Spurious
    print(f'  FN (Spurious->Real): {fn.sum()}')
    print(f'  FP (Real->Spurious): {fp.sum()}')

    # Save consensual errors with details
    cons_df = df[all_wrong].copy()
    cons_df['error_type'] = np.where(cons_df['human_label'] == 1, 'FN_spurious_as_real', 'FP_real_as_spurious')
    cons_df['date_str'] = cons_df['acq_datetime'].dt.strftime('%Y-%m-%d')
    # Thumbnail key pattern (matches docs/labeling_campaign/outputs/thumbnails/)
    cons_df['thumb_key'] = cons_df.apply(
        lambda r: f"m{abs(r['latitude']):.5f}_m{abs(r['longitude']):.5f}_{r['acq_datetime'].strftime('%Ym%mm%d')}".replace('.', '.'),
        axis=1
    )
    keep_cols = ['sample_id', 'latitude', 'longitude', 'date_str', 'human_label',
                 'error_type', 'dnbr_max', 'dnbr_mean', 'pct_burned', 'frp',
                 'brightness', 'persistence_score', 'land_use_class', 'thumb_key']
    keep_cols = [c for c in keep_cols if c in cons_df.columns]
    cons_df = cons_df[keep_cols].sort_values('error_type')
    out_path = OUT_DIR / 'consensual_errors.csv'
    cons_df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

    # Top representative samples
    print('\n--- Top 7 FN by |dNBR_max| (mais informativos) ---')
    fn_df = cons_df[cons_df['error_type'] == 'FN_spurious_as_real']
    if 'dnbr_max' in fn_df.columns:
        print(fn_df.nlargest(7, 'dnbr_max')[['sample_id', 'latitude', 'longitude', 'date_str', 'dnbr_max', 'frp']].to_string(index=False))

    print('\n--- Todos os FP (Real->Spurious) ---')
    print(cons_df[cons_df['error_type'] == 'FP_real_as_spurious'][['sample_id', 'latitude', 'longitude', 'date_str', 'dnbr_max', 'frp']].to_string(index=False))


if __name__ == '__main__':
    main()
