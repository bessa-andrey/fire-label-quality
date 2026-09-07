"""
Experimento Decisivo: Comparação de 4 Estratégias de Rotulagem

Treina 6 classificadores com cada fonte de rótulo e avalia todos
contra 560 anotações humanas (human_label) usando Cohen's Kappa.

Outputs (em outputs/decisive_experiment/):
  - decisive_results.csv         Tabela principal: modelos × estratégias × métricas
  - label_concordance.csv        Matriz Kappa 4×4 entre fontes de rótulo
  - feature_importance_by_strategy.csv
  - decisive_kappa_heatmap.png
  - label_concordance_heatmap.png
  - feature_importance_by_strategy.png
  - roc_curves_by_strategy.png
  - decisive_experiment_report.json

Uso:
    python src/models/run_decisive_experiment.py
    python src/models/run_decisive_experiment.py --n-folds 10
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PAINEL_EXPANDED_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_5strategies.csv'
PAINEL_EXPANDED_CSV_FALLBACK = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'
HUMAN_GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'decisive_experiment'

# 13 features: 11 FIRMS/ERA5 + month + is_dry_season
# SEM features espectrais (evitar tautologia com dNBR como GT)
FEATURE_COLS_BASE = [
    'brightness', 'confidence', 'frp', 'hotspot_count',
    'persistence_score', 'temperature', 'dewpoint',
    'wind_speed', 'precipitation', 'rh', 'drying_index',
]

# Sentinel-2 spectral features (9 additional)
FEATURE_COLS_SPECTRAL = [
    'mean_ndvi_pre', 'mean_ndvi_post', 'delta_ndvi',
    'mean_nbr_pre', 'mean_nbr_post',
    'dnbr_mean', 'dnbr_max', 'pct_burned',
    'land_use_class',
]

# Features more directly aligned with the dNBR thumbnail seen by the human annotator
# Used by the ablation track to test whether the 13->22 gain comes from dNBR proxy
DNBR_PROXY_COLS = ['dnbr_mean', 'dnbr_max', 'pct_burned']

# ---- Ablacao real do dNBR ----
# O dNBR e calculado a partir das cenas Sentinel-2 pre e pos-fogo, e essas mesmas
# duas cenas alimentam o NDVI e o NBR de cada janela. Remover apenas as colunas
# dNBR deixa no modelo variaveis que carregam a mesma informacao por outro caminho.
# A ablacao do M9 remove todas as features derivadas dessas cenas.
SCENE_DERIVED_COLS = [
    'mean_ndvi_pre', 'mean_ndvi_post', 'delta_ndvi',
    'mean_nbr_pre', 'mean_nbr_post',
    'dnbr_mean', 'dnbr_max', 'pct_burned',
]

# Derivadas apenas da cena pre-fogo. Nao podem codificar a cicatriz, porque a cena
# e anterior ao evento. Servem para separar contexto ambiental de evidencia de queima.
PRE_FIRE_SCENE_COLS = ['mean_ndvi_pre', 'mean_nbr_pre']

# land_use_class vem do MapBiomas, nao das cenas pre e pos-fogo, entao sobrevive
# a ablacao do M9.
NON_SCENE_SPECTRAL_COLS = ['land_use_class']

SENTINEL2_CSV = PROJECT_ROOT / 'data' / 'processed' / 'sentinel2_features.csv'


def build_ablation_ladder(base_cols, available_spectral):
    """Monta os niveis de ablacao do M9, do mais pobre ao completo.

    Todos os niveis rodam sobre a mesma subamostra, entao a diferenca de Kappa
    entre dois niveis vizinhos e atribuivel so ao bloco de features que entrou.

    Args:
        base_cols: as 13 features FIRMS/ERA5, ja com month e is_dry_season
        available_spectral: colunas espectrais presentes no dataframe

    Returns:
        lista de (chave, rotulo, colunas)
    """
    def present(cols):
        return [c for c in cols if c in available_spectral]

    non_scene = present(NON_SCENE_SPECTRAL_COLS)
    pre_only = present(PRE_FIRE_SCENE_COLS)
    no_dnbr = present([c for c in available_spectral if c not in DNBR_PROXY_COLS])
    full = present(available_spectral)

    ladder = [
        ('L0_base13', 'Base FIRMS/ERA5, sem nenhuma feature espectral',
         list(base_cols)),
        ('L1_no_scene', 'Ablacao M9, remove as 8 features derivadas das cenas',
         list(base_cols) + non_scene),
        ('L2_pre_fire', 'M9 mais a cena pre-fogo, ainda sem evidencia de cicatriz',
         list(base_cols) + non_scene + pre_only),
        ('L3_no_dnbr_cols', 'Ablacao fraca, remove so as 3 colunas dNBR',
         list(base_cols) + no_dnbr),
        ('L4_full', 'Conjunto completo de 22 features',
         list(base_cols) + full),
    ]
    return ladder

LABEL_STRATEGIES = {
    'weak_labels': {'col': 'is_reliable', 'invert': True, 'dataset': 'full',
                    'display': 'Weak\nlabels'},
    'dnbr_auto': {'col': 'manual_label', 'invert': False, 'dataset': 'gt_only',
                  'display': 'dNBR\nauto'},
    'mapbiomas_fogo': {'col': 'gt_mapbiomas_fire', 'invert': True, 'dataset': 'full',
                       'display': 'MapBiomas\nFogo'},
    'painel_fogo': {'col': 'gt_painel', 'invert': True, 'dataset': 'full',
                    'display': 'Painel\ndo Fogo'},
    'human_annotation': {'col': 'human_label', 'invert': False, 'dataset': 'gt_only',
                         'display': 'Anotação\nhumana'},
}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.titlesize': 10.5,
    'lines.linewidth': 1.3, 'axes.linewidth': 0.7,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': False,
    'axes.spines.top': False, 'axes.spines.right': False,
})

MODEL_COLORS = {
    'DummyClassifier': '#bdbdbd',
    'Logistic Regression': '#999999',
    'Random Forest': '#1b7837',
    'Extra Trees': '#762a83',
    'LightGBM': '#2166ac',
    'XGBoost': '#b2182b',
}


# ============================================================
# DATA LOADING
# ============================================================

def load_and_merge_data():
    """Load painel_expanded (3698) and human_gt (604), merge to get all labels.

    Returns:
        df_full: 3698 samples with is_reliable, gt_painel, features
        df_gt: 560 usable human-labeled samples (excl. uncertain=-1)
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load full dataset (3698 samples)
    csv_path = PAINEL_EXPANDED_CSV if PAINEL_EXPANDED_CSV.exists() else PAINEL_EXPANDED_CSV_FALLBACK
    df_full = pd.read_csv(csv_path)
    print(f"[OK] Loading from: {csv_path.name}")
    df_full['acq_datetime'] = pd.to_datetime(df_full['acq_datetime'], format='mixed')
    print(f"[OK] Painel expanded: {len(df_full)} samples")
    print(f"     gt_painel: {(df_full['gt_painel']==1).sum()} real, "
          f"{(df_full['gt_painel']==0).sum()} spurious")
    print(f"     is_reliable: {(df_full['is_reliable']==1).sum()} real, "
          f"{(df_full['is_reliable']==0).sum()} spurious")

    # Load human GT (604 samples)
    df_gt_raw = pd.read_csv(HUMAN_GT_CSV)
    df_gt_raw['acq_datetime'] = pd.to_datetime(df_gt_raw['acq_datetime'], format='mixed')
    print(f"\n[OK] Human GT: {len(df_gt_raw)} samples")

    # Filter to usable (exclude uncertain=-1)
    df_gt = df_gt_raw[df_gt_raw['human_label'].isin([0, 1])].copy()
    print(f"     Usable (excl. uncertain): {len(df_gt)}")
    print(f"     human_label: {(df_gt['human_label']==0).sum()} real (0), "
          f"{(df_gt['human_label']==1).sum()} spurious (1)")

    # Merge to get gt_painel onto human GT samples
    df_full['_lat_r'] = df_full['latitude'].round(4)
    df_full['_lon_r'] = df_full['longitude'].round(4)
    df_full['_date'] = df_full['acq_datetime'].dt.strftime('%Y-%m-%d')

    df_gt['_lat_r'] = df_gt['latitude'].round(4)
    df_gt['_lon_r'] = df_gt['longitude'].round(4)
    df_gt['_date'] = df_gt['acq_datetime'].dt.strftime('%Y-%m-%d')

    # Columns to merge from full dataset
    merge_cols_from_full = ['_lat_r', '_lon_r', '_date', 'gt_painel']
    if 'gt_mapbiomas_fire' in df_full.columns:
        merge_cols_from_full.append('gt_mapbiomas_fire')

    painel_cols = df_full[merge_cols_from_full].drop_duplicates()
    df_gt = df_gt.merge(painel_cols, on=['_lat_r', '_lon_r', '_date'], how='left',
                        suffixes=('', '_merged'))

    # Use merged columns if available
    for col in ['gt_painel', 'gt_mapbiomas_fire']:
        merged_col = f'{col}_merged'
        if merged_col in df_gt.columns:
            df_gt[col] = df_gt[merged_col].combine_first(
                df_gt.get(col, pd.Series(dtype=float))
            )
            df_gt.drop(columns=[merged_col], inplace=True, errors='ignore')

    n_with_painel = df_gt['gt_painel'].notna().sum()
    print(f"     With gt_painel after merge: {n_with_painel}/{len(df_gt)}")

    if 'gt_mapbiomas_fire' in df_gt.columns:
        n_with_mb = df_gt['gt_mapbiomas_fire'].notna().sum()
        n_mb_real = (df_gt['gt_mapbiomas_fire'] == 1).sum()
        n_mb_spur = (df_gt['gt_mapbiomas_fire'] == 0).sum()
        print(f"     With gt_mapbiomas_fire after merge: {n_with_mb}/{len(df_gt)}")
        print(f"     MapBiomas Fogo: {n_mb_real} real, {n_mb_spur} not burned")
    else:
        print("     [WARN] gt_mapbiomas_fire not found in full dataset")

    # Clean up temp columns
    for col in ['_lat_r', '_lon_r', '_date']:
        df_full.drop(columns=[col], inplace=True, errors='ignore')
        df_gt.drop(columns=[col], inplace=True, errors='ignore')

    return df_full, df_gt


def attach_spectral(df_gt):
    """Traz as features Sentinel-2 para o dataframe de GT humano, por (lat, lon, data).

    Devolve (dataframe, colunas_espectrais_disponiveis). Usado tanto pelo track
    principal com 22 features quanto pelo track de ablacao.
    """
    if not SENTINEL2_CSV.exists():
        print(f"[WARN] {SENTINEL2_CSV.name} nao encontrado, seguindo sem espectrais")
        return df_gt, []

    df_spectral = pd.read_csv(SENTINEL2_CSV)
    df_spectral['acq_datetime'] = pd.to_datetime(df_spectral['acq_datetime'], format='mixed')
    df_spectral['_lat_r'] = df_spectral['latitude'].round(4)
    df_spectral['_lon_r'] = df_spectral['longitude'].round(4)
    df_spectral['_date'] = df_spectral['acq_datetime'].dt.strftime('%Y-%m-%d')

    out = df_gt.copy()
    out['_lat_r'] = out['latitude'].round(4)
    out['_lon_r'] = out['longitude'].round(4)
    out['_date'] = pd.to_datetime(out['acq_datetime'], format='mixed').dt.strftime('%Y-%m-%d')

    available = [c for c in FEATURE_COLS_SPECTRAL if c in df_spectral.columns]
    out = out.merge(
        df_spectral[['_lat_r', '_lon_r', '_date'] + available].drop_duplicates(
            subset=['_lat_r', '_lon_r', '_date']),
        on=['_lat_r', '_lon_r', '_date'], how='left', suffixes=('', '_s2'),
    )

    for col in available:
        if f'{col}_s2' in out.columns:
            out[col] = out[f'{col}_s2'].combine_first(out[col])
            out.drop(columns=[f'{col}_s2'], inplace=True)

    out.drop(columns=['_lat_r', '_lon_r', '_date'], inplace=True, errors='ignore')
    return out, available


def prepare_features(df, spectral_cols=None):
    """Extract the feature matrix, add is_dry_season, impute NaN with median.

    Com spectral_cols=None devolve as 13 features FIRMS/ERA5. Passando a lista
    espectral devolve o conjunto de 22, que e o pedido dos itens M10 e R9.
    """
    feature_cols = FEATURE_COLS_BASE.copy()

    if 'month' not in df.columns:
        df['month'] = pd.to_datetime(df['acq_datetime'], format='mixed').dt.month
    feature_cols.append('month')

    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    feature_cols.append('is_dry_season')

    if spectral_cols:
        feature_cols += [c for c in spectral_cols if c in df.columns]

    X = df[feature_cols].copy()

    # Impute NaN with median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return X, feature_cols


def align_label(series, invert):
    """Align labels to unified convention: 0=Real, 1=Spurious."""
    vals = series.astype(int).values
    return 1 - vals if invert else vals


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():
    """Return {name: (model_instance, needs_scaling)}."""
    return {
        'DummyClassifier': (
            DummyClassifier(strategy='stratified', random_state=42),
            False,
        ),
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
            True,
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=200, max_depth=15,
                                   random_state=42, n_jobs=-1),
            False,
        ),
        'Extra Trees': (
            ExtraTreesClassifier(n_estimators=200, max_depth=15,
                                 random_state=42, n_jobs=-1),
            False,
        ),
        'LightGBM': (
            lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, max_depth=10,
                               n_estimators=100, random_state=42, verbose=-1),
            False,
        ),
        'XGBoost': (
            xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100,
                              random_state=42, verbosity=0,
                              use_label_encoder=False, eval_metric='logloss'),
            False,
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(y_true, y_pred, y_proba=None):
    """Compute all metrics against human GT."""
    metrics = {
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics['roc_auc'] = float('nan')
    return metrics


def bootstrap_kappa(y_true, y_pred, n_boot=1000):
    """Bootstrap 95% CI for Cohen's Kappa."""
    rng = np.random.RandomState(42)
    kappas = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            kappas.append(cohen_kappa_score(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    return {
        'mean': float(np.mean(kappas)),
        'ci_lo': float(np.percentile(kappas, 2.5)),
        'ci_hi': float(np.percentile(kappas, 97.5)),
    }


# ============================================================
# EXPERIMENT TRACKS
# ============================================================

def run_strategy_full_train(strategy_name, config, df_full, df_gt, models,
                            spectral_cols=None):
    """Track 1/3: Train on full 3698 samples, test on human-labeled samples.

    For strategies where training data is the full Painel dataset.
    """
    print(f"\n--- Strategy: {strategy_name} (train on {len(df_full)}, test on {len(df_gt)}) ---")

    # Prepare training data
    X_train, _ = prepare_features(df_full.copy(), spectral_cols)
    y_train = align_label(df_full[config['col']], config['invert'])

    # Prepare test data
    X_test, _ = prepare_features(df_gt.copy(), spectral_cols)
    y_test = df_gt['human_label'].astype(int).values  # GT is always human_label

    # Impute test NaN using training medians
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    trained = {}
    roc_data = {}

    for model_name, (model_template, needs_scaling) in models.items():
        model = clone(model_template)
        Xtr = X_train_s if needs_scaling else X_train.values
        Xte = X_test_s if needs_scaling else X_test.values

        model.fit(Xtr, y_train)

        y_pred = model.predict(Xte)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(Xte)[:, 1]

        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        ci = bootstrap_kappa(y_test, y_pred)
        metrics['kappa_ci_lo'] = ci['ci_lo']
        metrics['kappa_ci_hi'] = ci['ci_hi']

        results[model_name] = metrics
        trained[model_name] = model
        if y_proba is not None:
            roc_data[model_name] = (y_test, y_proba)

        print(f"  {model_name:<22} Kappa={metrics['kappa']:.4f} "
              f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]  "
              f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}")

    return results, trained, roc_data


def run_strategy_cv(strategy_name, config, df_gt, models, spectral_cols=None, n_folds=5):
    """Track 2/4: StratifiedKFold CV on human-labeled samples.

    For strategies where training data = test data (smaller datasets).
    """
    # Filter to rows with valid labels for this strategy
    label_col = config['col']
    mask = df_gt[label_col].notna() & (df_gt[label_col] != -1)
    df_valid = df_gt[mask].copy()

    print(f"\n--- Strategy: {strategy_name} ({n_folds}-fold CV on {len(df_valid)} samples) ---")

    X_all, _ = prepare_features(df_valid.copy(), spectral_cols)
    y_strategy = align_label(df_valid[label_col], config['invert'])
    y_human = df_valid['human_label'].astype(int).values

    results = {}
    trained = {}
    roc_data = {}

    for model_name, (model_template, needs_scaling) in models.items():
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        last_model = None

        for train_idx, test_idx in skf.split(X_all, y_strategy):
            X_train = X_all.iloc[train_idx].copy()
            X_test = X_all.iloc[test_idx].copy()
            y_train = y_strategy[train_idx]
            y_test_human = y_human[test_idx]

            # Impute NaN with training median
            for col in X_train.columns:
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = clone(model_template)
            Xtr = X_train_s if needs_scaling else X_train.values
            Xte = X_test_s if needs_scaling else X_test.values

            model.fit(Xtr, y_train)
            last_model = model

            y_pred = model.predict(Xte)
            all_y_true.extend(y_test_human)
            all_y_pred.extend(y_pred)

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(Xte)[:, 1]
                all_y_proba.extend(proba)

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba) if all_y_proba else None

        metrics = evaluate_predictions(all_y_true, all_y_pred, all_y_proba)
        ci = bootstrap_kappa(all_y_true, all_y_pred)
        metrics['kappa_ci_lo'] = ci['ci_lo']
        metrics['kappa_ci_hi'] = ci['ci_hi']

        results[model_name] = metrics
        trained[model_name] = last_model
        if all_y_proba is not None:
            roc_data[model_name] = (all_y_true, all_y_proba)

        print(f"  {model_name:<22} Kappa={metrics['kappa']:.4f} "
              f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]  "
              f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}")

    return results, trained, roc_data


# ============================================================
# CONCORDANCE MATRIX
# ============================================================

def compute_concordance_matrix(df_gt):
    """Compute pairwise Cohen's Kappa between all label sources.

    Dynamically handles 4 or 5 sources depending on available columns.
    For each pair, uses maximum available samples.
    """
    label_sources = {
        'Weak Labels': ('is_reliable', True),
        'dNBR Auto': ('manual_label', False),
        'Painel do Fogo': ('gt_painel', True),
        'Human': ('human_label', False),
    }

    # Add MapBiomas Fogo if available
    # gt_mapbiomas_fire: 1=burned (real), 0=not burned (spurious). Invert to unified 0=Real, 1=Spurious.
    if 'gt_mapbiomas_fire' in df_gt.columns and df_gt['gt_mapbiomas_fire'].notna().sum() > 0:
        label_sources['MapBiomas Fogo'] = ('gt_mapbiomas_fire', True)

    n_sources = len(label_sources)
    print("\n" + "=" * 60)
    print(f"CONCORDANCE MATRIX ({n_sources}x{n_sources})")
    print("=" * 60)

    # Base mask: human + manual_label + is_reliable always available
    mask_base = (
        df_gt['human_label'].isin([0, 1]) &
        df_gt['manual_label'].notna() & (df_gt['manual_label'] != -1) &
        df_gt['is_reliable'].notna()
    )
    df_base = df_gt[mask_base].copy()
    print(f"Samples with base labels (Weak, dNBR, Human): {len(df_base)}")

    # Build NxN matrix: for each pair, use max available samples
    names = list(label_sources.keys())
    n = len(names)
    matrix = pd.DataFrame(np.ones((n, n)), index=names, columns=names)
    n_samples_matrix = pd.DataFrame(0, index=names, columns=names)

    for i in range(n):
        for j in range(i + 1, n):
            ni, nj = names[i], names[j]
            col_i, inv_i = label_sources[ni]
            col_j, inv_j = label_sources[nj]

            # Build mask for this specific pair
            pair_mask = mask_base.copy()
            if col_i not in ['is_reliable', 'manual_label', 'human_label']:
                pair_mask = pair_mask & df_gt[col_i].notna()
            if col_j not in ['is_reliable', 'manual_label', 'human_label']:
                pair_mask = pair_mask & df_gt[col_j].notna()

            df_pair = df_gt[pair_mask]
            if len(df_pair) < 10:
                print(f"  {ni} vs {nj}: SKIPPED (n={len(df_pair)} < 10)")
                matrix.iloc[i, j] = float('nan')
                matrix.iloc[j, i] = float('nan')
                continue

            vi = df_pair[col_i].astype(int).values
            vj = df_pair[col_j].astype(int).values
            li = 1 - vi if inv_i else vi
            lj = 1 - vj if inv_j else vj
            n_samp = len(df_pair)

            k = cohen_kappa_score(li, lj)
            matrix.iloc[i, j] = k
            matrix.iloc[j, i] = k
            n_samples_matrix.iloc[i, j] = n_samp
            n_samples_matrix.iloc[j, i] = n_samp
            print(f"  {ni} vs {nj}: Kappa = {k:.4f} (n={n_samp})")

    return matrix, n_samples_matrix


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importances(trained_models_by_strategy, feature_cols):
    """Extract feature importance from best tree-based model per strategy."""
    fi_data = {}

    for strategy_name, models_dict in trained_models_by_strategy.items():
        # Pick best tree-based model (skip Dummy and LogReg)
        for model_name in ['Extra Trees', 'LightGBM', 'XGBoost', 'Random Forest']:
            if model_name in models_dict:
                model = models_dict[model_name]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    # Normalize to sum to 1
                    importances = importances / importances.sum()
                    fi_data[f"{strategy_name}\n({model_name})"] = importances
                    break

    fi_df = pd.DataFrame(fi_data, index=feature_cols)
    return fi_df


# ============================================================
# PLOTTING
# ============================================================

def plot_kappa_heatmap(all_results, output_dir):
    """Heatmap: rows=models, columns=strategies, values=Kappa."""
    strategies = list(all_results.keys())
    models = list(next(iter(all_results.values())).keys())

    data = np.zeros((len(models), len(strategies)))
    for j, strat in enumerate(strategies):
        for i, model in enumerate(models):
            data[i, j] = all_results[strat][model]['kappa']

    fig, ax = plt.subplots(figsize=(5.81, 3.90))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.8)

    ax.set_xticks(range(len(strategies)))
    display_names = [LABEL_STRATEGIES[s]['display'] for s in strategies]
    ax.set_xticklabels(display_names, fontsize=8.5, ha='center')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8.5)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(strategies)):
            val = data[i, j]
            color = 'white' if val < 0.1 or val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}'.replace('.', ','), ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)

    ax.set_title("Cohen's Kappa por modelo e estratégia de rotulagem,\n"
                 "avaliado contra a referência humana", fontsize=10)
    plt.colorbar(im, ax=ax, label="Cohen's Kappa", shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / 'decisive_kappa_heatmap.png')
    plt.close()
    print(f"[OK] decisive_kappa_heatmap.png")


def plot_concordance_heatmap(matrix, n_samples, output_dir):
    """4×4 annotated heatmap of label concordance."""
    fig, ax = plt.subplots(figsize=(5.20, 4.55))
    data = matrix.values
    im = ax.imshow(data, cmap='RdYlGn', vmin=-0.1, vmax=1.0)

    names = list(matrix.index)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)

    for i in range(len(names)):
        for j in range(len(names)):
            val = data[i, j]
            if i == j:
                text = '1,000'
            else:
                n = int(n_samples.iloc[i, j])
                text = f'{val:.3f}'.replace('.', ',') + f'\n(n={n})'
            color = 'white' if val > 0.7 or val < 0.05 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    ax.set_title("Label Source Concordance\n(Cohen's Kappa)", fontsize=13)
    plt.colorbar(im, ax=ax, label="Cohen's Kappa", shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / 'label_concordance_heatmap.png')
    plt.close()
    print(f"[OK] label_concordance_heatmap.png")


def plot_feature_importance(fi_df, output_dir):
    """Grouped horizontal bar chart comparing feature importance across strategies."""
    if fi_df.empty:
        print("[WARN] No feature importance data to plot")
        return

    n_features = len(fi_df)
    n_strategies = len(fi_df.columns)
    bar_height = 0.8 / n_strategies
    # Uma cor por estrategia. Sao cinco, e a lista precisa ter cinco entradas:
    # com quatro a quinta estrategia reciclava a cor da primeira e as duas
    # ficavam indistinguiveis na legenda.
    colors = ['#2166ac', '#b2182b', '#1b7837', '#762a83', '#e08214']

    fig, ax = plt.subplots(figsize=(5.52, 4.40))
    y_pos = np.arange(n_features)

    for idx, col in enumerate(fi_df.columns):
        offset = (idx - n_strategies / 2 + 0.5) * bar_height
        bars = ax.barh(y_pos + offset, fi_df[col].values, height=bar_height,
                       label=col, color=colors[idx % len(colors)], alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(fi_df.index, fontsize=10)
    ax.set_xlabel('Importância Relativa (normalizada)')
    ax.set_title('Importância de Features por Estratégia de Rotulagem', fontsize=13)
    ax.invert_yaxis()
    # A legenda fica fora da area de plotagem, na faixa abaixo do eixo. Dentro do
    # grafico ela cobria as barras do canto inferior direito, e o canto que hoje
    # esta livre deixa de estar assim que os dados mudam.
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=3, frameon=False, handlelength=1.4, columnspacing=1.2,
              borderaxespad=0.0)
    plt.tight_layout()
    fig.savefig(output_dir / 'feature_importance_by_strategy.png')
    plt.close()
    print(f"[OK] feature_importance_by_strategy.png")


def plot_feature_importance_22feat(fi_df, best_model_name, output_dir):
    """Horizontal bar chart of the 22 features, spectral ones in red.

    A altura da figura sai do numero de features, e nao de um valor fixo. Com
    4,45 polegadas para 22 rotulos a linha de base de um rotulo caia sobre o
    sublinhado do rotulo de cima, e as palavras se sobrepunham. A folga por
    rotulo abaixo reserva altura em vez de encolher a fonte, o que preservaria
    o defeito medido pelo audit de legibilidade.
    """
    imp_sorted = fi_df[best_model_name].sort_values(ascending=True)

    ALTURA_POR_ROTULO = 0.20     # polegadas, folga que separa dois rotulos de 10 pt
    ALTURA_FIXA = 1.15           # titulo de duas linhas mais o eixo x e seu rotulo
    altura = ALTURA_FIXA + ALTURA_POR_ROTULO * len(imp_sorted)

    fig, ax = plt.subplots(figsize=(4.93, altura))

    colors = ['#d73027' if f in FEATURE_COLS_SPECTRAL else '#4393c3'
              for f in imp_sorted.index]
    ax.barh(range(len(imp_sorted)), imp_sorted.values, color=colors)
    ax.set_yticks(range(len(imp_sorted)))
    ax.set_yticklabels(imp_sorted.index, fontsize=10)
    ax.set_xlabel('Importância Relativa (normalizada)')
    ax.set_title(f'Importância de Features - {best_model_name} (22 features)\n'
                 f'Vermelho = espectral, Azul = FIRMS/ERA5', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / 'feature_importance_22feat.png')
    plt.close()
    print(f"[OK] feature_importance_22feat.png")


def plot_ablation_ladder(ladder, ladder_results, models, n_samples, output_dir):
    """Figura da escada de ablacao do M9: Kappa por nivel, uma linha por modelo."""
    keys = [k for k, _, _ in ladder]
    n_feats = {k: len(c) for k, _, c in ladder}
    xlabels = [f"{k.split('_', 1)[0]}\n{n_feats[k]} feat" for k in keys]

    fig, ax = plt.subplots(figsize=(5.20, 3.50))
    x = np.arange(len(keys))

    for model_name in models:
        if model_name == 'DummyClassifier':
            continue
        ys = [ladder_results[k][model_name]['kappa'] for k in keys]
        ax.plot(x, ys, 'o-', linewidth=1.8, markersize=7,
                color=MODEL_COLORS.get(model_name, '#333333'), label=model_name)

    # Marca o nivel que responde ao M9
    idx_m9 = keys.index('L1_no_scene') if 'L1_no_scene' in keys else None
    if idx_m9 is not None:
        ax.axvline(idx_m9, color='#7A1F3D', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(idx_m9 + 0.06, ax.get_ylim()[1] * 0.97, 'ablacao M9',
                color='#7A1F3D', fontsize=9, va='top')

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Cohen's Kappa vs referencia humana")
    ax.set_xlabel('Nivel de ablacao')
    ax.set_title('Escada de ablacao das features derivadas das cenas\n'
                 f'(n={n_samples}, mesma subamostra em todos os niveis)', fontsize=12)
    ax.axhline(0, color='#999999', linewidth=0.8)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / 'm9_ablation_ladder.png')
    plt.close()
    print(f"[OK] m9_ablation_ladder.png")


def plot_roc_curves(roc_data_by_strategy, output_dir):
    """Multi-panel ROC curves, one per strategy."""
    strategies = list(roc_data_by_strategy.keys())
    n_strats = len(strategies)

    ncols = 3 if n_strats > 4 else 2
    nrows = (n_strats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.81, 4.85))
    axes = axes.ravel()

    # Legenda unica compartilhada. Uma legenda por painel cobria as curvas
    # depois que a figura passou a ser gerada no tamanho impresso.
    handles, rotulos = [], []
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        roc_data = roc_data_by_strategy[strategy]
        aucs = []

        for model_name, (y_true, y_proba) in roc_data.items():
            if model_name == 'DummyClassifier':
                continue
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_val = roc_auc_score(y_true, y_proba)
            aucs.append(auc_val)
            color = MODEL_COLORS.get(model_name, '#333333')
            linha, = ax.plot(fpr, tpr, color=color, linewidth=1.1)
            if idx == 0:
                handles.append(linha)
                rotulos.append(model_name)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
        display = LABEL_STRATEGIES[strategy]['display'].replace('\n', ' ')
        faixa = f'AUC {min(aucs):.2f} a {max(aucs):.2f}' if aucs else ''
        ax.set_title(f'{display}\n{faixa}', fontsize=8.5)
        ax.set_xlabel('FPR', fontsize=8)
        ax.set_ylabel('TPR', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Hide unused axes
    for idx in range(n_strats, len(axes)):
        axes[idx].set_visible(False)

    fig.legend(handles, rotulos, loc='lower right', fontsize=8,
               bbox_to_anchor=(0.98, 0.06), frameon=False)
    fig.suptitle('Curvas ROC por estratégia de rotulagem,\n'
                 'avaliadas contra a referência humana', fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / 'roc_curves_by_strategy.png')
    plt.close()
    print(f"[OK] roc_curves_by_strategy.png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experimento Decisivo: 4 Estratégias de Rotulagem vs GT Humano'
    )
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds for gt_only strategies (default: 5)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: outputs/decisive_experiment)')
    parser.add_argument('--expanded', action='store_true',
                        help='Run additional experiment: human_annotation track with 22 features (13 base + 9 spectral)')
    parser.add_argument('--features', type=int, choices=[13, 22], default=13,
                        help='Feature set for the main 5-strategy track. '
                             '13 = FIRMS/ERA5 only (default, historico). '
                             '22 = adiciona as 9 espectrais Sentinel-2 em TODAS as '
                             'estrategias.')
    # Insumos alternativos. Usados pela recomputacao da Estrategia 1
    # (src/preprocessing/recompute_weak_labels_mcd64a1.py), que grava copias do painel e da
    # referencia humana com is_reliable recomputado a partir do MCD64A1. Sem estas opcoes o
    # script roda exatamente como antes.
    parser.add_argument('--painel-csv', type=Path, default=None,
                        help='Substitui o painel de entrada '
                             '(padrao: painel_labeled_5strategies.csv)')
    parser.add_argument('--gt-csv', type=Path, default=None,
                        help='Substitui a referencia humana '
                             '(padrao: validation_manual_gt.csv)')
    args = parser.parse_args()

    global PAINEL_EXPANDED_CSV, HUMAN_GT_CSV
    if args.painel_csv:
        PAINEL_EXPANDED_CSV = args.painel_csv
    if args.gt_csv:
        HUMAN_GT_CSV = args.gt_csv

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.features == 22:
        output_dir = OUTPUT_DIR.parent / 'decisive_experiment_22feat'
    else:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENTO DECISIVO")
    print("Comparação de 4 Estratégias de Rotulagem")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. Load data
    df_full, df_gt = load_and_merge_data()

    # 1b. Feature set do track principal
    spectral_cols = None
    if args.features == 22:
        print("\n" + "=" * 60)
        print("TRACK PRINCIPAL COM 22 FEATURES")
        print("=" * 60)
        df_gt, available_spectral = attach_spectral(df_gt)

        missing_in_full = [c for c in available_spectral if c not in df_full.columns]
        if missing_in_full:
            raise SystemExit(
                f"[ERRO] O dataset completo nao tem as colunas espectrais {missing_in_full}. "
                f"Rode com --features 13 ou regenere painel_labeled_5strategies.csv."
            )

        # As duas bases sao restritas as amostras com espectral completo. Imputar
        # mediana em 770 linhas do dataset full inventaria evidencia de cicatriz
        # onde nao houve cena utilizavel, entao a restricao e a opcao honesta.
        n_full_before, n_gt_before = len(df_full), len(df_gt)
        df_full = df_full[df_full[available_spectral].notna().all(axis=1)].copy()
        df_gt = df_gt[df_gt[available_spectral].notna().all(axis=1)].copy()

        print(f"  Espectrais disponiveis: {len(available_spectral)} -> {available_spectral}")
        print(f"  Dataset full: {n_full_before} -> {len(df_full)} com espectral completo")
        print(f"  GT humano:    {n_gt_before} -> {len(df_gt)} com espectral completo")
        print(f"  human_label: {(df_gt['human_label']==0).sum()} Real, "
              f"{(df_gt['human_label']==1).sum()} Espurio")
        spectral_cols = available_spectral

    # Sanity check: verify label distributions after alignment
    print("\n--- Sanity Check: Label distributions (aligned 0=Real, 1=Spurious) ---")
    for strat_name, config in LABEL_STRATEGIES.items():
        col = config['col']
        invert = config['invert']
        if config['dataset'] == 'full':
            vals = align_label(df_full[col], invert)
            n = len(vals)
        else:
            mask = df_gt[col].notna() & (df_gt[col] != -1)
            vals = align_label(df_gt.loc[mask, col], invert)
            n = len(vals)
        n_real = (vals == 0).sum()
        n_spur = (vals == 1).sum()
        print(f"  {strat_name:<20} n={n:>5}  Real={n_real:>5} ({100*n_real/n:.1f}%)  "
              f"Spurious={n_spur:>5} ({100*n_spur/n:.1f}%)")

    # 2. Get models
    models = get_models()

    # 3. Run each strategy
    all_results = {}
    all_trained = {}
    all_roc_data = {}

    _, feature_cols = prepare_features(df_gt.copy(), spectral_cols)
    print(f"\n[INFO] Track principal com {len(feature_cols)} features: {feature_cols}")

    for strat_name, config in LABEL_STRATEGIES.items():
        if config['dataset'] == 'full':
            results, trained, roc_data = run_strategy_full_train(
                strat_name, config, df_full, df_gt, models, spectral_cols
            )
        else:
            results, trained, roc_data = run_strategy_cv(
                strat_name, config, df_gt, models, spectral_cols,
                n_folds=args.n_folds
            )
        all_results[strat_name] = results
        all_trained[strat_name] = trained
        all_roc_data[strat_name] = roc_data

    # 4. Build results table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Kappa against Human GT)")
    print("=" * 60)

    rows = []
    model_names = list(models.keys())
    for model_name in model_names:
        row = {'model': model_name}
        for strat_name in LABEL_STRATEGIES:
            m = all_results[strat_name][model_name]
            for metric in ['kappa', 'accuracy', 'f1', 'precision', 'recall', 'roc_auc',
                           'kappa_ci_lo', 'kappa_ci_hi']:
                row[f'{strat_name}_{metric}'] = m.get(metric, float('nan'))
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / 'decisive_results.csv', index=False)
    print(f"[OK] decisive_results.csv")

    # Print summary table
    print(f"\n{'Model':<22}", end='')
    for strat in LABEL_STRATEGIES:
        display = LABEL_STRATEGIES[strat]['display'].replace('\n', ' ')
        print(f"  {display:<18}", end='')
    print()
    print("-" * 100)
    for model_name in model_names:
        print(f"{model_name:<22}", end='')
        for strat in LABEL_STRATEGIES:
            k = all_results[strat][model_name]['kappa']
            print(f"  {k:>6.3f}            ", end='')
        print()

    # 5. Concordance matrix
    concordance, n_samples = compute_concordance_matrix(df_gt)
    concordance.to_csv(output_dir / 'label_concordance.csv')
    print(f"[OK] label_concordance.csv")

    # 6. Feature importance
    fi_df = extract_feature_importances(all_trained, feature_cols)
    fi_df.to_csv(output_dir / 'feature_importance_by_strategy.csv')
    print(f"[OK] feature_importance_by_strategy.csv")

    # 7. Plots
    print("\n--- Generating plots ---")
    plot_kappa_heatmap(all_results, output_dir)
    plot_concordance_heatmap(concordance, n_samples, output_dir)
    plot_feature_importance(fi_df, output_dir)
    plot_roc_curves(all_roc_data, output_dir)

    # 8. JSON report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_folds': args.n_folds,
            'feature_set': f'{args.features}feat',
            'spectral_cols': spectral_cols or [],
            'features': feature_cols,
            'n_features': len(feature_cols),
            'n_full_samples': len(df_full),
            'n_gt_samples': len(df_gt),
            'n_with_all_4_labels': int(df_gt['gt_painel'].notna().sum()),
        },
        'label_distributions': {},
        'results_by_strategy': {},
        'concordance_matrix': concordance.to_dict(),
        'best_per_strategy': {},
    }

    for strat_name, config in LABEL_STRATEGIES.items():
        report['results_by_strategy'][strat_name] = all_results[strat_name]

        # Best model per strategy (by kappa, excluding Dummy)
        best_model = max(
            [(m, r['kappa']) for m, r in all_results[strat_name].items()
             if m != 'DummyClassifier'],
            key=lambda x: x[1]
        )
        report['best_per_strategy'][strat_name] = {
            'model': best_model[0],
            'kappa': best_model[1],
        }

    with open(output_dir / 'decisive_experiment_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[OK] decisive_experiment_report.json")

    # 9. Final summary
    print("\n" + "=" * 60)
    print("BEST MODEL PER STRATEGY")
    print("=" * 60)
    for strat_name in LABEL_STRATEGIES:
        best = report['best_per_strategy'][strat_name]
        display = LABEL_STRATEGIES[strat_name]['display'].replace('\n', ' ')
        print(f"  {display:<30} -> {best['model']} (Kappa = {best['kappa']:.4f})")

    print(f"\nAll outputs saved to: {output_dir}")

    # 10. EXPANDED EXPERIMENT: Human annotation with 22 features
    if args.expanded:
        run_expanded_human_experiment(df_full, df_gt, models, output_dir, args.n_folds)

    print("Done!")


def run_expanded_human_experiment(df_full, df_gt, models, output_dir, n_folds):
    """Run human_annotation track with 22 features (13 base + 9 spectral).

    No tautology: human_label is integrated judgment, not a direct function of dNBR.
    Spectral features are merged from sentinel2_features.csv and painel_labeled_expanded.csv.
    """
    print("\n" + "=" * 60)
    print("EXPANDED EXPERIMENT: Human Annotation with 22 Features")
    print("(13 FIRMS/ERA5 + 9 Sentinel-2 spectral)")
    print("=" * 60)

    # Load spectral features
    if not SENTINEL2_CSV.exists():
        print(f"[ERROR] {SENTINEL2_CSV} not found. Cannot run expanded experiment.")
        return

    df_spectral = pd.read_csv(SENTINEL2_CSV)
    df_spectral['acq_datetime'] = pd.to_datetime(df_spectral['acq_datetime'], format='mixed')
    print(f"[OK] Spectral features: {len(df_spectral)} samples")

    # Merge spectral features onto df_gt by (lat, lon, date)
    df_spectral['_lat_r'] = df_spectral['latitude'].round(4)
    df_spectral['_lon_r'] = df_spectral['longitude'].round(4)
    df_spectral['_date'] = df_spectral['acq_datetime'].dt.strftime('%Y-%m-%d')

    df_gt_exp = df_gt.copy()
    df_gt_exp['_lat_r'] = df_gt_exp['latitude'].round(4)
    df_gt_exp['_lon_r'] = df_gt_exp['longitude'].round(4)
    df_gt_exp['_date'] = pd.to_datetime(df_gt_exp['acq_datetime'], format='mixed').dt.strftime('%Y-%m-%d')

    spectral_merge_cols = ['_lat_r', '_lon_r', '_date'] + FEATURE_COLS_SPECTRAL
    available_spectral = [c for c in FEATURE_COLS_SPECTRAL if c in df_spectral.columns]
    df_gt_exp = df_gt_exp.merge(
        df_spectral[['_lat_r', '_lon_r', '_date'] + available_spectral].drop_duplicates(),
        on=['_lat_r', '_lon_r', '_date'], how='left',
        suffixes=('', '_s2')
    )

    # Use merged spectral cols, prefer new ones
    for col in available_spectral:
        if f'{col}_s2' in df_gt_exp.columns:
            df_gt_exp[col] = df_gt_exp[f'{col}_s2'].combine_first(df_gt_exp[col])
            df_gt_exp.drop(columns=[f'{col}_s2'], inplace=True)

    # Clean temp cols
    for col in ['_lat_r', '_lon_r', '_date']:
        df_gt_exp.drop(columns=[col], inplace=True, errors='ignore')

    # Check spectral coverage
    n_with_spectral = df_gt_exp['dnbr_mean'].notna().sum()
    print(f"     Samples with spectral features: {n_with_spectral}/{len(df_gt_exp)}")

    # Filter to samples with spectral data available
    mask_spectral = df_gt_exp['dnbr_mean'].notna()
    df_gt_with = df_gt_exp[mask_spectral].copy()
    df_gt_without = df_gt_exp.copy()  # keep all for base comparison

    print(f"     Using {len(df_gt_with)} samples with complete spectral features")

    # Build expanded feature set
    feature_cols_expanded = FEATURE_COLS_BASE.copy()
    if 'month' not in df_gt_with.columns:
        df_gt_with['month'] = pd.to_datetime(df_gt_with['acq_datetime'], format='mixed').dt.month
    feature_cols_expanded.append('month')
    df_gt_with['is_dry_season'] = df_gt_with['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    feature_cols_expanded.append('is_dry_season')
    for col in available_spectral:
        if col in df_gt_with.columns:
            feature_cols_expanded.append(col)

    print(f"     Total features: {len(feature_cols_expanded)} "
          f"({len(FEATURE_COLS_BASE)} base + month + is_dry_season + {len(available_spectral)} spectral)")

    def run_variant(feat_cols, label):
        print(f"\n--- {label} ({len(feat_cols)} features) on {len(df_gt_with)} samples ---")
        results_v = {}
        roc_v = {}
        trained_v = {}

        X_all = df_gt_with[feat_cols].copy()
        for col in X_all.columns:
            if X_all[col].isna().any():
                X_all[col] = X_all[col].fillna(X_all[col].median())

        y_human = df_gt_with['human_label'].astype(int).values

        for model_name, (model_template, needs_scaling) in models.items():
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            all_y_true, all_y_pred, all_y_proba = [], [], []

            for train_idx, test_idx in skf.split(X_all, y_human):
                X_train = X_all.iloc[train_idx].copy()
                X_test = X_all.iloc[test_idx].copy()
                y_train = y_human[train_idx]
                y_test = y_human[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                model = clone(model_template)
                Xtr = X_train_s if needs_scaling else X_train.values
                Xte = X_test_s if needs_scaling else X_test.values

                model.fit(Xtr, y_train)
                y_pred = model.predict(Xte)
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                if hasattr(model, 'predict_proba'):
                    all_y_proba.extend(model.predict_proba(Xte)[:, 1])

            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_proba = np.array(all_y_proba) if all_y_proba else None

            metrics = evaluate_predictions(all_y_true, all_y_pred, all_y_proba)
            ci = bootstrap_kappa(all_y_true, all_y_pred)
            metrics['kappa_ci_lo'] = ci['ci_lo']
            metrics['kappa_ci_hi'] = ci['ci_hi']
            results_v[model_name] = metrics
            trained_v[model_name] = model
            if all_y_proba is not None:
                roc_v[model_name] = (all_y_true, all_y_proba)

            print(f"  {model_name:<22} Kappa={metrics['kappa']:.4f} "
                  f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]  "
                  f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}")

        return results_v, roc_v, trained_v

    # ------------------------------------------------------------------
    # Escada de ablacao das features derivadas das cenas
    # ------------------------------------------------------------------
    base_cols = FEATURE_COLS_BASE + ['month', 'is_dry_season']
    ladder = build_ablation_ladder(base_cols, available_spectral)

    print("\n" + "=" * 60)
    print("ESCADA DE ABLACAO")
    print("=" * 60)
    for key, label, cols in ladder:
        print(f"  {key:<16} {len(cols):>2} features  {label}")
    print(f"\n  Features de cena removidas na ablacao M9: {SCENE_DERIVED_COLS}")
    print(f"  Todos os niveis rodam sobre as mesmas {len(df_gt_with)} amostras.")

    ladder_results = {}
    ladder_trained = {}
    ladder_roc = {}
    for key, label, cols in ladder:
        res, roc, trained = run_variant(cols, f'{key}: {label}')
        ladder_results[key] = res
        ladder_trained[key] = trained
        ladder_roc[key] = roc

    # Nomes historicos, para nao quebrar quem consome estes objetos abaixo
    results_base = ladder_results['L0_base13']
    results_ablated = ladder_results['L3_no_dnbr_cols']
    results_m9 = ladder_results['L1_no_scene']
    results_expanded = ladder_results['L4_full']
    roc_expanded = ladder_roc['L4_full']
    trained_expanded = ladder_trained['L4_full']

    feature_cols_ablated = dict((k, c) for k, _, c in ladder)['L3_no_dnbr_cols']
    n_ablated = len(feature_cols_ablated)
    n_full = len(feature_cols_expanded)

    # Tabela da escada
    print("\n" + "=" * 78)
    print("KAPPA POR NIVEL DE ABLACAO (avaliado contra a referencia humana)")
    print(f"Subamostra unica: {len(df_gt_with)} amostras")
    print("=" * 78)
    header = f"{'Model':<20}" + ''.join(f"{k.split('_')[0]:>10}" for k, _, _ in ladder)
    print(header + f"{'D(L4-L1)':>11}")
    print("-" * 78)
    for model_name in models:
        if model_name == 'DummyClassifier':
            continue
        line = f"  {model_name:<18}"
        for key, _, _ in ladder:
            line += f"{ladder_results[key][model_name]['kappa']:>10.4f}"
        gap = (ladder_results['L4_full'][model_name]['kappa']
               - ladder_results['L1_no_scene'][model_name]['kappa'])
        print(line + f"{gap:>+11.4f}")

    print("\n  Leitura: L1 e o modelo sem nenhuma feature derivada das cenas pre e")
    print("  pos-fogo. A coluna D(L4-L1) e quanto do desempenho depende dessas cenas.")

    # CSV da escada, formato longo
    ladder_rows = []
    ladder_cols_by_key = {k: c for k, _, c in ladder}
    ladder_label_by_key = {k: lab for k, lab, _ in ladder}
    for key, _, cols in ladder:
        for model_name, m in ladder_results[key].items():
            ladder_rows.append({
                'level': key,
                'level_label': ladder_label_by_key[key],
                'n_features': len(cols),
                'model': model_name,
                'kappa': m['kappa'],
                'kappa_ci_lo': m['kappa_ci_lo'],
                'kappa_ci_hi': m['kappa_ci_hi'],
                'accuracy': m['accuracy'],
                'f1': m['f1'],
                'precision': m['precision'],
                'recall': m['recall'],
                'roc_auc': m.get('roc_auc', float('nan')),
                'n_samples': len(df_gt_with),
            })
    pd.DataFrame(ladder_rows).to_csv(output_dir / 'm9_ablation_ladder.csv', index=False)
    print(f"\n[OK] m9_ablation_ladder.csv")

    # Save expanded results (comparacao historica, agora com o nivel M9)
    expanded_rows = []
    for model_name in models:
        row = {
            'model': model_name,
            'kappa_13feat': results_base[model_name]['kappa'],
            'accuracy_13feat': results_base[model_name]['accuracy'],
            'f1_13feat': results_base[model_name]['f1'],
            'kappa_m9_no_scene': results_m9[model_name]['kappa'],
            'accuracy_m9_no_scene': results_m9[model_name]['accuracy'],
            'f1_m9_no_scene': results_m9[model_name]['f1'],
            f'kappa_{n_ablated}feat_no_dnbr': results_ablated[model_name]['kappa'],
            f'accuracy_{n_ablated}feat_no_dnbr': results_ablated[model_name]['accuracy'],
            f'f1_{n_ablated}feat_no_dnbr': results_ablated[model_name]['f1'],
            f'kappa_{n_full}feat_full': results_expanded[model_name]['kappa'],
            f'accuracy_{n_full}feat_full': results_expanded[model_name]['accuracy'],
            f'f1_{n_full}feat_full': results_expanded[model_name]['f1'],
            'kappa_delta_ablated_minus_13': results_ablated[model_name]['kappa'] - results_base[model_name]['kappa'],
            'kappa_delta_full_minus_ablated': results_expanded[model_name]['kappa'] - results_ablated[model_name]['kappa'],
            'kappa_delta_full_minus_13': results_expanded[model_name]['kappa'] - results_base[model_name]['kappa'],
            'kappa_delta_full_minus_m9': results_expanded[model_name]['kappa'] - results_m9[model_name]['kappa'],
        }
        expanded_rows.append(row)

    df_expanded = pd.DataFrame(expanded_rows)
    df_expanded.to_csv(output_dir / 'expanded_13vs22_comparison.csv', index=False)
    print(f"[OK] expanded_13vs22_comparison.csv (inclui o nivel de ablacao M9)")

    plot_ablation_ladder(ladder, ladder_results, models, len(df_gt_with), output_dir)

    # Feature importance for expanded model (best tree-based)
    fi_expanded = {}
    for model_name in ['LightGBM', 'Extra Trees', 'XGBoost', 'Random Forest']:
        if model_name in trained_expanded:
            model = trained_expanded[model_name]
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                imp = imp / imp.sum()
                fi_expanded[model_name] = imp

    if fi_expanded:
        fi_df = pd.DataFrame(fi_expanded, index=feature_cols_expanded)
        fi_df.to_csv(output_dir / 'feature_importance_22feat.csv')
        print(f"[OK] feature_importance_22feat.csv")

        best_model_name = max(fi_expanded.keys(),
                              key=lambda m: results_expanded[m]['kappa'])
        plot_feature_importance_22feat(fi_df, best_model_name, output_dir)

    # Save expanded report (includes no-dNBR ablation)
    expanded_report = {
        'n_samples': len(df_gt_with),
        'n_features_base': 13,
        'n_features_ablated_no_dnbr': len(feature_cols_ablated),
        'n_features_expanded': len(feature_cols_expanded),
        'feature_cols_base': FEATURE_COLS_BASE + ['month', 'is_dry_season'],
        'feature_cols_ablated': feature_cols_ablated,
        'feature_cols_expanded': feature_cols_expanded,
        'dnbr_proxy_cols_removed_in_ablation': DNBR_PROXY_COLS,
        'ablation_rationale': (
            'Removes features most directly aligned with the dNBR thumbnail the '
            'human annotator saw during blind labeling. Isolates whether the '
            '13->22 Kappa jump comes from generic spectral signal vs from a proxy '
            'of the same dNBR visual the annotator used to decide.'
        ),
        'results_13feat': results_base,
        'results_no_dnbr': results_ablated,
        'results_22feat': results_expanded,
        'm9_ablation': {
            'rationale': (
                'O dNBR e calculado a partir das '
                'cenas Sentinel-2 pre e pos-fogo, e essas mesmas cenas alimentam o '
                'NDVI e o NBR de cada janela. Remover so as colunas dNBR deixa no '
                'modelo variaveis que carregam a mesma informacao. A ablacao real '
                'remove as oito features derivadas dessas cenas, mantendo '
                'land_use_class, que vem do MapBiomas.'
            ),
            'scene_derived_cols_removed': SCENE_DERIVED_COLS,
            'kept_non_scene_spectral': NON_SCENE_SPECTRAL_COLS,
            'n_samples': len(df_gt_with),
            'levels': [
                {'key': k, 'label': lab, 'n_features': len(c), 'features': c,
                 'results': ladder_results[k]}
                for k, lab, c in ladder
            ],
        },
    }
    with open(output_dir / 'expanded_experiment_report.json', 'w') as f:
        json.dump(expanded_report, f, indent=2, default=str)
    print(f"[OK] expanded_experiment_report.json")


if __name__ == '__main__':
    main()
