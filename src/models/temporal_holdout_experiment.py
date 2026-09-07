"""
Temporal holdout experiment em DOIS cenarios:

Cenario A (intra-2022, 22 features):
  Treino: meses 1-8 (n~293), Teste: meses 9-11 (n~146)
  Mede generalizacao temporal DENTRO da temporada seca + features espectrais.

Cenario B (cross-year, 22 features):
  Treino: 2022 (n=439), Teste: 2023+2024 (n=95)
  Mede generalizacao para anos futuros. Alimenta a Tabela 4.13 (`tab:temporal_holdout`).

O Kappa de validacao cruzada usado como linha de base do Delta-kappa NAO e constante
neste script. Ele e lido da coluna `human_annotation_kappa` do CSV do experimento
decisivo com 22 features, a mesma fonte que alimenta a Tabela 4.7 (`tab:13vs22`) do
Capitulo 4. Ate set/2026 esses cinco valores eram um dicionario escrito a mao, e quatro
deles tinham envelhecido em relacao a fonte, o que publicou Delta-kappa errado na
Tabela 4.13 (`tab:temporal_holdout`). Nenhuma conferencia contra dado alcanca constante
escrita a mao: por isso a leitura passou a ser do CSV, e a falta de qualquer um dos
cinco modelos aborta a execucao.

Output:
- outputs/decisive_experiment/temporal_holdout.csv
- outputs/decisive_experiment/temporal_holdout_vs_cv.csv
- outputs/decisive_experiment/temporal_holdout.png

Usage:
    python src/models/temporal_holdout_experiment.py
    python src/models/temporal_holdout_experiment.py --decisive-csv <caminho>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
PAINEL_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'decisive_experiment'

# Fonte unica do Kappa de validacao cruzada com 22 features (coluna
# `human_annotation_kappa`, CV 5-fold estratificada sobre n=439). E o mesmo CSV que
# sustenta a coluna "22 full" da Tabela 4.7 do Capitulo 4. A rodada vigente e a da
# Estrategia 1 estrita, em outputs/weak_labels_v2/, com a rodada anterior mantida como
# alternativa porque a coluna do track humano e identica nas duas.
DECISIVE_22_CSV = PROJECT_ROOT / 'outputs' / 'weak_labels_v2' / 'decisive_22feat' / 'decisive_results.csv'
DECISIVE_22_CSV_ALT = PROJECT_ROOT / 'outputs' / 'decisive_experiment_22feat' / 'decisive_results.csv'

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


def load_cv_kappas_22(csv_path: Path) -> dict[str, float]:
    """Le o Kappa de CV com 22 features do CSV do experimento decisivo.

    Falha alto quando o CSV, a coluna ou qualquer um dos cinco modelos nao aparece,
    para que um resultado desatualizado nunca passe silenciosamente para a tabela.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f'CSV do experimento decisivo nao encontrado: {csv_path}. '
            f'Rode src/models/run_decisive_experiment.py --expanded ou aponte '
            f'--decisive-csv para {DECISIVE_22_CSV_ALT}.'
        )
    df = pd.read_csv(csv_path)
    for col in ('model', 'human_annotation_kappa'):
        if col not in df.columns:
            raise KeyError(f'Coluna ausente em {csv_path}: {col}')
    kappas = dict(zip(df['model'], df['human_annotation_kappa']))
    faltando = [m for m in get_models() if m not in kappas]
    if faltando:
        raise KeyError(f'Modelos ausentes em {csv_path}: {faltando}')
    return {m: float(kappas[m]) for m in get_models()}


def get_models():
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


def bootstrap_kappa_ci(y_true, y_pred, n_boot=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            k = cohen_kappa_score(y_true[idx], y_pred[idx])
            boots.append(k)
        except Exception:
            pass
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def prepare_data(features):
    main_df = pd.read_csv(GT_CSV)
    df = main_df[main_df['human_label'].isin([0, 1])].copy()
    df['human_label'] = df['human_label'].astype(int)
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')

    if any(f in FEATURES_SPECTRAL for f in features):
        painel = pd.read_csv(PAINEL_CSV)
        painel['acq_datetime'] = pd.to_datetime(painel['acq_datetime'], format='mixed')
        keys = ['latitude', 'longitude', 'acq_datetime']
        extra_cols = [c for c in FEATURES_SPECTRAL if c in painel.columns]
        df = df.merge(painel[keys + extra_cols].drop_duplicates(keys), on=keys, how='left')

    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    df['year'] = df['acq_datetime'].dt.year
    df = df.dropna(subset=features).copy()
    return df


def run_scenario(scenario_name, features, train_mask, test_mask, df):
    print(f'\n{"=" * 60}')
    print(f'CENARIO {scenario_name}')
    print(f'{"=" * 60}')

    train_df = df[train_mask]
    test_df = df[test_mask]
    print(f'Features: {len(features)}')
    print(f'Treino: n={len(train_df)} ({(train_df["human_label"]==0).sum()} Real, {(train_df["human_label"]==1).sum()} Espurio)')
    print(f'Teste:  n={len(test_df)} ({(test_df["human_label"]==0).sum()} Real, {(test_df["human_label"]==1).sum()} Espurio)')

    if len(test_df) < 15:
        print('Pulando: teste muito pequeno.')
        return []

    X_train = train_df[features].values
    y_train = train_df['human_label'].values
    X_test = test_df[features].values
    y_test = test_df['human_label'].values

    rows = []
    print('\nResultados:')
    for name, model in get_models().items():
        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        kappa = cohen_kappa_score(y_test, y_pred)
        acc = (y_pred == y_test).mean()
        klo, khi = bootstrap_kappa_ci(y_test, y_pred, n_boot=1000)
        report = classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
        rows.append({
            'scenario': scenario_name,
            'model': name,
            'n_features': len(features),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'kappa': round(kappa, 4),
            'kappa_ci_lo': round(klo, 4),
            'kappa_ci_hi': round(khi, 4),
            'accuracy': round(acc, 4),
            'f1_real': round(report['0']['f1-score'], 4),
            'f1_espurio': round(report['1']['f1-score'], 4),
            'precision_real': round(report['0']['precision'], 4),
            'recall_real': round(report['0']['recall'], 4),
            'precision_espurio': round(report['1']['precision'], 4),
            'recall_espurio': round(report['1']['recall'], 4),
        })
        print(f'  {name:25s} Kappa={kappa:+.3f} [{klo:+.3f}, {khi:+.3f}] Acc={acc:.3f} F1_Real={report["0"]["f1-score"]:.3f} F1_Esp={report["1"]["f1-score"]:.3f}')
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--decisive-csv', type=Path, default=DECISIVE_22_CSV,
        help='CSV do experimento decisivo com 22 features, fonte do Kappa de CV.',
    )
    return parser.parse_args()


def main(decisive_csv: Path = DECISIVE_22_CSV):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 60)
    print('TEMPORAL HOLDOUT (2 cenarios)')
    print('=' * 60)

    all_rows = []

    # Cenario A: intra-2022 com 22 features
    df_a = prepare_data(FEATURES_22)
    df_a['month'] = df_a['acq_datetime'].dt.month
    train_mask = (df_a['year'] == 2022) & (df_a['month'] <= 8)
    test_mask = (df_a['year'] == 2022) & (df_a['month'] >= 9)
    all_rows += run_scenario('A_intra2022_22feat', FEATURES_22, train_mask, test_mask, df_a)

    # Cenario B: cross-year com 22 features (viavel apos extracao GEE 2023+2024)
    df_b = prepare_data(FEATURES_22)
    train_mask = df_b['year'] == 2022
    test_mask = df_b['year'].isin([2023, 2024])
    all_rows += run_scenario('B_cross_year_22feat', FEATURES_22, train_mask, test_mask, df_b)

    df_out = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / 'temporal_holdout.csv'
    df_out.to_csv(out_csv, index=False)
    print(f'\nSalvo: {out_csv}')

    # Comparacao com baselines CV, lidos da fonte e nao escritos a mao
    cv_kappas_22 = load_cv_kappas_22(decisive_csv)
    print(f'\nKappa de CV lido de: {decisive_csv}')

    print('\n--- Comparacao com CV random ---')
    comparison = []
    for r in all_rows:
        if r['n_features'] != 22:
            raise ValueError(
                f'Cenario com {r["n_features"]} features sem linha de base de CV. '
                f'Os dois cenarios deste script rodam com 22 features. Para 13, leia a '
                f'coluna human_annotation_kappa do CSV do track de 13 features.'
            )
        cv_k = cv_kappas_22[r['model']]
        comparison.append({
            'scenario': r['scenario'],
            'model': r['model'],
            'cv_kappa': round(cv_k, 4),
            'temporal_kappa': r['kappa'],
            'delta': round(r['kappa'] - cv_k, 4),
        })
    comp_df = pd.DataFrame(comparison)
    comp_csv = OUT_DIR / 'temporal_holdout_vs_cv.csv'
    comp_df.to_csv(comp_csv, index=False)
    print(comp_df.to_string(index=False))
    print(f'\nSalvo: {comp_csv}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, scenario, cv_map, title in [
        (axes[0], 'A_intra2022_22feat', cv_kappas_22, 'Cenario A: intra-2022, 22 features\n(treino: meses 1-8 | teste: 9-11)'),
        (axes[1], 'B_cross_year_22feat', cv_kappas_22, 'Cenario B: cross-year, 22 features\n(treino: 2022 | teste: 2023+2024)'),
    ]:
        rows = [r for r in all_rows if r['scenario'] == scenario]
        if not rows:
            ax.set_title(title + '\n(sem dados)')
            continue
        models = [r['model'] for r in rows]
        cv_vals = [cv_map.get(m, 0) for m in models]
        temp_vals = [r['kappa'] for r in rows]
        x = np.arange(len(models))
        width = 0.38
        ax.bar(x - width/2, cv_vals, width, label='CV random 5-fold', color='#4a6fa5')
        ax.bar(x + width/2, temp_vals, width, label='Temporal holdout', color='#c8362d')
        for r in rows:
            i = models.index(r['model'])
            ax.errorbar(i + width/2, r['kappa'],
                        yerr=[[r['kappa']-r['kappa_ci_lo']], [r['kappa_ci_hi']-r['kappa']]],
                        fmt='none', color='black', capsize=3, alpha=0.7)
        ax.set_ylabel("Cohen's Kappa")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / 'temporal_holdout.png'
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f'Salvo: {out_png}')


if __name__ == '__main__':
    main(parse_args().decisive_csv)
