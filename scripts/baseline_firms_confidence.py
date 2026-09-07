"""
Baseline simples: usar apenas a coluna FIRMS confidence como classificador.
Regra: se confidence < threshold, classifica como Espurio (1); caso contrario, Real (0).
Varre thresholds de 30 a 90 e reporta Kappa, acuracia, F1 etc.

Compara contra a Regressao Logistica + 22 features (modelo recomendado).

Responde a pergunta: "Quanto o ML adiciona sobre uma regra trivial?"

Output:
- outputs/decisive_experiment/baseline_firms_confidence.csv
- outputs/decisive_experiment/baseline_firms_confidence.png

Usage:
    python scripts/baseline_firms_confidence.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Tamanho de fonte calibrado para o tamanho impresso.
# A figura e gerada na largura que ocupara na pagina, entao o valor declarado
# aqui e o que sai no papel. Corpo do documento: 12 pt.
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.titlesize': 10.5,
    'lines.linewidth': 1.3, 'axes.linewidth': 0.7,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def load_data():
    main_df = pd.read_csv(GT_CSV)
    df = main_df[main_df['human_label'].isin([0, 1])].copy()
    df['human_label'] = df['human_label'].astype(int)
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')

    painel = pd.read_csv(PAINEL_CSV)
    painel['acq_datetime'] = pd.to_datetime(painel['acq_datetime'], format='mixed')
    keys = ['latitude', 'longitude', 'acq_datetime']
    extra_cols = [c for c in FEATURES_SPECTRAL if c in painel.columns]
    df = df.merge(painel[keys + extra_cols].drop_duplicates(keys), on=keys, how='left')
    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    df = df.dropna(subset=FEATURES_22).copy()
    return df


def baseline_threshold(confidence, threshold):
    """Regra: se conf < threshold -> Espurio (1); senao -> Real (0)."""
    return (confidence < threshold).astype(int)


def bootstrap_kappa_ci(y_true, y_pred, n_boot=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boots.append(cohen_kappa_score(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def lr_22feat_cv_kappa(df, random_state=42, n_splits=5, n_boot=1000):
    """Kappa da LR + 22 features por CV, com IC 95% via bootstrap real das
    predicoes out-of-fold. Mesma metodologia de src/models/learning_curve.py
    (Pipeline StandardScaler + LogisticRegression, StratifiedKFold)."""
    X = df[FEATURES_22].values
    y = df['human_label'].astype(int).values
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(pipe, X, y, cv=skf)
    kappa = cohen_kappa_score(y, y_pred)
    ci_lo, ci_hi = bootstrap_kappa_ci(y, y_pred, n_boot=n_boot, random_state=random_state)
    return kappa, ci_lo, ci_hi, y, y_pred


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 60)
    print('BASELINE FIRMS CONFIDENCE vs LR + 22 feat')
    print('=' * 60)

    df = load_data()
    y_true = df['human_label'].astype(int).values
    confidence = df['confidence'].values
    print(f'N amostras: {len(df)} ({(y_true==0).sum()} Real, {(y_true==1).sum()} Espurio)')
    print(f'Confidence range: [{confidence.min()}, {confidence.max()}]')

    # Varre thresholds
    thresholds = [30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95]
    rows = []
    for thr in thresholds:
        y_pred = baseline_threshold(confidence, thr)
        kappa = cohen_kappa_score(y_true, y_pred)
        acc = (y_pred == y_true).mean()
        k_lo, k_hi = bootstrap_kappa_ci(y_true, y_pred, n_boot=500)
        report = classification_report(y_true, y_pred, labels=[0, 1],
                                       output_dict=True, zero_division=0)
        rows.append({
            'metodo': f'FIRMS conf < {thr}',
            'threshold': thr,
            'kappa': round(kappa, 4),
            'kappa_ci_lo': round(k_lo, 4),
            'kappa_ci_hi': round(k_hi, 4),
            'acuracia': round(acc, 4),
            'precision_real': round(report['0']['precision'], 4),
            'recall_real': round(report['0']['recall'], 4),
            'precision_esp': round(report['1']['precision'], 4),
            'recall_esp': round(report['1']['recall'], 4),
            'f1_esp': round(report['1']['f1-score'], 4),
        })

    # Para referencia, kappa do LR 22 feat por CV 5-fold, com IC 95% do
    # bootstrap real das predicoes out-of-fold (nao um valor fixo em codigo).
    lr_kappa, lr_ci_lo, lr_ci_hi, lr_y_true, lr_y_pred = lr_22feat_cv_kappa(df)
    lr_report = classification_report(lr_y_true, lr_y_pred, labels=[0, 1],
                                       output_dict=True, zero_division=0)
    rows.append({
        'metodo': 'LR + 22 features (CV)',
        'threshold': None,
        'kappa': round(lr_kappa, 4),
        'kappa_ci_lo': round(lr_ci_lo, 4),
        'kappa_ci_hi': round(lr_ci_hi, 4),
        'acuracia': round((lr_y_pred == lr_y_true).mean(), 4),
        'precision_real': round(lr_report['0']['precision'], 4),
        'recall_real': round(lr_report['0']['recall'], 4),
        'precision_esp': round(lr_report['1']['precision'], 4),
        'recall_esp': round(lr_report['1']['recall'], 4),
        'f1_esp': round(lr_report['1']['f1-score'], 4),
    })

    df_out = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'baseline_firms_confidence.csv'
    df_out.to_csv(out_csv, index=False)

    # Print
    print(f'\n--- Resultados ---')
    print(df_out[['metodo', 'kappa', 'kappa_ci_lo', 'kappa_ci_hi',
                  'acuracia', 'recall_esp', 'precision_esp']].to_string(index=False))
    print(f'\nSalvo: {out_csv}')

    # Identificar melhor threshold
    df_thresholds = df_out[df_out['threshold'].notna()]
    best_thr = df_thresholds.loc[df_thresholds['kappa'].idxmax()]
    print(f'\n--- Melhor threshold: conf < {int(best_thr["threshold"])} ---')
    print(f'  Kappa = {best_thr["kappa"]:.4f}')
    print(f'  vs LR+22feat: Kappa = {lr_kappa:.4f}')
    gain = lr_kappa - best_thr['kappa']
    print(f'  Ganho do ML: +{gain:.4f} ({gain/best_thr["kappa"]*100:.1f}% relativo)')

    # Plot. Usa os tamanhos de fonte calibrados no topo do arquivo, ja
    # proporcionais ao tamanho impresso, sem sobrescrever com
    # valores maiores que forcavam o layout a transbordar.
    NAVY = '#1f3a5f'
    BRICK = '#8a3324'
    fig, ax = plt.subplots(figsize=(4.93, 3.4))
    df_t = df_out[df_out['threshold'].notna()].copy()
    thrs = df_t['threshold'].values
    kappas = df_t['kappa'].values
    los = df_t['kappa_ci_lo'].values
    his = df_t['kappa_ci_hi'].values

    best_base = max(float(kappas.max()), 0.0)
    gain = lr_kappa - best_base

    # Faixa que representa o ganho do modelo sobre o melhor baseline
    ax.axhspan(best_base, lr_kappa, color=NAVY, alpha=0.06, zorder=0)

    # Linha do acaso (kappa = 0)
    ax.axhline(0, color='gray', linewidth=1.2, linestyle=':', zorder=1)
    ax.text(thrs.min(), 0.015, 'Nível do acaso ($\\kappa = 0$)', color='gray',
            va='bottom')

    # Baseline trivial
    ax.plot(thrs, kappas, 'o-', color=BRICK, linewidth=2.2, markersize=8,
            label='Baseline trivial: regra de confidence FIRMS', zorder=3)
    ax.fill_between(thrs, los, his, color=BRICK, alpha=0.15, zorder=2)

    # Modelo de referencia (LR + 22 features)
    lr_lbl = f'Modelo de referência: LR + 22 features ($\\kappa = {lr_kappa:.3f}$)'.replace('.', ',')
    ax.axhline(lr_kappa, color=NAVY, linewidth=2.4, linestyle='--',
               label=lr_lbl, zorder=3)
    # A faixa do IC do modelo cobre toda a largura do eixo. Limitada ao
    # intervalo dos dados, ela terminava em degrau no meio do grafico e parecia
    # um retangulo solto sobre o fundo.
    ax.axhspan(lr_ci_lo, lr_ci_hi, color=NAVY, alpha=0.12, zorder=2)

    # Anotacao do ganho com seta de duas pontas
    x_arrow = float(thrs[len(thrs) // 2])
    ax.annotate('', xy=(x_arrow, lr_kappa), xytext=(x_arrow, best_base),
                arrowprops=dict(arrowstyle='<->', color=NAVY, lw=1.8))
    ax.text(x_arrow + 1.5, (best_base + lr_kappa) / 2,
            f'Ganho do modelo\n$\\approx +{gain:.2f}$'.replace('.', ','),
            color=NAVY, fontweight='bold', va='center')

    ax.set_xlabel('Limiar de confidence FIRMS (%)')
    ax.set_ylabel("Kappa ($\\kappa$) vs. referência humana")
    ax.set_title('O ganho do aprendizado de máquina sobre uma regra trivial',
                 color=NAVY, fontweight='bold')
    ax.set_ylim(-0.18, 0.62)
    ax.grid(alpha=0.3)
    # A legenda vai abaixo do eixo. Dentro da area de dados ela cobria a linha
    # do modelo de referencia e a faixa do IC, que sao justamente o que a figura
    # quer mostrar.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
              framealpha=0.95, borderaxespad=0.0)
    fig.tight_layout()
    out_png = OUT_DIR / 'baseline_firms_confidence.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {out_png}')


if __name__ == '__main__':
    main()
