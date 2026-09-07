"""
Calibracao temporal: verifica se o modelo recomendado (Regressao Logistica + 22 features)
mantem boa calibracao quando avaliado em periodos fora do treino.

Estrategia:
  1. CV 5-fold em 2022 (n=439) -> probabilidades out-of-fold para 2022.
  2. Treina em 2022 (n=439) -> probabilidades em holdout 2023+2024 (n=95).
  3. Combina os dois conjuntos (n=534) e estratifica por (ano, mes).
  4. Computa Brier score e BSS por estrato temporal.
  5. Gera figura e CSV.

Output:
  outputs/decisive_experiment/calibration_temporal.csv
  Dissertacao/.../figuras/cap5/calibration_temporal.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
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

from matplotlib.patches import Rectangle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
PAINEL_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'
OUT_CSV = PROJECT_ROOT / 'outputs' / 'decisive_experiment' / 'calibration_temporal.csv'
OUT_FIG = PROJECT_ROOT / 'dissertacao' / 'latex' / 'figuras' / 'cap5' / 'calibration_temporal.png'

FEATURES_22 = [
    'brightness', 'confidence', 'frp', 'hotspot_count', 'persistence_score',
    'temperature', 'dewpoint', 'wind_speed', 'precipitation', 'rh',
    'drying_index', 'month', 'is_dry_season',
    'mean_ndvi_pre', 'mean_ndvi_post', 'delta_ndvi',
    'mean_nbr_pre', 'mean_nbr_post', 'dnbr_mean',
    'dnbr_max', 'pct_burned', 'land_use_class',
]

NAVY = '#1a2b4a'
RED = '#c8362d'
GREEN = '#2d6a4f'
GRAY = '#6c757d'


def load_data():
    df = pd.read_csv(GT_CSV)
    df = df[df['human_label'].isin([0, 1])].copy().reset_index(drop=True)
    df['human_label'] = df['human_label'].astype(int)
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')
    painel = pd.read_csv(PAINEL_CSV)
    painel['acq_datetime'] = pd.to_datetime(painel['acq_datetime'], format='mixed')
    keys = ['latitude', 'longitude', 'acq_datetime']
    extra_cols = [c for c in FEATURES_22 if c in painel.columns and c not in df.columns]
    df = df.merge(painel[keys + extra_cols].drop_duplicates(keys), on=keys, how='left')
    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    df = df.dropna(subset=FEATURES_22).copy().reset_index(drop=True)
    df['year'] = df['acq_datetime'].dt.year
    df['ym'] = df['acq_datetime'].dt.strftime('%Y-%m')
    return df


def brier_skill_score(y_true, y_prob, prevalence=None):
    """BSS = 1 - Brier_model / Brier_baseline; baseline = preditor constante = prevalence."""
    brier = np.mean((y_prob - y_true) ** 2)
    if prevalence is None:
        prevalence = y_true.mean()
    baseline = prevalence * (1 - prevalence)  # variance of bernoulli = brier of constant prevalence preditor
    bss = 1 - brier / baseline if baseline > 0 else np.nan
    return brier, bss


def main():
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f'Total samples: {len(df)}')
    print(f'Year breakdown: {df.groupby("year").size().to_dict()}')

    model_cls = lambda: Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])

    # 1) OOF probs for 2022 via 5-fold CV
    df_2022 = df[df['year'] == 2022].copy().reset_index(drop=True)
    X22, y22 = df_2022[FEATURES_22].values, df_2022['human_label'].values
    oof_22 = np.zeros(len(df_2022))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X22, y22):
        clf = model_cls()
        clf.fit(X22[tr], y22[tr])
        oof_22[te] = clf.predict_proba(X22[te])[:, 1]
    df_2022['p_spurious'] = oof_22
    df_2022['split'] = 'cv_2022'

    # 2) Train on 2022, evaluate on 2023+24 holdout
    df_test = df[df['year'].isin([2023, 2024])].copy().reset_index(drop=True)
    clf_full = model_cls()
    clf_full.fit(X22, y22)
    df_test['p_spurious'] = clf_full.predict_proba(df_test[FEATURES_22].values)[:, 1]
    df_test['split'] = 'holdout_2324'

    df_all = pd.concat([df_2022, df_test], ignore_index=True)
    print(f'Total with probs: {len(df_all)}')

    # 3) BSS por estrato
    GLOBAL_PREV = df_all['human_label'].mean()
    print(f'Global prevalence (Spurious): {GLOBAL_PREV:.3f}')

    rows = []
    for (year, month), grp in df_all.groupby(['year', 'month']):
        if len(grp) < 8:
            continue
        brier, bss = brier_skill_score(grp['human_label'].values, grp['p_spurious'].values,
                                        prevalence=GLOBAL_PREV)
        rows.append({
            'year': int(year), 'month': int(month),
            'n': len(grp),
            'prev_local': round(grp['human_label'].mean(), 3),
            'mean_p_pred': round(grp['p_spurious'].mean(), 3),
            'brier': round(brier, 4),
            'bss_vs_global': round(bss, 3),
        })
    res = pd.DataFrame(rows).sort_values(['year', 'month'])
    res.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')
    print()
    print(res.to_string(index=False))

    # Overall metrics
    brier_all, bss_all = brier_skill_score(df_all['human_label'].values, df_all['p_spurious'].values)
    print(f'\nOverall: Brier={brier_all:.4f}, BSS={bss_all:.3f}, n={len(df_all)}')

    # 4) Plot. Canvas calibrado para a largura util da pagina.
    # A altura reserva, alem dos paineis, uma faixa inferior para a legenda,
    # que serve aos dois lados e fica fora da area de plotagem.
    FIG_H, FAIXA_LEGENDA = 2.77, 0.22
    fig, axes = plt.subplots(1, 2, figsize=(5.55, FIG_H),
                             gridspec_kw={'width_ratios': [2.2, 1]})

    # Left: BSS per (year, month) as grouped bar
    ax = axes[0]
    res['label'] = res.apply(lambda r: f'{int(r["year"])}-{int(r["month"]):02d}', axis=1)
    colors = [GREEN if y == 2022 else (RED if y == 2023 else NAVY) for y in res['year']]
    bars = ax.bar(res['label'], res['bss_vs_global'], color=colors, edgecolor='black', linewidth=0.5)

    # Annotate n per bar
    # O alinhamento vertical acompanha o sinal da barra. Sem isso o rotulo de
    # uma barra quase nula era cortado ao meio pela linha do zero.
    for b, n in zip(bars, res['n']):
        height = b.get_height()
        positiva = height >= 0
        ax.text(b.get_x() + b.get_width()/2,
                height + 0.02 if positiva else height - 0.02,
                f'n={n}', ha='center', va='bottom' if positiva else 'top',
                fontsize=6.5, color='#444')

    ax.axhline(0, color='black', linewidth=0.5)
    # Folga vertical suficiente para o rotulo n= da barra mais alta e da mais
    # baixa, que antes eram cortados pelas bordas do eixo.
    _b = res['bss_vs_global']
    _lo, _hi = min(float(_b.min()), 0.0), max(float(_b.max()), 0.0)
    _amp = max(_hi - _lo, 0.1)
    ax.set_ylim(_lo - 0.16 * _amp, _hi + 0.16 * _amp)
    ax.set_ylabel("Brier Skill Score", fontsize=6.6)
    ax.set_xlabel("Ano-mês de teste", fontsize=6.6)
    ax.set_title("BSS por mês: estabilidade temporal da calibração",
                 fontsize=7, color=NAVY, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=5.3)
    ax.tick_params(axis='y', labelsize=5.3)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=GREEN, label='2022 (treino, OOF CV)'),
               Patch(color=RED, label='2023 (holdout)'),
               Patch(color=NAVY, label='2024 (holdout)')]
    # A legenda vai para a faixa reservada abaixo dos dois paineis. Dentro do
    # eixo ela cobria as barras, e as cores do ano valem tambem para os pontos
    # do painel da direita, entao uma legenda unica serve aos dois.
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=5.3,
               frameon=False, columnspacing=1.6, handlelength=1.3,
               handletextpad=0.5, bbox_to_anchor=(0.5, 0.008))

    # Right: scatter of mean predicted prob vs local prevalence
    ax2 = axes[1]
    ax2.scatter(res['prev_local'], res['mean_p_pred'], s=res['n'] * 1.5,
                c=colors, edgecolor='black', linewidth=0.5, alpha=0.7)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    ax2.set_xlim(0, max(res['prev_local'].max(), res['mean_p_pred'].max()) + 0.05)
    ax2.set_ylim(0, max(res['prev_local'].max(), res['mean_p_pred'].max()) + 0.05)
    ax2.set_xlabel("Prevalência local de Espúrio", fontsize=6)
    ax2.set_ylabel("Probabilidade média prevista", fontsize=6)
    ax2.set_title("Calibração marginal por mês\n(área ∝ n)", fontsize=6.6, color=NAVY, fontweight='bold')
    ax2.tick_params(labelsize=5.3)
    ax2.grid(alpha=0.3)

    plt.tight_layout(rect=[0, FAIXA_LEGENDA / FIG_H, 1, 1])
    plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUT_FIG}')


if __name__ == '__main__':
    main()
