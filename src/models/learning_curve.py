"""
Curva de aprendizado: Regressao Logistica com 22 features, treinada com
fracoes crescentes da amostra (20, 40, 60, 80, 100%).

Dois desenhos rodam lado a lado.

  cv_subamostra  Desenho original do texto. Sub-amostra a base inteira e mede
                 Kappa em CV 5-fold DENTRO da sub-amostra, de modo que treino e
                 teste encolhem juntos. Nas fracoes pequenas o Kappa e medido em
                 poucas dezenas de espurios e a estimativa fica dominada por
                 ruido. Mantido para continuidade com os numeros publicados.

  teste_fixo     Desenho correto para uma curva de aprendizado. Separa 30% como
                 conjunto de teste, treina com fracoes crescentes dos 70%
                 restantes e mede sempre no MESMO teste. So o treino encolhe.

Sementes: 30 em cada desenho. Com as 5 originais a dispersao entre particoes
era grande o bastante para achatar a curva, e o ponto de 20% publicado (0,444)
e um sorteio afortunado frente a media de 30 sementes.

O crescimento e testado com Wilcoxon pareado sobre as 30 repeticoes, comparando
100% contra 20%. Nao ha regra automatica declarando saturacao: identificar um
ponto de saturacao exige um teste que este desenho nao faz.

Output:
- outputs/decisive_experiment/learning_curve.csv            (cv_subamostra)
- outputs/decisive_experiment/learning_curve_fixed_test.csv (teste_fixo)
- outputs/decisive_experiment/learning_curve_growth_test.json
- outputs/decisive_experiment/learning_curve.png

Usage:
    python src/models/learning_curve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.3, 'axes.linewidth': 0.7,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

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


def bootstrap_ci_pooled(pairs, n_boot=2000, seed=42):
    """IC 95% bootstrap de Cohen's Kappa, agrupando as repeticoes de seed.

    Recebe uma lista de pares (y_true, y_pred) out-of-fold, um por seed.
    Em cada iteracao sorteia um seed e reamostra as amostras daquele seed com
    reposicao, de modo que o intervalo carrega as duas fontes de variacao,
    a incerteza amostral e a variacao entre particoes.

    Isto substitui o percentil p5/p95 calculado sobre cinco valores de seed,
    que media dispersao entre sementes e nao incerteza amostral.
    """
    rng = np.random.default_rng(seed)
    kappas = []
    for _ in range(n_boot):
        y_true, y_pred = pairs[rng.integers(0, len(pairs))]
        n = len(y_true)
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        kappas.append(cohen_kappa_score(y_true[idx], y_pred[idx]))
    return float(np.percentile(kappas, 2.5)), float(np.percentile(kappas, 97.5))


SEEDS = list(range(1000, 1030))     # 30 sementes
FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
TEST_SIZE = 0.30                     # fatia reservada no desenho de teste fixo


def _fit_predict(X_tr, y_tr, X_te, seed):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def _stratified_subsample(y, frac, rng):
    """Indices de uma sub-amostra estratificada, minimo de 5 por classe."""
    if frac >= 1.0:
        return np.arange(len(y))
    idx_real = np.where(y == 0)[0]
    idx_esp = np.where(y == 1)[0]
    idx = np.concatenate([
        rng.choice(idx_real, max(int(len(idx_real) * frac), 5), replace=False),
        rng.choice(idx_esp, max(int(len(idx_esp) * frac), 5), replace=False),
    ])
    rng.shuffle(idx)
    return idx


def kappa_at_fraction(X, y, frac, n_splits=5, seed=42):
    """Desenho cv_subamostra: sub-amostra, CV dentro dela, Kappa out-of-fold.

    Treino e teste encolhem juntos. Retorna (kappa, n, y_sub, y_pred).
    """
    rng = np.random.default_rng(seed)
    idx = _stratified_subsample(y, frac, rng)
    X_sub, y_sub = X[idx], y[idx]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = np.full_like(y_sub, fill_value=-99)
    for tr, te in skf.split(X_sub, y_sub):
        y_pred[te] = _fit_predict(X_sub[tr], y_sub[tr], X_sub[te], seed)
    return cohen_kappa_score(y_sub, y_pred), len(idx), y_sub, y_pred


def kappa_at_fraction_fixed_test(X, y, frac, seed=42, test_size=TEST_SIZE):
    """Desenho teste_fixo: 30% reservado como teste, treino cresce dentro dos 70%.

    O conjunto de teste depende so da semente, nao da fracao, entao os cinco
    pontos da curva sao medidos na MESMA base e ficam comparaveis entre si.
    Retorna (kappa, n_treino, y_teste, y_pred).
    """
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_i, te_i = next(splitter.split(X, y))
    X_tr, y_tr, X_te, y_te = X[tr_i], y[tr_i], X[te_i], y[te_i]

    rng = np.random.default_rng(seed)
    sub = _stratified_subsample(y_tr, frac, rng)
    y_pred = _fit_predict(X_tr[sub], y_tr[sub], X_te, seed)
    return cohen_kappa_score(y_te, y_pred), len(sub), y_te, y_pred


def run_curve(X, y, kappa_fn, label):
    """Roda um desenho em todas as fracoes e sementes.

    Retorna (DataFrame com uma linha por fracao, dict frac -> lista de kappas).
    """
    print(f'\n--- Desenho: {label} ---')
    per_seed = {f: [] for f in FRACTIONS}
    rows = []
    for frac in FRACTIONS:
        pairs = []
        n_eff = 0
        for seed in SEEDS:
            k, n, y_true, y_pred = kappa_fn(X, y, frac, seed=seed)
            per_seed[frac].append(k)
            pairs.append((y_true, y_pred))
            n_eff = n
        kappas = per_seed[frac]
        mean_k, std_k = float(np.mean(kappas)), float(np.std(kappas))
        seed_lo = float(np.percentile(kappas, 5))
        seed_hi = float(np.percentile(kappas, 95))
        boot_lo, boot_hi = bootstrap_ci_pooled(pairs)
        print(f'  frac={frac:.1f} (n={n_eff:>3d}): kappa = {mean_k:.4f} +/- {std_k:.4f}  '
              f'IC95 bootstrap [{boot_lo:.3f}, {boot_hi:.3f}]  '
              f'envelope de sementes [{seed_lo:.3f}, {seed_hi:.3f}]')
        rows.append({
            'design': label,
            'fraction': frac,
            'n_samples': n_eff,
            'n_seeds': len(SEEDS),
            'kappa_mean': round(mean_k, 4),
            'kappa_std': round(std_k, 4),
            'kappa_boot_ci_lo': round(boot_lo, 4),
            'kappa_boot_ci_hi': round(boot_hi, 4),
            'kappa_seed_envelope_p5': round(seed_lo, 4),
            'kappa_seed_envelope_p95': round(seed_hi, 4),
            'kappas_per_seed': str([round(v, 4) for v in kappas]),
        })
    return pd.DataFrame(rows), per_seed


def growth_test(per_seed, label):
    """Wilcoxon pareado sobre as sementes, 100% contra 20%, e o ultimo degrau."""
    a = np.array(per_seed[1.0])
    b = np.array(per_seed[0.2])
    stat = wilcoxon(a, b)
    delta_last = float(np.mean(per_seed[1.0]) - np.mean(per_seed[0.8]))
    stat_last = wilcoxon(np.array(per_seed[1.0]), np.array(per_seed[0.8]))
    res = {
        'design': label,
        'n_seeds': len(SEEDS),
        'delta_20_para_100': round(float(a.mean() - b.mean()), 4),
        'wilcoxon_p_20_para_100': float(stat.pvalue),
        'delta_80_para_100': round(delta_last, 4),
        'wilcoxon_p_80_para_100': float(stat_last.pvalue),
    }
    print(f'  {label}: delta(20% -> 100%) = {res["delta_20_para_100"]:+.4f}  '
          f'Wilcoxon pareado p = {res["wilcoxon_p_20_para_100"]:.2e}')
    print(f'  {label}: delta(80% -> 100%) = {res["delta_80_para_100"]:+.4f}  '
          f'Wilcoxon pareado p = {res["wilcoxon_p_80_para_100"]:.2e}')
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 68)
    print('CURVA DE APRENDIZADO (LR + 22 feat, 30 sementes, dois desenhos)')
    print('=' * 68)

    df = load_data()
    X = df[FEATURES_22].values
    y = df['human_label'].astype(int).values
    print(f'Total amostras: {len(df)} ({(y==0).sum()} Real, {(y==1).sum()} Espurio)')
    print(f'Fracoes: {FRACTIONS}')
    print(f'Sementes: {len(SEEDS)} ({SEEDS[0]} a {SEEDS[-1]})')

    df_cv, seeds_cv = run_curve(X, y, kappa_at_fraction, 'cv_subamostra')
    df_ft, seeds_ft = run_curve(X, y, kappa_at_fraction_fixed_test, 'teste_fixo')

    out_csv = OUT_DIR / 'learning_curve.csv'
    df_cv.to_csv(out_csv, index=False)
    print(f'\nSalvo: {out_csv}')
    out_csv_ft = OUT_DIR / 'learning_curve_fixed_test.csv'
    df_ft.to_csv(out_csv_ft, index=False)
    print(f'Salvo: {out_csv_ft}')

    print('\n--- Teste de crescimento (Wilcoxon pareado sobre as sementes) ---')
    growth = [growth_test(seeds_cv, 'cv_subamostra'), growth_test(seeds_ft, 'teste_fixo')]
    out_json = OUT_DIR / 'learning_curve_growth_test.json'
    out_json.write_text(json.dumps(growth, indent=2), encoding='utf-8')
    print(f'Salvo: {out_json}')

    # Plot: os dois desenhos no mesmo eixo
    fig, ax = plt.subplots(figsize=(4.53, 2.85))
    for frame, color, lbl in ((df_cv, '#1f3a5f', 'CV na sub-amostra (desenho original)'),
                              (df_ft, '#2d6a4f', 'Conjunto de teste fixo')):
        ns = frame['n_samples'].values
        means = frame['kappa_mean'].values
        ax.plot(ns, means, 'o-', color=color, linewidth=2, markersize=7, label=lbl)
        ax.fill_between(ns, frame['kappa_boot_ci_lo'].values, frame['kappa_boot_ci_hi'].values,
                        color=color, alpha=0.15)
        for n, m in zip(ns, means):
            ax.annotate(f'{m:.3f}', xy=(n, m), xytext=(0, 9), textcoords='offset points',
                        ha='center', fontsize=7, color=color)

    ax.axhline(0.364, color='#c8362d', linestyle='--', linewidth=1.5, alpha=0.8,
               label='Fleiss kappa entre voluntarios (0,364)')
    ax.set_xlabel('Tamanho da amostra de treino (n)')
    ax.set_ylabel("Cohen's Kappa (media de 30 sementes)")
    ax.set_title('Curva de aprendizado, Regressao Logistica + 22 features')
    ax.set_ylim(0, 0.7)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=6.5)
    plt.tight_layout()
    out_png = OUT_DIR / 'learning_curve.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {out_png}')


if __name__ == '__main__':
    main()
