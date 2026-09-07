"""
Baseline de limiar dNBR.

Compara o modelo com o dNBR, alem da comparacao com a referencia humana. O
script responde a pergunta na forma mais direta possivel: se o dNBR sozinho ja
resolve a tarefa, o classificador nao se justifica.

Compara quatro decisores contra a mesma referencia humana, na mesma subamostra:

  1. dNBR auto     rotulo automatico ja usado na dissertacao (manual_label),
                   regra fixa de limiar definida na extracao Sentinel-2
  2. Limiar otimo  melhor limiar unico sobre dnbr_mean e sobre dnbr_max,
                   escolhido POR VALIDACAO CRUZADA, nao no conjunto inteiro
  3. Limiar oraculo melhor limiar escolhido olhando todo o conjunto. E otimista
                   por construcao, serve de teto para o que um limiar pode dar
  4. Modelo        Regressao Logistica com 22 features, CV 5-fold

Saidas em outputs/decisive_experiment/:
  - baseline_dnbr_threshold.csv          tabela comparativa
  - baseline_dnbr_threshold_sweep.csv    varredura completa de limiares
  - baseline_dnbr_threshold.png          curva de Kappa por limiar

Uso:
    python scripts/baseline_dnbr_threshold.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
SENTINEL2_CSV = PROJECT_ROOT / 'data' / 'processed' / 'sentinel2_features.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'decisive_experiment'

FEATURES_13 = [
    'brightness', 'confidence', 'frp', 'hotspot_count',
    'persistence_score', 'temperature', 'dewpoint',
    'wind_speed', 'precipitation', 'rh', 'drying_index',
    'month', 'is_dry_season',
]
FEATURES_SPECTRAL = [
    'mean_ndvi_pre', 'mean_ndvi_post', 'delta_ndvi',
    'mean_nbr_pre', 'mean_nbr_post', 'dnbr_mean',
    'dnbr_max', 'pct_burned', 'land_use_class',
]
FEATURES_22 = FEATURES_13 + FEATURES_SPECTRAL

N_FOLDS = 5
SEED = 42

# Cortes de severidade de queimada de Key e Benson (2006), FIRESEV/FIREMON.
# Sao valores publicados, escolhidos sem olhar para estes dados. Servem para
# testar se o limiar so empata com o modelo por ter sido calibrado na amostra.
LIMIARES_LITERATURA = [
    ('limite nao queimado / baixa', 0.10),
    ('severidade baixa', 0.27),
    ('severidade moderada-baixa', 0.44),
    ('severidade moderada-alta', 0.66),
]

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


def bootstrap_kappa(y_true, y_pred, n_boot=2000, seed=SEED):
    """IC 95% de Cohen's Kappa por reamostragem com reposicao."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    ks = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        ks.append(cohen_kappa_score(y_true[idx], y_pred[idx]))
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))


def load_data():
    """GT humano com as features Sentinel-2 anexadas, restrito ao subset completo."""
    gt = pd.read_csv(GT_CSV)
    gt['acq_datetime'] = pd.to_datetime(gt['acq_datetime'], format='mixed')
    s2 = pd.read_csv(SENTINEL2_CSV)
    s2['acq_datetime'] = pd.to_datetime(s2['acq_datetime'], format='mixed')

    for d in (gt, s2):
        d['_lat_r'] = d['latitude'].round(4)
        d['_lon_r'] = d['longitude'].round(4)
        d['_date'] = d['acq_datetime'].dt.strftime('%Y-%m-%d')

    keys = ['_lat_r', '_lon_r', '_date']
    df = gt.merge(
        s2[keys + FEATURES_SPECTRAL].drop_duplicates(subset=keys),
        on=keys, how='left', suffixes=('', '_s2'),
    )
    for col in FEATURES_SPECTRAL:
        if f'{col}_s2' in df.columns:
            df[col] = df[f'{col}_s2'].combine_first(df[col])
            df.drop(columns=[f'{col}_s2'], inplace=True)
    df.drop(columns=keys, inplace=True, errors='ignore')

    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)

    df = df[df['human_label'].isin([0, 1])].copy()
    df = df.dropna(subset=FEATURES_SPECTRAL).copy()
    return df


def sweep_threshold(values, y_true, n_steps=200):
    """Varre limiares e devolve o Kappa de cada um.

    Convencao: 0 = Fogo Real, 1 = Espurio. dNBR alto indica cicatriz de queima,
    entao a regra e 'dNBR >= limiar -> Real (0)', caso contrario Espurio (1).
    """
    lo, hi = np.nanpercentile(values, 1), np.nanpercentile(values, 99)
    grid = np.linspace(lo, hi, n_steps)
    rows = []
    for t in grid:
        y_pred = (values < t).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        rows.append({
            'threshold': float(t),
            'kappa': float(cohen_kappa_score(y_true, y_pred)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            'pct_predicted_spurious': float(y_pred.mean()),
        })
    return pd.DataFrame(rows)


def threshold_cv(values, y_true, n_folds=N_FOLDS, seed=SEED):
    """Limiar escolhido no treino de cada fold e aplicado no teste.

    E a forma sem vazamento de estimar o baseline de limiar. Escolher o limiar
    sobre todo o conjunto e depois reportar o Kappa nesse mesmo conjunto
    superestima o baseline, porque o limiar viu o rotulo de todas as amostras.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_true = np.asarray(y_true)
    values = np.asarray(values)
    oof_pred = np.zeros(len(y_true), dtype=int)
    fold_thresholds = []

    for train_idx, test_idx in skf.split(values.reshape(-1, 1), y_true):
        sweep = sweep_threshold(values[train_idx], y_true[train_idx])
        best_t = float(sweep.loc[sweep['kappa'].idxmax(), 'threshold'])
        fold_thresholds.append(best_t)
        oof_pred[test_idx] = (values[test_idx] < best_t).astype(int)

    return oof_pred, fold_thresholds


def model_cv(df, features, n_folds=N_FOLDS, seed=SEED):
    """Regressao Logistica em CV, devolve as predicoes out-of-fold."""
    X = df[features].copy()
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    y = df['human_label'].astype(int).values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X.iloc[train_idx])
        Xte = scaler.transform(X.iloc[test_idx])
        clf = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
        clf.fit(Xtr, y[train_idx])
        oof[test_idx] = clf.predict(Xte)
    return oof


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 68)
    print('BASELINE DE LIMIAR dNBR vs MODELO')
    print('=' * 68)

    df = load_data()
    y = df['human_label'].astype(int).values
    print(f'Amostras com referencia humana e espectral completa: {len(df)}')
    print(f'  Real (0): {(y == 0).sum()}   Espurio (1): {(y == 1).sum()}')

    rows = []
    sweeps = {}

    # 1. dNBR auto, a regra fixa ja usada na dissertacao
    mask_auto = df['manual_label'].notna() & (df['manual_label'] != -1)
    y_auto = df.loc[mask_auto, 'manual_label'].astype(int).values
    y_ref_auto = df.loc[mask_auto, 'human_label'].astype(int).values
    k_auto = cohen_kappa_score(y_ref_auto, y_auto)
    lo, hi = bootstrap_kappa(y_ref_auto, y_auto)
    rows.append({
        'decisor': 'dNBR auto (regra fixa da extracao)',
        'tipo': 'limiar',
        'variavel': 'manual_label',
        'kappa': k_auto, 'kappa_ci_lo': lo, 'kappa_ci_hi': hi,
        'accuracy': accuracy_score(y_ref_auto, y_auto),
        'f1': f1_score(y_ref_auto, y_auto, pos_label=1, zero_division=0),
        'n': int(mask_auto.sum()),
        'limiar': '',
    })
    print(f'\n1. dNBR auto              Kappa = {k_auto:.4f} [{lo:.3f}, {hi:.3f}]  n={mask_auto.sum()}')

    # 2 e 3. Limiar unico sobre dnbr_mean e dnbr_max
    for var in ['dnbr_mean', 'dnbr_max']:
        vals = df[var].values

        sweep = sweep_threshold(vals, y)
        sweep['variavel'] = var
        sweeps[var] = sweep
        best = sweep.loc[sweep['kappa'].idxmax()]

        # Oraculo: limiar escolhido vendo todo o conjunto
        y_or = (vals < best['threshold']).astype(int)
        lo, hi = bootstrap_kappa(y, y_or)
        rows.append({
            'decisor': f'Limiar oraculo em {var}',
            'tipo': 'limiar',
            'variavel': var,
            'kappa': float(best['kappa']), 'kappa_ci_lo': lo, 'kappa_ci_hi': hi,
            'accuracy': float(best['accuracy']), 'f1': float(best['f1']),
            'n': len(df),
            'limiar': f"{best['threshold']:.4f}",
        })
        print(f"   Limiar oraculo {var:<10} Kappa = {best['kappa']:.4f} "
              f"[{lo:.3f}, {hi:.3f}]  limiar = {best['threshold']:.4f}")

        # CV: limiar escolhido no treino de cada fold
        y_cv, folds = threshold_cv(vals, y)
        k_cv = cohen_kappa_score(y, y_cv)
        lo, hi = bootstrap_kappa(y, y_cv)
        rows.append({
            'decisor': f'Limiar unico em {var} (CV)',
            'tipo': 'limiar',
            'variavel': var,
            'kappa': k_cv, 'kappa_ci_lo': lo, 'kappa_ci_hi': hi,
            'accuracy': accuracy_score(y, y_cv),
            'f1': f1_score(y, y_cv, pos_label=1, zero_division=0),
            'n': len(df),
            'limiar': ', '.join(f'{t:.3f}' for t in folds),
        })
        print(f"   Limiar CV      {var:<10} Kappa = {k_cv:.4f} [{lo:.3f}, {hi:.3f}]")

    # 4. Modelos
    for label, feats in [('Modelo LR + 13 features', FEATURES_13),
                         ('Modelo LR + 22 features', FEATURES_22)]:
        y_m = model_cv(df, feats)
        k_m = cohen_kappa_score(y, y_m)
        lo, hi = bootstrap_kappa(y, y_m)
        rows.append({
            'decisor': label,
            'tipo': 'modelo',
            'variavel': f'{len(feats)} features',
            'kappa': k_m, 'kappa_ci_lo': lo, 'kappa_ci_hi': hi,
            'accuracy': accuracy_score(y, y_m),
            'f1': f1_score(y, y_m, pos_label=1, zero_division=0),
            'n': len(df),
            'limiar': '',
        })
        print(f'\n4. {label:<24} Kappa = {k_m:.4f} [{lo:.3f}, {hi:.3f}]')

    # 4b. Limiares FIXOS da literatura, sem nenhum ajuste aos dados
    # Se um corte publicado, aplicado as cegas, ja empata com o modelo, cai o
    # argumento de que o limiar so funciona porque foi calibrado nos dados.
    print('\n' + '=' * 68)
    print('LIMIARES FIXOS DA LITERATURA (Key e Benson 2006), sem ajuste')
    print('=' * 68)
    for var in ['dnbr_mean', 'dnbr_max']:
        vals = df[var].values
        for nome_t, t in LIMIARES_LITERATURA:
            y_l = (vals < t).astype(int)
            if len(np.unique(y_l)) < 2:
                print(f'  {var:<10} {nome_t:<34} degenerado')
                continue
            k_l = cohen_kappa_score(y, y_l)
            lo, hi = bootstrap_kappa(y, y_l)
            rows.append({
                'decisor': f'Limiar fixo {t:.2f} em {var} ({nome_t})',
                'tipo': 'limiar_literatura',
                'variavel': var,
                'kappa': k_l, 'kappa_ci_lo': lo, 'kappa_ci_hi': hi,
                'accuracy': accuracy_score(y, y_l),
                'f1': f1_score(y, y_l, pos_label=1, zero_division=0),
                'n': len(df),
                'limiar': f'{t:.2f}',
            })
            print(f'  {var:<10} {nome_t:<34} t={t:.2f}  '
                  f'Kappa = {k_l:.4f} [{lo:.3f}, {hi:.3f}]')

    # 5. O modelo supera o limiar de forma significativa?
    # Bootstrap pareado da diferenca de Kappa, reamostrando as MESMAS amostras
    # para os dois decisores. E o teste que responde a pergunta do R10.
    print('\n' + '=' * 68)
    print('TESTE PAREADO: o modelo supera o limiar unico de dNBR?')
    print('=' * 68)

    y_thr_cv, _ = threshold_cv(df['dnbr_max'].values, y)
    y_model = model_cv(df, FEATURES_22)
    # Limiar fixo da literatura, sem nenhum ajuste. E o adversario mais duro,
    # porque nao consumiu informacao nenhuma do conjunto de dados.
    y_thr_lit = (df['dnbr_max'].values < 0.44).astype(int)

    def teste_pareado(y_ref, y_a, y_b, n_boot=5000):
        rng = np.random.RandomState(SEED)
        d = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(y_ref), size=len(y_ref))
            if len(np.unique(y_ref[idx])) < 2:
                continue
            d.append(cohen_kappa_score(y_ref[idx], y_a[idx])
                     - cohen_kappa_score(y_ref[idx], y_b[idx]))
        d = np.array(d)
        obs = cohen_kappa_score(y_ref, y_a) - cohen_kappa_score(y_ref, y_b)
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        return obs, lo, hi, p, d

    d_lit, lit_lo, lit_hi, p_lit, _ = teste_pareado(y, y_model, y_thr_lit)
    print(f'  Modelo contra o limiar FIXO 0,44 da literatura:')
    print(f'    Kappa limiar fixo       : {cohen_kappa_score(y, y_thr_lit):.4f}')
    print(f'    Diferenca               : {d_lit:+.4f}  '
          f'IC 95% [{lit_lo:+.4f}, {lit_hi:+.4f}]   p = {p_lit:.4f}')

    d_obs, d_lo, d_hi, p_two, diffs = teste_pareado(y, y_model, y_thr_cv)

    # McNemar sobre os acertos de cada decisor
    correct_model = (y_model == y)
    correct_thr = (y_thr_cv == y)
    b = int((correct_model & ~correct_thr).sum())
    c = int((~correct_model & correct_thr).sum())
    from scipy.stats import binomtest
    p_mcnemar = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0

    print(f'  Kappa modelo LR + 22 feat : {cohen_kappa_score(y, y_model):.4f}')
    print(f'  Kappa limiar dnbr_max (CV): {cohen_kappa_score(y, y_thr_cv):.4f}')
    print(f'  Diferenca                 : {d_obs:+.4f}  IC 95% [{d_lo:+.4f}, {d_hi:+.4f}]')
    print(f'  p bootstrap pareado       : {p_two:.4f}')
    print(f'  McNemar b={b} c={c}, p    : {p_mcnemar:.4f}')
    veredito = ('o modelo supera o limiar' if d_lo > 0 else
                'nao ha evidencia de que o modelo supere o limiar unico')
    print(f'  Veredito ao nivel de 5%   : {veredito}')

    with open(OUTPUT_DIR / 'baseline_dnbr_paired_test.json', 'w') as f:
        json.dump({
            'n': int(len(y)),
            'kappa_model_lr22': float(cohen_kappa_score(y, y_model)),
            'kappa_threshold_dnbr_max_cv': float(cohen_kappa_score(y, y_thr_cv)),
            'kappa_diff': float(d_obs),
            'kappa_diff_ci95': [float(d_lo), float(d_hi)],
            'p_paired_bootstrap': float(p_two),
            'mcnemar_b_model_only_correct': b,
            'mcnemar_c_threshold_only_correct': c,
            'p_mcnemar': float(p_mcnemar),
            'verdict': veredito,
            'literature_threshold': {
                'source': 'Key e Benson (2006), limite de severidade moderada-baixa',
                'variable': 'dnbr_max',
                'threshold': 0.44,
                'fitted_to_data': False,
                'kappa': float(cohen_kappa_score(y, y_thr_lit)),
                'kappa_diff_model_minus_threshold': float(d_lit),
                'kappa_diff_ci95': [float(lit_lo), float(lit_hi)],
                'p_paired_bootstrap': float(p_lit),
            },
        }, f, indent=2)
    print('[OK] baseline_dnbr_paired_test.json')

    out = pd.DataFrame(rows).sort_values('kappa', ascending=False)
    out.to_csv(OUTPUT_DIR / 'baseline_dnbr_threshold.csv', index=False)
    pd.concat(sweeps.values()).to_csv(
        OUTPUT_DIR / 'baseline_dnbr_threshold_sweep.csv', index=False)

    print('\n' + '=' * 68)
    print('RANKING (Kappa contra a referencia humana, n=%d)' % len(df))
    print('=' * 68)
    for _, r in out.iterrows():
        print(f"  {r['decisor']:<38} {r['kappa']:>7.4f}  "
              f"[{r['kappa_ci_lo']:.3f}, {r['kappa_ci_hi']:.3f}]")

    # Figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.81, 2.40))

    for var, color in [('dnbr_mean', '#1f3a5f'), ('dnbr_max', '#c8362d')]:
        s = sweeps[var]
        ax1.plot(s['threshold'], s['kappa'], color=color, linewidth=1.8, label=var)
        best = s.loc[s['kappa'].idxmax()]
        ax1.plot(best['threshold'], best['kappa'], 'o', color=color, markersize=8)

    k_model = float(out.loc[out['decisor'] == 'Modelo LR + 22 features', 'kappa'].iloc[0])
    ax1.axhline(k_model, color='#2d6a4f', linestyle='--', linewidth=1.6,
                label=f'Modelo LR + 22 feat ({k_model:.3f})'.replace('.', ','))
    ax1.axhline(k_auto, color='#8a6d3b', linestyle=':', linewidth=1.6,
                label=f'dNBR auto ({k_auto:.3f})'.replace('.', ','))
    ax1.set_xlabel('Limiar de dNBR')
    ax1.set_ylabel("Cohen's Kappa vs referencia humana")
    ax1.set_title('Varredura de limiar unico de dNBR')
    ax1.legend(fontsize=8, loc='lower center')
    ax1.grid(alpha=0.25)

    plot_df = out[out['decisor'] != ''].sort_values('kappa')
    colors = ['#2d6a4f' if t == 'modelo' else '#1f3a5f' for t in plot_df['tipo']]
    ypos = np.arange(len(plot_df))
    ax2.barh(ypos, plot_df['kappa'], color=colors, alpha=0.88)
    ax2.errorbar(plot_df['kappa'], ypos,
                 xerr=[plot_df['kappa'] - plot_df['kappa_ci_lo'],
                       plot_df['kappa_ci_hi'] - plot_df['kappa']],
                 fmt='none', ecolor='#333333', elinewidth=1.1, capsize=3)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(plot_df['decisor'], fontsize=9)
    ax2.set_xlabel("Cohen's Kappa vs referencia humana")
    ax2.set_title('Modelo comparado ao dNBR\n(barras de erro: IC 95% bootstrap)')
    ax2.grid(axis='x', alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'baseline_dnbr_threshold.png')
    plt.close()

    print('\n[OK] baseline_dnbr_threshold.csv')
    print('[OK] baseline_dnbr_threshold_sweep.csv')
    print('[OK] baseline_dnbr_threshold.png')


if __name__ == '__main__':
    main()
