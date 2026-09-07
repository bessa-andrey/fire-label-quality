"""
Modelo de Dawid & Skene (1979) para estimar sensibilidade e especificidade de cada
fonte de rotulo, tratando todas as 5 fontes como observadores ruidosos de um rotulo
latente desconhecido. Estimativa via EM + 95% CIs por bootstrap nao-parametrico.

Convencao: 0 = Real, 1 = Espurio.

Fontes consideradas:
  - weak (is_reliable invertido)
  - dnbr_auto (manual_label, ja na convencao)
  - mapbiomas (gt_mapbiomas_fire invertido)
  - painel (gt_painel invertido)
  - humano (human_label, ja na convencao)

Parametros estimados por fonte s:
  sens_s = P(y_s = 1 | z = 1)  (true positive rate; "espurio segundo a fonte | espurio latente")
  spec_s = P(y_s = 0 | z = 0)  (true negative rate; "real segundo a fonte | real latente")
  prevalence = P(z = 1)        (proporcao do construto latente "espurio")

Output:
  outputs/decisive_experiment/dawid_skene_estimates.csv  (point + 95% IC)
  Dissertacao/.../figuras/cap5/dawid_skene_posteriors.png (densidades bootstrap)
"""

from __future__ import annotations
import argparse
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

from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
PAINEL_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_5strategies.csv'
OUT_CSV = PROJECT_ROOT / 'outputs' / 'decisive_experiment' / 'dawid_skene_estimates.csv'
OUT_FIG = PROJECT_ROOT / 'dissertacao' / 'latex' / 'figuras' / 'cap5' / 'dawid_skene_posteriors.png'

NAVY = '#1a2b4a'
RED = '#c8362d'
GRAY = '#6c757d'

SOURCE_NAMES = ['Weak Labels', 'dNBR Auto', 'MapBiomas Fogo', 'Painel do Fogo', 'Anotação Humana']
SOURCE_COLORS = ['#5f7e9f', '#7fa05c', '#c2825a', '#9b6fb0', RED]


def load_labels():
    """Retorna matriz Y (n_samples x n_sources) com convencao unificada 0=Real, 1=Espurio."""
    df = pd.read_csv(GT_CSV)
    df = df[df['human_label'].isin([0, 1])].copy().reset_index(drop=True)
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')

    painel = pd.read_csv(PAINEL_CSV)
    painel['acq_datetime'] = pd.to_datetime(painel['acq_datetime'], format='mixed')
    keys = ['latitude', 'longitude', 'acq_datetime']
    cols_to_merge = ['gt_painel']
    if 'gt_mapbiomas_fire' in painel.columns:
        cols_to_merge.append('gt_mapbiomas_fire')
    df = df.merge(painel[keys + cols_to_merge].drop_duplicates(keys), on=keys, how='left', suffixes=('', '_p'))

    # Recuperar valores duplicados
    for c in cols_to_merge:
        cm = f'{c}_p'
        if cm in df.columns:
            df[c] = df[cm].combine_first(df.get(c, pd.Series(dtype=float)))
            df.drop(columns=[cm], inplace=True, errors='ignore')

    # Aplicar convencao 0=Real, 1=Espurio
    Y = pd.DataFrame()
    Y['weak'] = 1 - df['is_reliable'].astype(float)  # inverte
    Y['dnbr_auto'] = df['manual_label'].astype(float)  # ja na convencao
    if 'gt_mapbiomas_fire' in df.columns:
        Y['mapbiomas'] = 1 - df['gt_mapbiomas_fire'].astype(float)
    Y['painel'] = 1 - df['gt_painel'].astype(float)
    Y['humano'] = df['human_label'].astype(float)

    # Subset onde TODAS as fontes existem
    Y = Y.dropna()
    print(f'Subset com todas as 5 fontes: n = {len(Y)}')
    return Y.values.astype(int), list(Y.columns)


def em_dawid_skene(Y, max_iter=200, tol=1e-6, rng=None):
    """EM para Dawid-Skene binario.

    Y: (n_samples, n_sources), entradas em {0, 1}.
    Retorna: prev, sens (n_sources,), spec (n_sources,), z_post (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n, S = Y.shape
    # Init: z_i = majority vote
    z_init = (Y.mean(axis=1) > 0.5).astype(float)
    # Soft initial assignment with smoothing
    q = np.column_stack([1 - z_init, z_init])
    q = 0.9 * q + 0.05  # smoothing
    q = q / q.sum(axis=1, keepdims=True)

    prev = q[:, 1].mean()
    sens = np.zeros(S)
    spec = np.zeros(S)

    prev_old = -1
    for it in range(max_iter):
        # M-step: sens, spec, prev
        # sens_s = P(Y_s = 1 | z = 1) = sum_i q[i,1] * 1[Y_is = 1] / sum_i q[i,1]
        sum_q1 = q[:, 1].sum()
        sum_q0 = q[:, 0].sum()
        for s in range(S):
            y_s = Y[:, s]
            sens[s] = (q[:, 1] * (y_s == 1)).sum() / max(sum_q1, 1e-8)
            spec[s] = (q[:, 0] * (y_s == 0)).sum() / max(sum_q0, 1e-8)
        # clamp for stability
        sens = np.clip(sens, 1e-5, 1 - 1e-5)
        spec = np.clip(spec, 1e-5, 1 - 1e-5)
        prev = np.clip(sum_q1 / n, 1e-5, 1 - 1e-5)

        # E-step: q[i, c] proportional to prev_c * prod_s P(Y_is | z=c)
        # log-space for stability
        logp1 = np.log(prev) * np.ones(n)
        logp0 = np.log(1 - prev) * np.ones(n)
        for s in range(S):
            y_s = Y[:, s]
            # if z=1: P(y_s) = sens if y_s=1 else (1-sens)
            logp1 += np.where(y_s == 1, np.log(sens[s]), np.log(1 - sens[s]))
            logp0 += np.where(y_s == 0, np.log(spec[s]), np.log(1 - spec[s]))
        m = np.maximum(logp1, logp0)
        a = np.exp(logp1 - m)
        b = np.exp(logp0 - m)
        q[:, 1] = a / (a + b)
        q[:, 0] = b / (a + b)

        if abs(prev - prev_old) < tol:
            break
        prev_old = prev

    return prev, sens, spec, q[:, 1]


def bootstrap_ds(Y, n_boot=500, seed=42):
    """Bootstrap nao-parametrico de DS-EM. Retorna distribuicoes posteriores aproximadas."""
    rng = np.random.default_rng(seed)
    n, S = Y.shape
    boots = {'prev': [], 'sens': [], 'spec': []}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        Yb = Y[idx]
        try:
            prev, sens, spec, _ = em_dawid_skene(Yb, max_iter=80, rng=rng)
            boots['prev'].append(prev)
            boots['sens'].append(sens)
            boots['spec'].append(spec)
        except Exception:
            pass
        if (b + 1) % 100 == 0:
            print(f'  bootstrap {b+1}/{n_boot}')
    return {
        'prev': np.array(boots['prev']),
        'sens': np.array(boots['sens']),
        'spec': np.array(boots['spec']),
    }


def main():
    # Insumos e saidas alternativos. Usados pela recomputacao da Estrategia 1
    # (src/preprocessing/recompute_weak_labels_mcd64a1.py), que grava copias do painel e da
    # referencia humana com is_reliable recomputado a partir do MCD64A1. Sem estas opcoes o
    # script roda exatamente como antes.
    global GT_CSV, PAINEL_CSV, OUT_CSV, OUT_FIG
    ap = argparse.ArgumentParser(description='Dawid & Skene sobre as cinco fontes de rotulo')
    ap.add_argument('--gt-csv', type=Path, default=None)
    ap.add_argument('--painel-csv', type=Path, default=None)
    ap.add_argument('--out-csv', type=Path, default=None)
    ap.add_argument('--out-fig', type=Path, default=None)
    args = ap.parse_args()
    if args.gt_csv:
        GT_CSV = args.gt_csv
    if args.painel_csv:
        PAINEL_CSV = args.painel_csv
    if args.out_csv:
        OUT_CSV = args.out_csv
    if args.out_fig:
        OUT_FIG = args.out_fig

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    Y, col_names = load_labels()
    print(f'Fontes: {col_names}')

    # Point estimates
    print('\n--- EM (point estimates) ---')
    prev, sens, spec, z_post = em_dawid_skene(Y, max_iter=300)
    print(f'Prevalencia latente (Espurio): {prev:.3f}')
    for i, s in enumerate(col_names):
        print(f'  {s:15s}: sens={sens[i]:.3f}, spec={spec[i]:.3f}')

    # Bootstrap
    print('\n--- Bootstrap (500 resamples) ---')
    boots = bootstrap_ds(Y, n_boot=500)

    # Construct results DataFrame
    rows = []
    for i, s in enumerate(col_names):
        sens_dist = boots['sens'][:, i]
        spec_dist = boots['spec'][:, i]
        rows.append({
            'source': s,
            'sensitivity': round(sens[i], 3),
            'sens_ci_lo': round(np.percentile(sens_dist, 2.5), 3),
            'sens_ci_hi': round(np.percentile(sens_dist, 97.5), 3),
            'specificity': round(spec[i], 3),
            'spec_ci_lo': round(np.percentile(spec_dist, 2.5), 3),
            'spec_ci_hi': round(np.percentile(spec_dist, 97.5), 3),
            'youdens_j': round(sens[i] + spec[i] - 1, 3),  # diagnostic accuracy index
        })
    res = pd.DataFrame(rows)
    res['prev_latent'] = round(prev, 3)
    res['prev_ci_lo'] = round(np.percentile(boots['prev'], 2.5), 3)
    res['prev_ci_hi'] = round(np.percentile(boots['prev'], 97.5), 3)
    res.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')
    print(res.to_string(index=False))

    # Figure: bootstrap posterior densities of sensitivity and specificity per
    # source. Canvas calibrado para a largura util da pagina,
    # sem sobrescrever os tamanhos de fonte ja calibrados no topo do arquivo.
    # Geometria em polegadas. Cada painel tem uma faixa propria reservada
    # abaixo dele para a legenda, que sai da area de plotagem: no topo do eixo
    # ela cobria a cauda alta das densidades e o pico degenerado do MapBiomas.
    FIG_W = 5.55
    esq, dir_, vao = 0.50, 0.06, 0.30
    pw = (FIG_W - esq - dir_ - vao) / 2
    ph = 1.55
    topo = 0.58          # titulo geral e titulo de painel de duas linhas
    base = 0.98          # rotulos do eixo x mais a faixa da legenda
    FIG_H = topo + ph + base

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = []
    for i in range(2):
        rect = [(esq + i * (pw + vao)) / FIG_W, base / FIG_H, pw / FIG_W, ph / FIG_H]
        ax = fig.add_axes(rect, sharey=axes[0] if axes else None)
        if i:
            ax.tick_params(labelleft=False)
        axes.append(ax)

    point = {'sens': sens, 'spec': spec}
    for ax, metric, title in [
        (axes[0], 'sens',
         'Sensibilidade\n$P($fonte $=$ "Espúrio" $\\mid$ latente $=$ "Espúrio"$)$'),
        (axes[1], 'spec',
         'Especificidade\n$P($fonte $=$ "Real" $\\mid$ latente $=$ "Real"$)$'),
    ]:
        for i, name in enumerate(col_names):
            dist = boots[metric][:, i]
            # Histograma em contorno preenchido (mais legivel quando ha sobreposicao)
            ax.hist(dist, bins=30, alpha=0.45, density=True,
                    color=SOURCE_COLORS[i], histtype='stepfilled',
                    edgecolor=SOURCE_COLORS[i], linewidth=1.4,
                    label=f'{SOURCE_NAMES[i]} ({point[metric][i]:.2f})'.replace('.', ','))
            # Linha vertical no estimador pontual de cada fonte
            ax.axvline(point[metric][i], color=SOURCE_COLORS[i],
                       linewidth=1.8, linestyle='--', alpha=0.9)
        ax.set_xlabel(metric.capitalize() + '  (0 = nunca acerta, 1 = sempre acerta)', fontsize=7)
        ax.set_title(title, fontsize=6.6, color=NAVY, fontweight='bold')
        ax.set_xlim(0, 1)
        # Limita o eixo y para revelar a dispersao bootstrap; picos degenerados
        # (ex.: MapBiomas com sens = 1,00) ficam indicados pela linha tracejada.
        ax.set_ylim(0, 30)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=6.3)
        # A legenda de cada painel carrega a estimativa pontual daquela
        # metrica, entao os dois paineis nao podem dividir uma legenda so.
        # Cada uma vai para a faixa reservada abaixo do seu proprio eixo.
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.31), ncol=2,
                  fontsize=5.6, framealpha=1.0, borderaxespad=0.0,
                  columnspacing=1.0, handlelength=1.4, handletextpad=0.5,
                  title='Fonte (estimativa pontual)', title_fontsize=5.6)
    axes[0].set_ylabel('Densidade (bootstrap)', fontsize=7)

    fig.text(0.5, 1 - 0.06 / FIG_H,
             'Perfil diagnóstico de cada fonte de rótulo (modelo de Dawid-Skene)',
             ha='center', va='top', fontsize=8.5, color=NAVY, fontweight='bold')
    plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUT_FIG}')


if __name__ == '__main__':
    main()
