"""
Extended Concordance Analysis: p0, Cohen's Kappa, Gwet's AC1, prevalence and bias indices.

Addresses reviewer concern: Kappa can be low for structural reasons (prevalence paradox),
not only because labelers truly disagree. This script computes additional metrics
recommended by Byrt et al. (1993) and Gwet (2008).

A convencao de rotulagem entra por `--convencao`. O padrao e a convencao principal da
dissertacao, a `sensu_sensor`, sob a qual a saida e identica a de antes. A alternativa
`sensu_gestor` reprojeta a referencia humana e grava com sufixo proprio, de modo que uma
rodada alternativa nunca sobrescreve a saida que alimenta a figura do capitulo. A regra de
projecao mora em `src/preprocessing/labeling_convention.py`.

Usage:
    python src/models/compute_extended_concordance.py
    python src/models/compute_extended_concordance.py --convencao sensu_gestor
    python src/models/compute_extended_concordance.py --convencao sensu_gestor \
        --gestor-mascara uso_solo

Output (com <sufixo> vazio sob a convencao principal):
    outputs/decisive_experiment/extended_concordance<sufixo>.csv
    outputs/decisive_experiment/extended_concordance_report<sufixo>.json
    outputs/decisive_experiment/extended_concordance_heatmap<sufixo>.png
"""

import argparse
import json
import sys
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
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from preprocessing.labeling_convention import CONVENCOES, MASCARAS, aplicar_convencao
# Prefer the 5-strategies CSV (has gt_painel + gt_mapbiomas_fire); fallback to expanded.
PAINEL_EXPANDED_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_5strategies.csv'
PAINEL_EXPANDED_CSV_FALLBACK = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'
HUMAN_GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'decisive_experiment'

plt.rcParams.update({
    'axes.spines.top': False, 'axes.spines.right': False,
})


def gwet_ac1(y1, y2):
    """Compute Gwet's AC1 agreement coefficient.

    Less sensitive to prevalence paradox than Cohen's Kappa.
    Gwet (2008) "Computing inter-rater reliability and its variance
    in the presence of high agreement" British J Math Stat Psychol.
    """
    n = len(y1)
    # Contingency table
    a = np.sum((y1 == 0) & (y2 == 0))  # both Real
    b = np.sum((y1 == 0) & (y2 == 1))  # y1=Real, y2=Spur
    c = np.sum((y1 == 1) & (y2 == 0))  # y1=Spur, y2=Real
    d = np.sum((y1 == 1) & (y2 == 1))  # both Spurious

    # Observed agreement
    p0 = (a + d) / n

    # Marginal proportions
    pi1 = ((a + b) + (a + c)) / (2 * n)  # proportion of "Real" (class 0)
    pi2 = ((c + d) + (b + d)) / (2 * n)  # proportion of "Spurious" (class 1)

    # Expected chance agreement under Gwet's model
    pe = 2 * pi1 * (1 - pi1)

    # AC1
    if pe == 1.0:
        return 1.0 if p0 == 1.0 else 0.0
    ac1 = (p0 - pe) / (1 - pe)
    return float(ac1)


def prevalence_index(y1, y2):
    """Prevalence index (Byrt et al. 1993).
    Measures imbalance in the contingency table diagonal.
    PI = |a - d| / n, where a=both-positive, d=both-negative.
    High PI can deflate Kappa even with high agreement.
    """
    n = len(y1)
    a = np.sum((y1 == 0) & (y2 == 0))
    d = np.sum((y1 == 1) & (y2 == 1))
    return float(abs(a - d) / n)


def bias_index(y1, y2):
    """Bias index (Byrt et al. 1993).
    Measures asymmetry in off-diagonal cells.
    BI = |b - c| / n.
    High BI indicates systematic disagreement direction.
    """
    n = len(y1)
    b = np.sum((y1 == 0) & (y2 == 1))
    c = np.sum((y1 == 1) & (y2 == 0))
    return float(abs(b - c) / n)


def observed_agreement(y1, y2):
    """Raw observed agreement p0 = (a + d) / n."""
    n = len(y1)
    return float(np.sum(y1 == y2) / n)


def carregar_referencia(painel_csv=None, gt_csv=None, verbose=True):
    """Carrega a referencia humana binaria ja com as colunas do painel anexadas.

    Fonte unica do recorte de entrada da analise de concordancia. `sensibilidade_convencao`
    consome esta funcao para nao reimplementar o pareamento por (lat, lon, data).
    """
    painel_csv = painel_csv or PAINEL_EXPANDED_CSV
    gt_csv = gt_csv or HUMAN_GT_CSV
    # Load data (prefer 5strategies CSV for gt_painel + gt_mapbiomas_fire)
    csv_path = painel_csv if painel_csv.exists() else PAINEL_EXPANDED_CSV_FALLBACK
    df_full = pd.read_csv(csv_path)
    df_full['acq_datetime'] = pd.to_datetime(df_full['acq_datetime'], format='mixed')
    if verbose:
        print(f"[OK] Loading painel from: {csv_path.name}")

    df_gt_raw = pd.read_csv(gt_csv)
    df_gt_raw['acq_datetime'] = pd.to_datetime(df_gt_raw['acq_datetime'], format='mixed')
    df_gt = df_gt_raw[df_gt_raw['human_label'].isin([0, 1])].copy()

    # Merge to get gt_painel
    df_full['_lat_r'] = df_full['latitude'].round(4)
    df_full['_lon_r'] = df_full['longitude'].round(4)
    df_full['_date'] = df_full['acq_datetime'].dt.strftime('%Y-%m-%d')
    df_gt['_lat_r'] = df_gt['latitude'].round(4)
    df_gt['_lon_r'] = df_gt['longitude'].round(4)
    df_gt['_date'] = pd.to_datetime(df_gt['acq_datetime'], format='mixed').dt.strftime('%Y-%m-%d')

    merge_cols = ['_lat_r', '_lon_r', '_date', 'gt_painel']
    if 'gt_mapbiomas_fire' in df_full.columns:
        merge_cols.append('gt_mapbiomas_fire')
    # land_use_class so e consumida pela mascara `uso_solo` da convencao alternativa.
    if 'land_use_class' in df_full.columns:
        merge_cols.append('land_use_class')

    painel_cols = df_full[merge_cols].drop_duplicates()
    df_gt = df_gt.merge(painel_cols, on=['_lat_r', '_lon_r', '_date'], how='left',
                        suffixes=('', '_p'))
    for col in ['gt_painel', 'gt_mapbiomas_fire', 'land_use_class']:
        merged_col = f'{col}_p'
        if merged_col in df_gt.columns:
            df_gt[col] = df_gt[merged_col].combine_first(
                df_gt.get(col, pd.Series(dtype=float)))
            df_gt.drop(columns=[merged_col], inplace=True, errors='ignore')

    for col in ['_lat_r', '_lon_r', '_date']:
        df_gt.drop(columns=[col], inplace=True, errors='ignore')
        df_full.drop(columns=[col], inplace=True, errors='ignore')

    return df_gt, csv_path


def parse_args():
    """Permite apontar o script para insumos alternativos.

    Usado pela recomputacao da Estrategia 1 (`src/preprocessing/recompute_weak_labels_mcd64a1.py`),
    que grava copias do painel e da referencia humana com `is_reliable` recomputado a partir
    do MCD64A1. Sem estas opcoes o script roda exatamente como antes.
    """
    ap = argparse.ArgumentParser(description='Extended concordance analysis')
    ap.add_argument('--painel-csv', type=Path, default=None,
                    help='Substitui o painel de entrada (padrao: painel_labeled_5strategies.csv)')
    ap.add_argument('--gt-csv', type=Path, default=None,
                    help='Substitui a referencia humana (padrao: validation_manual_gt.csv)')
    ap.add_argument('--output-dir', type=Path, default=None,
                    help='Substitui o diretorio de saida')
    ap.add_argument('--convencao', choices=CONVENCOES, default='sensu_sensor',
                    help='Convencao de rotulagem aplicada a referencia humana. '
                         '`sensu_sensor` e a convencao principal da dissertacao e o padrao. '
                         '`sensu_gestor` reclassifica a combustao de manejo como espuria. '
                         'Ver src/preprocessing/labeling_convention.py')
    ap.add_argument('--gestor-mascara', choices=MASCARAS, default='notas',
                    help='Como identificar a combustao de manejo sob `sensu_gestor`. '
                         '`notas` usa a taxonomia registrada na anotacao cega, que e a '
                         'operacionalizacao declarada no Cap. 3. `uso_solo` usa a classe '
                         'MapBiomas do pixel, disponivel como analise de robustez')
    ap.add_argument('--sufixo', default=None,
                    help='Sufixo dos arquivos de saida. Por padrao, vazio sob a convencao '
                         'principal e derivado da convencao alternativa, de modo que uma '
                         'rodada alternativa nunca sobrescreve a saida principal')
    return ap.parse_args()


def main():
    global PAINEL_EXPANDED_CSV, HUMAN_GT_CSV, OUTPUT_DIR
    args = parse_args()
    if args.painel_csv:
        PAINEL_EXPANDED_CSV = args.painel_csv
    if args.gt_csv:
        HUMAN_GT_CSV = args.gt_csv
    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    if args.sufixo is not None:
        sufixo = args.sufixo
    elif args.convencao == 'sensu_sensor':
        sufixo = ''
    else:
        sufixo = f'_{args.convencao}_{args.gestor_mascara}'

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXTENDED CONCORDANCE ANALYSIS")
    print("p0, Cohen's Kappa, Gwet's AC1, Prevalence & Bias Indices")
    print(f"Convencao: {args.convencao}" +
          (f" (mascara: {args.gestor_mascara})" if args.convencao != 'sensu_sensor' else ""))
    print("=" * 60)

    df_gt, csv_path = carregar_referencia(PAINEL_EXPANDED_CSV, HUMAN_GT_CSV)

    # Projecao da referencia humana sobre o eixo da convencao pedida. Sob a convencao
    # principal (`sensu_sensor`) o rotulo sai intacto e o script roda como antes. As fontes
    # automaticas nao sao tocadas: elas medem cicatriz e medem a mesma coisa sob as duas
    # convencoes, logo os pares entre elas tem valor identico nas duas rodadas.
    df_gt['human_label_sensu_sensor'] = df_gt['human_label']
    rotulos, movidas = aplicar_convencao(
        df_gt, convencao=args.convencao, mascara=args.gestor_mascara)
    df_gt['human_label'] = rotulos
    n_movidas = int(movidas.sum())
    if args.convencao != 'sensu_sensor':
        prop = df_gt['human_label'].value_counts(normalize=True)
        prop_antes = df_gt['human_label_sensu_sensor'].value_counts(normalize=True)
        print(f"[OK] Convencao {args.convencao} (mascara {args.gestor_mascara}): "
              f"{n_movidas} de {len(df_gt)} amostras migraram de real para espurio")
        print(f"     Referencia sensu sensor: {prop_antes.get(0, 0):.1%} real / "
              f"{prop_antes.get(1, 0):.1%} espurio")
        print(f"     Referencia {args.convencao}: {prop.get(0, 0):.1%} real / "
              f"{prop.get(1, 0):.1%} espurio")

    has_mapbiomas = 'gt_mapbiomas_fire' in df_gt.columns and df_gt['gt_mapbiomas_fire'].notna().sum() > 0

    print(f"[OK] GT samples: {len(df_gt)}")
    print(f"     With gt_painel: {df_gt['gt_painel'].notna().sum()}")
    if has_mapbiomas:
        print(f"     With gt_mapbiomas_fire: {df_gt['gt_mapbiomas_fire'].notna().sum()}")

    # Label sources (all aligned to 0=Real, 1=Spurious)
    # gt_mapbiomas_fire: 1=burned (real), 0=not burned (spurious). Invert.
    # Ordem canonica do Capitulo 3: weak labels, dNBR, MapBiomas, Painel, humana.
    label_sources = {
        'Weak Labels': ('is_reliable', True),
        'dNBR Auto': ('manual_label', False),
    }
    if has_mapbiomas:
        label_sources['MapBiomas Fogo'] = ('gt_mapbiomas_fire', True)
    label_sources['Painel do Fogo'] = ('gt_painel', True)
    label_sources['Human'] = ('human_label', False)

    # Always-available filter (Weak, dNBR, Human)
    mask_3 = (
        df_gt['human_label'].isin([0, 1]) &
        df_gt['manual_label'].notna() & (df_gt['manual_label'] != -1) &
        df_gt['is_reliable'].notna()
    )
    df_3src = df_gt[mask_3].copy()
    print(f"     3 labels (Weak, dNBR, Human): {len(df_3src)}")

    # Compute all pairwise metrics. For each pair, use the maximum subset of rows
    # where both labels are available (lat/lon-date already merged onto df_3src).
    names = list(label_sources.keys())
    results = []

    print(f"\n{'Pair':<40} {'p0':>6} {'Kappa':>7} {'AC1':>7} {'PI':>6} {'BI':>6} {'n':>5}")
    print("-" * 80)

    optional_cols = {'gt_painel', 'gt_mapbiomas_fire'}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            col_i, inv_i = label_sources[ni]
            col_j, inv_j = label_sources[nj]

            # Start from the 3-source mask, then add NaN filters for optional cols
            pair_mask = mask_3.copy()
            if col_i in optional_cols and col_i in df_gt.columns:
                pair_mask = pair_mask & df_gt[col_i].notna()
            if col_j in optional_cols and col_j in df_gt.columns:
                pair_mask = pair_mask & df_gt[col_j].notna()

            df_sub = df_gt[pair_mask]
            if len(df_sub) < 10:
                print(f"  {ni} vs {nj}: SKIPPED (n={len(df_sub)} < 10)")
                continue

            vi = df_sub[col_i].astype(int).values
            vj = df_sub[col_j].astype(int).values
            li = 1 - vi if inv_i else vi
            lj = 1 - vj if inv_j else vj

            p0 = observed_agreement(li, lj)
            kappa = cohen_kappa_score(li, lj)
            ac1 = gwet_ac1(li, lj)
            pi = prevalence_index(li, lj)
            bi = bias_index(li, lj)
            n_samp = len(df_sub)

            # Marginal distributions
            prop_real_i = (li == 0).mean()
            prop_real_j = (lj == 0).mean()

            results.append({
                'source_1': ni, 'source_2': nj,
                'convencao': args.convencao,
                'gestor_mascara': args.gestor_mascara if args.convencao != 'sensu_sensor' else '',
                'n': n_samp,
                'p0': p0, 'kappa': kappa, 'ac1': ac1,
                'prevalence_index': pi, 'bias_index': bi,
                'pct_real_source1': prop_real_i,
                'pct_real_source2': prop_real_j,
            })

            print(f"  {ni} vs {nj:<25} {p0:>5.3f}  {kappa:>6.4f}  {ac1:>6.4f}  {pi:>5.3f}  {bi:>5.3f}  {n_samp:>4}")

    df_results = pd.DataFrame(results)
    nome_csv = f'extended_concordance{sufixo}.csv'
    df_results.to_csv(OUTPUT_DIR / nome_csv, index=False)
    print(f"\n[OK] {nome_csv}")

    # Build matrices for plotting (NxN dynamic)
    n_src = len(names)
    # O primeiro titulo precisa de duas linhas para nao ultrapassar a largura
    # do painel. Os outros dois ganham uma linha vazia acima para que as tres
    # ultimas linhas fiquem na mesma altura.
    metrics_to_plot = {'p0': 'Concordância\nobservada ($p_0$)', 'kappa': '\nKappa de Cohen',
                        'ac1': '\nAC1 de Gwet'}
    # Rotulos curtos em portugues, uma palavra por eixo para caber nas
    # celulas da matriz 5x5. A forma por extenso de cada fonte esta na
    # legenda da figura no capitulo (nao repetida aqui).
    NOME_CURTO = {
        'Weak Labels': 'Weak',
        'dNBR Auto': 'dNBR',
        'Painel do Fogo': 'Painel',
        'Human': 'Humana',
        'MapBiomas Fogo': 'MapBiomas',
    }
    nomes_plot = [NOME_CURTO.get(n, n) for n in names]
    matrices = {}
    for metric in metrics_to_plot:
        mat = pd.DataFrame(np.ones((n_src, n_src)), index=names, columns=names)
        for _, row in df_results.iterrows():
            i_idx = names.index(row['source_1'])
            j_idx = names.index(row['source_2'])
            mat.iloc[i_idx, j_idx] = row[metric]
            mat.iloc[j_idx, i_idx] = row[metric]
        matrices[metric] = mat

    # Plot: 3-panel heatmap (p0, Kappa, AC1). Canvas calibrado para a largura
    # util da pagina. Kappa e AC1 dividem uma unica barra de cor,
    # ja que usam a mesma escala, liberando largura para as celulas.
    #
    # Revisao visual: com wspace uniforme entre paineis, a folga reservada para
    # as barras de cor era replicada nos vaos que nao tem barra, o que estreitava
    # as celulas ate o ponto de os rotulos numericos se tocarem e transbordarem
    # a matriz. Aqui as barras de cor ocupam colunas proprias do gridspec, com
    # vao pequeno, de modo que a celula fica larga o bastante para conter o
    # numero. A altura acompanha a celula quadrada, sem vao morto vertical.
    n_cells = n_src
    cell_in = 0.258                     # lado da celula em polegadas
    panel_in = n_cells * cell_in
    left_in, right_in = 0.62, 0.32      # rotulos do eixo y e folga direita
    cbar_in, gap_in = 0.11, 0.10
    fig_w = left_in + 3 * panel_in + 2 * (cbar_in + 2 * gap_in) + right_in
    bottom_in, top_in = 0.62, 0.56      # rotulos rotacionados e titulos de duas linhas
    fig_h = bottom_in + panel_in + top_in

    fig = plt.figure(figsize=(fig_w, fig_h))

    def _axes(x_in, w_in):
        return fig.add_axes([x_in / fig_w, bottom_in / fig_h,
                             w_in / fig_w, panel_in / fig_h])

    x = left_in
    ax0 = _axes(x, panel_in); x += panel_in + gap_in
    cax0 = _axes(x, cbar_in); x += cbar_in + gap_in + 2 * gap_in
    ax1 = _axes(x, panel_in); x += panel_in + gap_in
    ax2 = _axes(x, panel_in); x += panel_in + gap_in
    cax1 = _axes(x, cbar_in)
    axes = [ax0, ax1, ax2]

    ims = {}
    for idx, (metric, title) in enumerate(metrics_to_plot.items()):
        ax = axes[idx]
        mat = matrices[metric].values

        if metric == 'p0':
            vmin, vmax, cmap = 0.4, 1.0, 'YlGn'
        else:
            vmin, vmax, cmap = -0.1, 1.0, 'RdYlGn'

        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ims[metric] = im
        ax.set_xticks(range(n_src))
        ax.set_xticklabels(nomes_plot, fontsize=6.3, rotation=45, ha='right')
        ax.set_yticks(range(n_src))
        ax.set_yticklabels(nomes_plot if idx == 0 else [], fontsize=6.3)
        ax.set_title(title, fontsize=7.5, pad=3)

        for ii in range(n_src):
            for jj in range(n_src):
                val = mat[ii, jj]
                color = 'white' if (val > 0.7 and metric != 'p0') or val < 0.1 else 'black'
                if ii == jj:
                    text = '1,00'
                else:
                    text = f'{val:.2f}'.replace('.', ',')
                ax.text(jj, ii, text, ha='center', va='center',
                        fontsize=5.6, fontweight='bold', color=color)

    cbar_p0 = fig.colorbar(ims['p0'], cax=cax0)
    cbar_p0.ax.tick_params(labelsize=6, length=2, pad=1.5)
    cbar_ka = fig.colorbar(ims['ac1'], cax=cax1)
    cbar_ka.ax.tick_params(labelsize=6, length=2, pad=1.5)
    cbar_ka.set_label('Kappa / AC1', fontsize=6.3, labelpad=2)

    titulo = 'Concordância entre fontes de rotulagem: $p_0$, Kappa e AC1'
    if args.convencao != 'sensu_sensor':
        titulo += '\nConvenção alternativa: $sensu\\ gestor$'
    fig.suptitle(titulo, fontsize=9, fontweight='bold', y=0.995)
    nome_png = f'extended_concordance_heatmap{sufixo}.png'
    fig.savefig(OUTPUT_DIR / nome_png, bbox_inches='tight')
    plt.close()
    print(f"[OK] {nome_png}")

    # Interpretation summary
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    for _, row in df_results.iterrows():
        pair = f"{row['source_1']} vs {row['source_2']}"
        p0_val = row['p0']
        k_val = row['kappa']
        ac1_val = row['ac1']
        pi_val = row['prevalence_index']

        if abs(ac1_val - k_val) > 0.15:
            flag = "** PREVALENCE PARADOX: AC1 >> Kappa (high p0 but low Kappa due to prevalence imbalance)"
        elif pi_val > 0.3:
            flag = "* High prevalence index, Kappa may be deflated"
        else:
            flag = "  Kappa and AC1 broadly consistent"

        print(f"  {pair:<35} p0={p0_val:.3f}  K={k_val:.4f}  AC1={ac1_val:.4f}  PI={pi_val:.3f}  {flag}")

    # JSON report
    report = {
        'description': 'Extended concordance analysis with p0, Kappa, AC1, prevalence/bias indices',
        'convencao': args.convencao,
        'gestor_mascara': args.gestor_mascara if args.convencao != 'sensu_sensor' else None,
        'n_amostras_reclassificadas': n_movidas,
        'painel_csv': str(csv_path),
        'gt_csv': str(HUMAN_GT_CSV),
        'motivation': 'Kappa can be deflated by prevalence paradox (Byrt et al. 1993). '
                      'AC1 (Gwet 2008) is less sensitive to marginal distributions.',
        'pairs': results,
        'interpretation_guide': {
            'p0': 'Raw observed agreement (proportion of matching labels)',
            'kappa': "Cohen's Kappa (chance-corrected agreement, sensitive to prevalence)",
            'ac1': "Gwet's AC1 (chance-corrected, robust to prevalence paradox)",
            'prevalence_index': '|a-d|/n, high values can deflate Kappa',
            'bias_index': '|b-c|/n, measures systematic disagreement direction',
        },
        'references': [
            'Byrt T, Bishop J, Carlin JB (1993) Bias, prevalence and kappa. J Clin Epidemiol 46:423-429',
            'Gwet KL (2008) Computing inter-rater reliability and its variance. Br J Math Stat Psychol 61:29-48',
            'Landis JR, Koch GG (1977) The measurement of observer agreement. Biometrics 33:159-174',
        ],
    }
    nome_json = f'extended_concordance_report{sufixo}.json'
    with open(OUTPUT_DIR / nome_json, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OK] {nome_json}")
    print("Done!")


if __name__ == '__main__':
    main()
