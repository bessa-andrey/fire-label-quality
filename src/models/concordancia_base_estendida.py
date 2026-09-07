"""
Concordancia entre fontes de rotulo sobre a BASE ESTENDIDA (analise de robustez).

A matriz primaria do Cap. 4 (`compute_extended_concordance.py`) roda sobre o recorte que
tem anotacao humana: n=535 nos pares que so precisam de Weak/dNBR/Humano e n=439 nos pares
que tambem exigem `gt_painel` ou `gt_mapbiomas_fire`. Esses n pequenos deixam os intervalos
de confianca largos o bastante para que boa parte dos pares fique estatisticamente
indistinguivel entre si.

Este script recomputa os pares que NAO dependem da anotacao humana sobre o painel inteiro.
O que limita o painel e o rotulo dNBR: ele so foi materializado como `manual_label` nas
amostras que passaram pela campanha de validacao. As estatisticas dNBR (`dnbr_mean`,
`dnbr_max`, `pct_burned`), porem, existem para 2.928 das 3.698 deteccoes do painel. Como o
rotulo dNBR e uma REGRA DETERMINISTICA dessas tres estatisticas
(`src/preprocessing/dnbr_rule.classify_dnbr`, a mesma regra do Cap. 3), ele pode ser
estendido a todas as 2.928 linhas sem nenhuma anotacao adicional.

O script primeiro AUDITA a extensao: aplica a regra as amostras que ja tem `manual_label`
publicado e mede a taxa de reproducao. So depois estende ao painel. As quatro fontes
automaticas ficam disponiveis simultaneamente nas 2.928 linhas, logo os seis pares entre
elas rodam todos sobre o mesmo n, ao contrario da tabela primaria.

Os pares que envolvem a anotacao humana NAO entram aqui: a referencia humana existe apenas
nas 604 amostras anotadas e nao pode ser estendida por regra alguma. Eles permanecem
exclusivamente na tabela primaria.

Usage:
    python src/models/concordancia_base_estendida.py
    python src/models/concordancia_base_estendida.py \
        --painel-csv outputs/weak_labels_v2/painel_labeled_5strategies_weakv2.csv \
        --gt-csv outputs/weak_labels_v2/validation_manual_gt_weakv2.csv \
        --output-dir outputs/weak_labels_v2/concordancia

As opcoes `--painel-csv`, `--gt-csv` e `--output-dir` apontam o script para insumos
alternativos, do mesmo jeito que `compute_extended_concordance.py` e
`run_decisive_experiment.py` ja faziam. Sem elas o comportamento e identico ao anterior.
Elas existem para a Estrategia 1 recomputada por
`src/preprocessing/recompute_weak_labels_mcd64a1.py`.

Output:
    outputs/decisive_experiment/concordancia_base_estendida.csv
    outputs/decisive_experiment/concordancia_base_estendida_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models.compute_extended_concordance import (  # noqa: E402
    HUMAN_GT_CSV,
    bias_index,
    carregar_referencia,
    gwet_ac1,
    observed_agreement,
    prevalence_index,
)
from preprocessing.dnbr_rule import classify_series  # noqa: E402

PAINEL_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_5strategies.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'decisive_experiment'

N_BOOT = 2000
SEED = 42

# Convencao unificada 0 = fogo real, 1 = espurio. `inverter=True` marca as colunas
# gravadas na convencao oposta (1 = evento/queimado na fonte).
FONTES = {
    'Weak Labels': ('is_reliable', True),
    'dNBR Auto': ('dnbr_label_regra', False),
    'Painel do Fogo': ('gt_painel', True),
    'MapBiomas Fogo': ('gt_mapbiomas_fire', True),
}


def kappa_ci(y1, y2, n_boot=N_BOOT, seed=SEED):
    """IC 95% do Kappa por bootstrap percentil sobre os pares de rotulos."""
    rng = np.random.RandomState(seed)
    n = len(y1)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        a, b = y1[idx], y2[idx]
        if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
            continue
        vals.append(cohen_kappa_score(a, b))
    if not vals:
        return float('nan'), float('nan')
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def metricas_par(li, lj):
    return {
        'n': int(len(li)),
        'p0': observed_agreement(li, lj),
        'kappa': float(cohen_kappa_score(li, lj)),
        'ac1': gwet_ac1(li, lj),
        'prevalence_index': prevalence_index(li, lj),
        'bias_index': bias_index(li, lj),
    }


def auditar_regra(gt_csv=None):
    """Compara o rotulo da regra com o `manual_label` publicado, amostra a amostra.

    A auditoria roda sobre TODAS as amostras com `manual_label` publicado, nao so sobre o
    recorte binario humano de n=535, porque o que esta em julgamento e a regra dNBR e nao a
    anotacao humana. Retorna o dicionario que vai para o relatorio. A extensao so e
    defensavel se esta taxa for alta: ela mede quanto do rotulo dNBR publicado a regra
    reconstroi sozinha.
    """
    df_gt = pd.read_csv(gt_csv or HUMAN_GT_CSV)
    cols = ['auto_mean_dnbr', 'auto_max_dnbr', 'auto_pct_burned']
    sub = df_gt[df_gt['manual_label'].notna() & (df_gt['manual_label'] != -1)
                & df_gt[cols].notna().all(axis=1)].copy()
    regra, escala = classify_series(sub['auto_mean_dnbr'], sub['auto_max_dnbr'],
                                    sub['auto_pct_burned'])
    publicado = sub['manual_label'].astype(int).values
    concorda = regra == publicado
    return {
        'n_auditadas': int(len(sub)),
        'escala_pct_detectada': escala,
        'n_divergentes': int((~concorda).sum()),
        'taxa_reproducao': float(concorda.mean()),
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description='Concordancia entre fontes de rotulo sobre a base estendida')
    ap.add_argument('--painel-csv', type=Path, default=None,
                    help='Substitui o painel de entrada '
                         '(padrao: painel_labeled_5strategies.csv)')
    ap.add_argument('--gt-csv', type=Path, default=None,
                    help='Substitui a referencia humana '
                         '(padrao: validation_manual_gt.csv)')
    ap.add_argument('--output-dir', type=Path, default=None,
                    help='Substitui o diretorio de saida '
                         '(padrao: outputs/decisive_experiment)')
    return ap.parse_args()


def main():
    global PAINEL_CSV, OUTPUT_DIR
    args = parse_args()
    if args.painel_csv:
        PAINEL_CSV = args.painel_csv
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    gt_csv = args.gt_csv or HUMAN_GT_CSV

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('CONCORDANCIA SOBRE A BASE ESTENDIDA (analise de robustez)')
    print('=' * 72)

    auditoria = auditar_regra(gt_csv)
    print(f"\n[1] Auditoria da extensao do rotulo dNBR")
    print(f"    regra: src/preprocessing/dnbr_rule.classify_dnbr (escala "
          f"{auditoria['escala_pct_detectada']})")
    print(f"    amostras com manual_label publicado e estatisticas dNBR: "
          f"{auditoria['n_auditadas']}")
    print(f"    divergentes: {auditoria['n_divergentes']}")
    print(f"    taxa de reproducao: {auditoria['taxa_reproducao']:.4f} "
          f"({auditoria['taxa_reproducao'] * 100:.1f}%)")

    df = pd.read_csv(PAINEL_CSV)
    print(f"\n[2] Painel: {len(df)} deteccoes em {PAINEL_CSV.name}")

    # O gargalo e a estatistica dNBR. `is_reliable`, `gt_painel` e `gt_mapbiomas_fire`
    # existem para as 3.698 linhas, mas entram no filtro para deixar o criterio explicito.
    estat = ['dnbr_mean', 'dnbr_max', 'pct_burned']
    mask = df[estat + ['is_reliable', 'gt_painel', 'gt_mapbiomas_fire']].notna().all(axis=1)
    base = df[mask].copy()
    print(f"    com as tres estatisticas dNBR e as tres fontes restantes: {len(base)}")

    rotulos, escala = classify_series(base['dnbr_mean'], base['dnbr_max'], base['pct_burned'])
    base['dnbr_label_regra'] = rotulos
    print(f"    rotulo dNBR estendido a {len(base)} linhas (escala pct: {escala})")
    print(f"    distribuicao: {(rotulos == 0).sum()} real / {(rotulos == 1).sum()} espurio")

    # Kappa e IC tambem sobre a base primaria, para quantificar o estreitamento do IC.
    df_gt, _ = carregar_referencia(PAINEL_CSV, gt_csv, verbose=False)
    df_gt = df_gt.rename(columns={'manual_label': 'dnbr_label_regra'})
    mask_prim = (df_gt['human_label'].isin([0, 1])
                 & df_gt['dnbr_label_regra'].notna() & (df_gt['dnbr_label_regra'] != -1)
                 & df_gt['is_reliable'].notna())

    nomes = list(FONTES.keys())
    resultados = []

    print(f"\n[3] Pares entre as quatro fontes automaticas (base estendida)\n")
    cab = (f"{'Par':<38} {'p0':>6} {'kappa':>7} {'IC 95%':>16} {'AC1':>7} "
           f"{'PI':>6} {'BI':>6} {'n':>6}")
    print(cab)
    print('-' * len(cab))

    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            ni, nj = nomes[i], nomes[j]
            ci_col, inv_i = FONTES[ni]
            cj_col, inv_j = FONTES[nj]

            li = base[ci_col].astype(int).values
            lj = base[cj_col].astype(int).values
            li = 1 - li if inv_i else li
            lj = 1 - lj if inv_j else lj

            m = metricas_par(li, lj)
            lo, hi = kappa_ci(li, lj)
            m.update({'source_1': ni, 'source_2': nj,
                      'kappa_ic_lo': lo, 'kappa_ic_hi': hi,
                      'kappa_ic_largura': hi - lo})

            # Mesmo par sobre a base primaria, para o fator de estreitamento do IC.
            pm = mask_prim.copy()
            for col in (ci_col, cj_col):
                if col in ('gt_painel', 'gt_mapbiomas_fire'):
                    pm = pm & df_gt[col].notna()
            dp = df_gt[pm]
            pi_ = dp[ci_col].astype(int).values
            pj_ = dp[cj_col].astype(int).values
            pi_ = 1 - pi_ if inv_i else pi_
            pj_ = 1 - pj_ if inv_j else pj_
            k_prim = float(cohen_kappa_score(pi_, pj_))
            lo_p, hi_p = kappa_ci(pi_, pj_)
            m.update({'n_primaria': int(len(dp)), 'kappa_primaria': k_prim,
                      'kappa_primaria_ic_lo': lo_p, 'kappa_primaria_ic_hi': hi_p,
                      'kappa_primaria_ic_largura': hi_p - lo_p,
                      'fator_estreitamento_ic': (hi_p - lo_p) / (hi - lo)})

            resultados.append(m)
            par = f'{ni} vs {nj}'
            ic = f"[{lo:.3f}; {hi:.3f}]"
            print(f"{par:<38} {m['p0']:>6.3f} {m['kappa']:>7.3f} {ic:>16} "
                  f"{m['ac1']:>7.3f} {m['prevalence_index']:>6.3f} "
                  f"{m['bias_index']:>6.3f} {m['n']:>6}")

    print(f"\n[4] Comparacao com a tabela primaria\n")
    cab2 = (f"{'Par':<38} {'k prim':>7} {'n prim':>7} {'k est':>7} {'n est':>7} "
            f"{'IC prim':>7} {'IC est':>7} {'fator':>6}")
    print(cab2)
    print('-' * len(cab2))
    for m in resultados:
        par = f"{m['source_1']} vs {m['source_2']}"
        print(f"{par:<38} {m['kappa_primaria']:>7.3f} {m['n_primaria']:>7} "
              f"{m['kappa']:>7.3f} {m['n']:>7} {m['kappa_primaria_ic_largura']:>7.3f} "
              f"{m['kappa_ic_largura']:>7.3f} {m['fator_estreitamento_ic']:>6.2f}")

    cols = ['source_1', 'source_2', 'n', 'p0', 'kappa', 'kappa_ic_lo', 'kappa_ic_hi',
            'kappa_ic_largura', 'ac1', 'prevalence_index', 'bias_index',
            'n_primaria', 'kappa_primaria', 'kappa_primaria_ic_lo',
            'kappa_primaria_ic_hi', 'kappa_primaria_ic_largura', 'fator_estreitamento_ic']
    df_out = pd.DataFrame(resultados)[cols]
    df_out.to_csv(OUTPUT_DIR / 'concordancia_base_estendida.csv', index=False)
    print(f"\n[OK] concordancia_base_estendida.csv")

    report = {
        'descricao': 'Concordancia entre as quatro fontes automaticas sobre a base estendida '
                     'do painel, com o rotulo dNBR reconstruido por regra deterministica.',
        'papel': 'analise de robustez da Tabela de concordancia primaria do Cap. 4; '
                 'a tabela primaria permanece a analise de referencia',
        'painel_csv': str(PAINEL_CSV),
        'gt_csv': str(gt_csv),
        'regra_dnbr': 'src/preprocessing/dnbr_rule.classify_dnbr',
        'auditoria_extensao': auditoria,
        'n_base_estendida': int(len(base)),
        'criterio_base_estendida': 'linhas do painel com dnbr_mean, dnbr_max e pct_burned '
                                   'nao nulos; is_reliable, gt_painel e gt_mapbiomas_fire '
                                   'estao presentes em todas as 3.698 linhas',
        'fora_do_escopo': 'pares que envolvem a anotacao humana; a referencia humana existe '
                          'so nas 604 amostras anotadas e nao e extensivel por regra',
        'bootstrap': {'n_reamostragens': N_BOOT, 'seed': SEED, 'metodo': 'percentil 2,5/97,5'},
        'proporcao_fogo_real': [
            {
                'fonte': nome,
                'n': int(len(base)),
                'n_real': int(((1 - base[col].astype(int)) if inverter
                               else base[col].astype(int)) .eq(0).sum()),
                'pct_real': round(float((((1 - base[col].astype(int)) if inverter
                                          else base[col].astype(int)) == 0).mean() * 100), 2),
            }
            for nome, (col, inverter) in FONTES.items()
        ],
        'pares': resultados,
    }
    with open(OUTPUT_DIR / 'concordancia_base_estendida_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] concordancia_base_estendida_report.json")
    print('\nDone!')


if __name__ == '__main__':
    main()
