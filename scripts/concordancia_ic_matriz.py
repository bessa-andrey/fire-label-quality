"""Matriz de concordancia entre as fontes de rotulo, com intervalo de confianca.

POR QUE ESTE SCRIPT EXISTE
--------------------------
`src/models/compute_extended_concordance.py` produz a matriz primaria do capitulo de
resultados com p0, Kappa, AC1, indice de prevalencia e indice de vies, sem intervalo de
confianca. `src/models/concordancia_base_estendida.py` produz intervalo, porem so para os
seis pares entre as quatro fontes automaticas, sobre a base estendida de 2.928 deteccoes.
Faltava o intervalo dos quatro pares que envolvem a anotacao humana e faltava a matriz
sobre o painel inteiro de 3.698 deteccoes. Este script preenche as duas lacunas, sem tocar
em nenhum dos dois, e reaproveita `carregar_referencia` para nao reimplementar o
pareamento por (lat, lon, data).

BASES COBERTAS
--------------
  primaria    o recorte com anotacao humana. Cada par roda sobre o maior subconjunto em
              que as duas fontes existem, o que da n=535 nos pares que so precisam de
              Weak, dNBR e Humana, e n=439 nos que tambem exigem Painel do Fogo ou
              MapBiomas Fogo. Mesmo criterio de `compute_extended_concordance.py`
  painel      o painel inteiro de 3.698 deteccoes, onde existem apenas as tres fontes
              gravadas coluna a coluna: Weak Labels, Painel do Fogo e MapBiomas Fogo. O
              rotulo dNBR nao esta materializado no painel inteiro, e a anotacao humana
              existe so nas 604 amostras anotadas

A base estendida de 2.928 deteccoes NAO entra aqui. Ela tem fonte unica em
`src/models/concordancia_base_estendida.py`, que reconstroi o rotulo dNBR por regra
deterministica e ja reporta intervalo.

CONVENCAO
---------
Convencao unificada 0 = fogo real, 1 = espurio. `is_reliable`, `gt_painel` e
`gt_mapbiomas_fire` sao gravadas com a polaridade oposta e entram invertidas. A convencao
de rotulagem operativa e a `sensu_sensor`, unica adotada pela dissertacao.

INTERVALO DE CONFIANCA
----------------------
Bootstrap percentil de 2,5 e 97,5, com 2.000 reamostragens e semente 42, os mesmos
parametros de `src/models/concordancia_base_estendida.py`.

Uso:
    python scripts/concordancia_ic_matriz.py
    python scripts/concordancia_ic_matriz.py \
        --painel-csv outputs/weak_labels_v2/painel_labeled_5strategies_weakv2.csv \
        --gt-csv outputs/weak_labels_v2/validation_manual_gt_weakv2.csv \
        --output-dir outputs/weak_labels_v2/concordancia

Saida:
    <output-dir>/concordancia_ic_matriz.csv
    <output-dir>/concordancia_ic_matriz.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from models.compute_extended_concordance import (  # noqa: E402
    HUMAN_GT_CSV,
    PAINEL_EXPANDED_CSV,
    bias_index,
    carregar_referencia,
    gwet_ac1,
    observed_agreement,
    prevalence_index,
)

OUTPUT_DIR_PADRAO = PROJECT_ROOT / 'outputs' / 'decisive_experiment'

N_BOOT = 2000
SEED = 42

# nome exibido -> (coluna, inverter para a convencao 0 = fogo real)
FONTES_PRIMARIA = {
    'Weak Labels': ('is_reliable', True),
    'dNBR Auto': ('manual_label', False),
    'Painel do Fogo': ('gt_painel', True),
    'Human': ('human_label', False),
    'MapBiomas Fogo': ('gt_mapbiomas_fire', True),
}

# No painel inteiro so existem as tres fontes gravadas coluna a coluna.
FONTES_PAINEL = {
    'Weak Labels': ('is_reliable', True),
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


def _alinhar(serie, inverter):
    v = serie.astype(int).values
    return 1 - v if inverter else v


def pares(df, fontes, base, mascara_extra=None):
    """Todos os pares de fontes sobre o maior subconjunto em que as duas existem."""
    nomes = list(fontes)
    linhas = []
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            ni, nj = nomes[i], nomes[j]
            ci, invi = fontes[ni]
            cj, invj = fontes[nj]
            if ci not in df.columns or cj not in df.columns:
                continue
            m = pd.Series(True, index=df.index) if mascara_extra is None else mascara_extra.copy()
            for c in (ci, cj):
                m &= df[c].notna() & (df[c] != -1)
            sub = df[m]
            if len(sub) < 10:
                continue
            li = _alinhar(sub[ci], invi)
            lj = _alinhar(sub[cj], invj)
            lo, hi = kappa_ci(li, lj)
            k = float(cohen_kappa_score(li, lj))
            linhas.append({
                'base': base,
                'fonte_1': ni, 'fonte_2': nj, 'n': int(len(sub)),
                'p0': observed_agreement(li, lj),
                'kappa': k, 'kappa_ic_lo': lo, 'kappa_ic_hi': hi,
                'kappa_ic_largura': hi - lo,
                'ac1': gwet_ac1(li, lj),
                'prevalence_index': prevalence_index(li, lj),
                'bias_index': bias_index(li, lj),
                'pct_real_fonte_1': float((li == 0).mean() * 100),
                'pct_real_fonte_2': float((lj == 0).mean() * 100),
            })
    return linhas


def proporcoes(df, fontes, base, mascara_extra=None):
    """Proporcao classificada como fogo real por fonte, cada uma no seu proprio n."""
    linhas = []
    for nome, (col, inverter) in fontes.items():
        if col not in df.columns:
            continue
        m = df[col].notna() & (df[col] != -1)
        if mascara_extra is not None:
            m &= mascara_extra
        sub = df[m]
        if len(sub) == 0:
            continue
        lab = _alinhar(sub[col], inverter)
        linhas.append({
            'base': base, 'fonte': nome, 'n': int(len(sub)),
            'n_real': int((lab == 0).sum()),
            'pct_real': float((lab == 0).mean() * 100),
        })
    return linhas


def _imprimir(linhas):
    cab = (f"{'Par':<38} {'n':>6} {'p0':>6} {'kappa':>7} {'IC 95%':>18} "
           f"{'AC1':>7} {'PI':>6} {'BI':>6}")
    print(cab)
    print('-' * len(cab))
    for r in linhas:
        par = f"{r['fonte_1']} vs {r['fonte_2']}"
        ic = f"[{r['kappa_ic_lo']:.3f}; {r['kappa_ic_hi']:.3f}]"
        print(f"{par:<38} {r['n']:>6} {r['p0']:>6.3f} {r['kappa']:>7.4f} {ic:>18} "
              f"{r['ac1']:>7.3f} {r['prevalence_index']:>6.3f} {r['bias_index']:>6.3f}")


def parse_args():
    ap = argparse.ArgumentParser(
        description='Matriz de concordancia com intervalo de confianca por bootstrap')
    ap.add_argument('--painel-csv', type=Path, default=None,
                    help='Substitui o painel de entrada '
                         '(padrao: painel_labeled_5strategies.csv)')
    ap.add_argument('--gt-csv', type=Path, default=None,
                    help='Substitui a referencia humana '
                         '(padrao: validation_manual_gt.csv)')
    ap.add_argument('--output-dir', type=Path, default=OUTPUT_DIR_PADRAO,
                    help='Diretorio de saida (padrao: outputs/decisive_experiment)')
    return ap.parse_args()


COMANDO_INSUMOS = 'python src/preprocessing/recompute_weak_labels_mcd64a1.py --no-cache'


def _exigir_painel_existente(caminho):
    """O painel ausente e o caso perigoso, porque ele falha em silencio.

    Sem o painel estrito, `carregar_referencia` recai sobre
    `outputs/ground_truth_validation/painel_labeled_expanded.csv`. O que vem do painel
    sao apenas `gt_painel`, `gt_mapbiomas_fire` e `land_use_class`, pelo `merge_cols` de
    `compute_extended_concordance.py`. As colunas `is_reliable`, `manual_label` e
    `human_label` vem todas do CSV da referencia humana, entao os pares que envolvem o
    weak label continuam sobre a coluna estrita. O que a troca silenciosa desloca sao o n
    e os quatro pares que envolvem o Painel do Fogo ou o MapBiomas Fogo na matriz
    primaria, por diferenca de cobertura entre os dois paineis, e isso sai impresso antes
    de a rodada morrer ao ler o painel para a matriz do painel inteiro.
    """
    if caminho.exists():
        return
    raise SystemExit(
        f'ERRO: o painel {caminho} nao existe.\n'
        'Sem ele, carregar_referencia() de compute_extended_concordance.py cai em silencio\n'
        'em outputs/ground_truth_validation/painel_labeled_expanded.csv.\n'
        'O painel fornece gt_painel, gt_mapbiomas_fire e land_use_class, entao a troca\n'
        'desloca o n e os quatro pares que envolvem o Painel do Fogo ou o MapBiomas Fogo\n'
        'na matriz primaria. Os pares com o weak label seguem sobre a coluna estrita,\n'
        'porque is_reliable vem do CSV da referencia humana.\n'
        f'Gere os insumos estritos com:\n  {COMANDO_INSUMOS}')


def _exigir_gt_existente(caminho):
    """A referencia humana ausente morre com FileNotFoundError cru do pandas."""
    if caminho.exists():
        return
    raise SystemExit(
        f'ERRO: a referencia humana {caminho} nao existe.\n'
        f'Gere os insumos estritos com:\n  {COMANDO_INSUMOS}')


def main():
    args = parse_args()
    painel_csv = args.painel_csv or PAINEL_EXPANDED_CSV
    gt_csv = args.gt_csv or HUMAN_GT_CSV
    _exigir_painel_existente(painel_csv)
    _exigir_gt_existente(gt_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('MATRIZ DE CONCORDANCIA COM INTERVALO DE CONFIANCA')
    print('=' * 72)
    print(f'painel:     {painel_csv}')
    print(f'referencia: {gt_csv}')
    print(f'bootstrap:  {N_BOOT} reamostragens, semente {SEED}, percentil 2,5/97,5')

    # --- base primaria, o recorte com anotacao humana ---
    df_gt, _ = carregar_referencia(painel_csv, gt_csv, verbose=True)
    mask_3 = (df_gt['human_label'].isin([0, 1])
              & df_gt['manual_label'].notna() & (df_gt['manual_label'] != -1)
              & df_gt['is_reliable'].notna())
    print(f'\n[1] Base primaria: {int(mask_3.sum())} amostras com Weak, dNBR e Humana\n')
    linhas = pares(df_gt, FONTES_PRIMARIA, 'primaria', mask_3)
    _imprimir(linhas)
    props = proporcoes(df_gt, FONTES_PRIMARIA, 'primaria', mask_3)

    # --- painel inteiro ---
    painel = pd.read_csv(painel_csv)
    print(f'\n[2] Painel inteiro: {len(painel)} deteccoes\n')
    linhas_painel = pares(painel, FONTES_PAINEL, 'painel_3698')
    _imprimir(linhas_painel)
    props += proporcoes(painel, FONTES_PAINEL, 'painel_3698')

    print('\n[3] Proporcao classificada como fogo real por fonte\n')
    for r in props:
        print(f"  {r['base']:<12} {r['fonte']:<18} {r['pct_real']:>6.2f}%  "
              f"({r['n_real']}/{r['n']})")

    todas = linhas + linhas_painel
    df_out = pd.DataFrame(todas)
    df_out.to_csv(args.output_dir / 'concordancia_ic_matriz.csv', index=False)
    print(f"\n[OK] {args.output_dir / 'concordancia_ic_matriz.csv'}")

    relatorio = {
        'painel_csv': str(painel_csv),
        'gt_csv': str(gt_csv),
        'bootstrap': {'n_reamostragens': N_BOOT, 'seed': SEED,
                      'metodo': 'percentil 2,5/97,5'},
        'convencao': 'sensu_sensor, 0 = fogo real, 1 = espurio',
        'bases': {
            'primaria': 'recorte com anotacao humana, cada par no maior subconjunto '
                        'em que as duas fontes existem',
            'painel_3698': 'painel inteiro, so as tres fontes gravadas coluna a coluna',
        },
        'fora_do_escopo': 'base estendida de 2.928 deteccoes, cuja fonte unica e '
                          'src/models/concordancia_base_estendida.py',
        'pares': todas,
        'proporcoes': props,
    }
    with open(args.output_dir / 'concordancia_ic_matriz.json', 'w', encoding='utf-8') as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] {args.output_dir / 'concordancia_ic_matriz.json'}")


if __name__ == '__main__':
    main()
