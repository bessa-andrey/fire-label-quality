"""Proporcao classificada como fogo real por fonte, sobre a base comum as 5 fontes.

Motivacao. O numero-ancora citado no Resumo, no Abstract, na Introducao, no
Cap. 3 e na Conclusao era "de 22% a 73%". Os dois extremos vinham de bases
diferentes: os 22% do MapBiomas saiam do painel inteiro (n=3.698) e os 73% do
dNBR auto saiam da referencia humana binaria (n=535). Misturar bases produz um
intervalo que nenhuma comparacao pareada sustenta, e deixa a anotacao humana
fora do intervalo.

Este script mede as cinco fontes sobre a MESMA base: as deteccoes em que as
cinco rotulagens estao simultaneamente disponiveis (n=439). A logica de merge
entre o painel e a referencia humana (lat/lon arredondados a 4 casas mais a
data) e a mesma de `src/models/compute_extended_concordance.py`, que produz a
Tabela de concordancia do Cap. 4.

Insumos:
    outputs/ground_truth_validation/painel_labeled_5strategies.csv
        (is_reliable, gt_painel, gt_mapbiomas_fire)
    data/processed/validation/validation_manual_gt.csv
        (human_label, manual_label)

Uso:
    python scripts/proporcao_fogo_real_base_comum.py
    python scripts/proporcao_fogo_real_base_comum.py \
        --painel-csv outputs/weak_labels_v2/painel_labeled_5strategies_weakv2.csv \
        --gt-csv outputs/weak_labels_v2/validation_manual_gt_weakv2.csv

As opcoes `--painel-csv` e `--gt-csv` apontam o script para insumos alternativos, do
mesmo jeito que `compute_extended_concordance.py` e `run_decisive_experiment.py` ja
faziam. Sem elas o comportamento e identico ao anterior. Elas existem para a Estrategia 1
recomputada por `src/preprocessing/recompute_weak_labels_mcd64a1.py`.

Convencao unificada, a mesma do Cap. 3: 0 = Fogo Real, 1 = Espurio, -1 = Incerto.
`is_reliable`, `gt_painel` e `gt_mapbiomas_fire` sao invertidos para chegar nela.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAINEL_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_5strategies.csv'
HUMAN_GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'

# nome exibido -> (coluna, precisa inverter para a convencao 0 = Real)
SOURCES = {
    'Weak labels (FIRMS x MCD64A1)': ('is_reliable', True),
    'dNBR auto (Sentinel-2)': ('manual_label', False),
    'Painel do Fogo (CENSIPAM)': ('gt_painel', True),
    'MapBiomas Fogo': ('gt_mapbiomas_fire', True),
    'Anotacao humana': ('human_label', False),
}


def _merge_painel_onto_gt(painel_csv=None, gt_csv=None):
    """Traz gt_painel e gt_mapbiomas_fire para as linhas da referencia humana."""
    painel = pd.read_csv(painel_csv or PAINEL_CSV)
    painel['acq_datetime'] = pd.to_datetime(painel['acq_datetime'], format='mixed')

    gt = pd.read_csv(gt_csv or HUMAN_GT_CSV)
    gt['acq_datetime'] = pd.to_datetime(gt['acq_datetime'], format='mixed')
    gt = gt[gt['human_label'].isin([0, 1])].copy()

    for frame in (painel, gt):
        frame['_lat_r'] = frame['latitude'].round(4)
        frame['_lon_r'] = frame['longitude'].round(4)
        frame['_date'] = frame['acq_datetime'].dt.strftime('%Y-%m-%d')

    keys = ['_lat_r', '_lon_r', '_date']
    cols = keys + ['gt_painel', 'gt_mapbiomas_fire']
    gt = gt.merge(painel[cols].drop_duplicates(), on=keys, how='left', suffixes=('', '_p'))
    for col in ('gt_painel', 'gt_mapbiomas_fire'):
        if f'{col}_p' in gt.columns:
            gt[col] = gt[f'{col}_p'].combine_first(gt[col]) if col in gt.columns else gt[f'{col}_p']
            gt.drop(columns=[f'{col}_p'], inplace=True)
    return gt.drop(columns=keys)


def parse_args():
    ap = argparse.ArgumentParser(
        description='Proporcao de fogo real por fonte sobre a base comum as cinco')
    ap.add_argument('--painel-csv', type=Path, default=None,
                    help='Substitui o painel de entrada '
                         '(padrao: painel_labeled_5strategies.csv)')
    ap.add_argument('--gt-csv', type=Path, default=None,
                    help='Substitui a referencia humana '
                         '(padrao: validation_manual_gt.csv)')
    return ap.parse_args()


def main():
    args = parse_args()
    gt = _merge_painel_onto_gt(args.painel_csv, args.gt_csv)
    print(f'Painel:     {args.painel_csv or PAINEL_CSV}')
    print(f'Referencia: {args.gt_csv or HUMAN_GT_CSV}')
    print(f'Referencia humana binaria (human_label em 0/1): n = {len(gt)}')

    mask = pd.Series(True, index=gt.index)
    for name, (col, _) in SOURCES.items():
        available = gt[col].notna() & (gt[col] != -1)
        mask &= available
        print(f'  + {name:32s} disponivel em {available.sum():4d}, base comum acumulada {mask.sum():4d}')

    common = gt[mask]
    print(f'\nBase comum as 5 fontes: n = {len(common)}\n')

    pcts = {}
    for name, (col, invert) in SOURCES.items():
        values = common[col].astype(int)
        labels = 1 - values if invert else values
        pct = float((labels == 0).mean() * 100)
        pcts[name] = pct
        print(f'  {name:32s} {pct:5.1f}% real  ({int((labels == 0).sum())}/{len(labels)})')

    low, high = min(pcts, key=pcts.get), max(pcts, key=pcts.get)
    print(f'\nIntervalo entre as cinco fontes sobre n = {len(common)}: '
          f'{pcts[low]:.1f}% ({low}) a {pcts[high]:.1f}% ({high})')


if __name__ == '__main__':
    main()
