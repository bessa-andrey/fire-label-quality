"""Exemplo minimo: concordancia pareada de cada fonte automatica contra a anotacao
humana, sobre o arquivo publicado.

Reproduz os quatro valores da coluna "vs. anotacao humana" da matriz de concordancia
da dissertacao. Depende apenas de pandas e scikit-learn.

    python examples/pairwise_agreement.py
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

GT_CSV = Path(__file__).resolve().parent.parent / 'data' / 'matopiba_fire_human_gt_v1.csv'

# Convencao unificada: 0 = Fogo Real, 1 = Espurio, -1 = Incerto.
# Cada fonte chega numa orientacao propria, entao tres delas sao invertidas.
FONTES = {
    'weak_labels': ('is_reliable', True),
    'dnbr_auto': ('manual_label', False),
    'mapbiomas_fogo': ('gt_mapbiomas_fire', True),
    'painel_fogo': ('gt_painel', True),
}


def main():
    df = pd.read_csv(GT_CSV)

    # As incertas, -1, saem de toda medida de concordancia.
    binarias = df[df['human_label'].isin([0, 1])].copy()
    print(f'Deteccoes anotadas: {len(df)}')
    print(f'Com rotulo binario: {len(binarias)}')
    print()

    y_humano = binarias['human_label']

    print(f'{"fonte":<16} {"kappa":>8} {"concordancia":>14} {"n":>6}')
    print('-' * 48)
    for nome, (coluna, inverter) in FONTES.items():
        y_fonte = 1 - binarias[coluna] if inverter else binarias[coluna]
        par = pd.DataFrame({'humano': y_humano, 'fonte': y_fonte}).dropna()
        kappa = cohen_kappa_score(par['humano'], par['fonte'])
        acordo = (par['humano'] == par['fonte']).mean()
        print(f'{nome:<16} {kappa:>8.3f} {acordo:>13.1%} {len(par):>6}')

    print()
    print('As duas ultimas fontes rodam sobre 439 deteccoes porque Painel do Fogo e')
    print('MapBiomas Fogo so estao disponiveis no subconjunto com cobertura completa.')


if __name__ == '__main__':
    main()
