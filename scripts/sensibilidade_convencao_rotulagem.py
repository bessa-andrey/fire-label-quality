"""Analise de sensibilidade da convencao de rotulagem sobre a matriz de concordancia.

PERGUNTA
--------
A dissertacao reporta a concordancia entre as cinco fontes sob a Convencao *sensu sensor*,
em que toda combustao de biomassa no pixel e fogo real. Sob a alternativa declarada no
Cap. 3, a Convencao *sensu gestor*, a queimada de manejo passa a contar como espuria.
Este script roda a matriz sob as duas convencoes, lado a lado, e testa se a ORDEM entre
as fontes automaticas muda quando a convencao muda.

O QUE ELE NAO FAZ
-----------------
Nao roda o classificador. A comparacao aqui e entre rotulos, nao entre modelos treinados,
e o Kappa principal da dissertacao, que e do classificador sob a convencao principal, nao
e tocado.

COMO A CONVENCAO ENTRA
----------------------
Pela projecao da referencia humana, implementada em
`src/preprocessing/labeling_convention.py`. As fontes automaticas medem cicatriz e medem a
mesma coisa sob as duas convencoes, entao os pares entre elas tem valor identico nas duas
rodadas, e a sensibilidade se concentra nos pares que envolvem a referencia humana.

SAIDA
-----
  <output-dir>/sensibilidade_convencao.csv    uma linha por (convencao, mascara, fonte, base)
  <output-dir>/sensibilidade_convencao.json   o mesmo, mais os testes de bootstrap pareado

Uso:
    python scripts/sensibilidade_convencao_rotulagem.py --output-dir outputs/decisive_experiment
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
    bias_index, carregar_referencia, gwet_ac1, observed_agreement, prevalence_index)
from preprocessing.labeling_convention import aplicar_convencao  # noqa: E402

SEMENTE = 20260823
N_BOOTSTRAP = 5000

# (nome, coluna, inverter) na convencao unificada 0 = fogo real, 1 = espurio.
FONTES_AUTOMATICAS = [
    ('Weak Labels', 'is_reliable', True),
    ('dNBR Auto', 'manual_label', False),
    ('Painel do Fogo', 'gt_painel', True),
    ('MapBiomas Fogo', 'gt_mapbiomas_fire', True),
]

RODADAS = [
    ('sensu_sensor', 'notas'),
    ('sensu_gestor', 'notas'),
    ('sensu_gestor', 'uso_solo'),
]


def _valores(df, coluna, inverter):
    v = df[coluna].astype(int).values
    return (1 - v) if inverter else v


def _recortes(df_gt):
    """As duas bases sobre as quais a matriz e lida no texto."""
    base3 = (df_gt['human_label'].isin([0, 1])
             & df_gt['manual_label'].notna() & (df_gt['manual_label'] != -1)
             & df_gt['is_reliable'].notna())
    base5 = base3 & df_gt['gt_painel'].notna() & df_gt['gt_mapbiomas_fire'].notna()
    return [('3 fontes', base3, ['Weak Labels', 'dNBR Auto']),
            ('5 fontes', base5, [nome for nome, _, _ in FONTES_AUTOMATICAS])]


def bootstrap_diferenca(humano, a, b, rng, n=N_BOOTSTRAP):
    """Bootstrap pareado da diferenca de Kappa entre duas fontes contra a mesma referencia.

    Reamostra as detecoes, nao os rotulos, entao as duas fontes veem sempre o mesmo
    conjunto reamostrado, que e o que torna a diferenca pareada.
    """
    idx = np.arange(len(humano))
    difs = []
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(humano[i])) < 2:
            continue
        difs.append(cohen_kappa_score(a[i], humano[i]) - cohen_kappa_score(b[i], humano[i]))
    difs = np.asarray(difs)
    return {
        'diferenca_media': float(difs.mean()),
        'ic95_inferior': float(np.percentile(difs, 2.5)),
        'ic95_superior': float(np.percentile(difs, 97.5)),
        'p_inversao': float((difs < 0).mean()),
        'n_reamostragens_validas': int(len(difs)),
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--painel-csv', type=Path, default=None)
    ap.add_argument('--gt-csv', type=Path, default=None)
    ap.add_argument('--output-dir', type=Path,
                    default=PROJECT_ROOT / 'outputs' / 'decisive_experiment')
    return ap.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)

    df_gt, painel_path = carregar_referencia(args.painel_csv, args.gt_csv)
    df_gt['human_label_sensu_sensor'] = df_gt['human_label']

    linhas = []
    testes = []

    for convencao, mascara in RODADAS:
        rotulos, movidas = aplicar_convencao(
            df_gt, convencao=convencao, mascara=mascara,
            coluna_rotulo='human_label_sensu_sensor')
        df_gt['_h'] = rotulos

        for base_nome, base_mask, fontes in _recortes(df_gt):
            sub = df_gt[base_mask]
            humano = sub['_h'].astype(int).values
            kappas = {}
            for nome, coluna, inverter in FONTES_AUTOMATICAS:
                if nome not in fontes:
                    continue
                v = _valores(sub, coluna, inverter)
                kappas[nome] = (cohen_kappa_score(v, humano), v)
                linhas.append({
                    'convencao': convencao,
                    'mascara': mascara if convencao != 'sensu_sensor' else '',
                    'base': base_nome,
                    'n': int(len(sub)),
                    'n_reclassificadas': int(movidas[base_mask].sum()),
                    'fonte': nome,
                    'p0': observed_agreement(v, humano),
                    'kappa': float(cohen_kappa_score(v, humano)),
                    'ac1': gwet_ac1(v, humano),
                    'prevalence_index': prevalence_index(v, humano),
                    'bias_index': bias_index(v, humano),
                    'pct_real_fonte': float((v == 0).mean()),
                    'pct_real_referencia': float((humano == 0).mean()),
                })

            teste = bootstrap_diferenca(
                humano, kappas['dNBR Auto'][1], kappas['Weak Labels'][1], rng)
            teste.update({
                'convencao': convencao,
                'mascara': mascara if convencao != 'sensu_sensor' else '',
                'base': base_nome, 'n': int(len(sub)),
                'kappa_dnbr': float(kappas['dNBR Auto'][0]),
                'kappa_weak': float(kappas['Weak Labels'][0]),
                'lider_pontual': ('dNBR Auto' if kappas['dNBR Auto'][0] >= kappas['Weak Labels'][0]
                                  else 'Weak Labels'),
            })
            testes.append(teste)

    df = pd.DataFrame(linhas)
    df.to_csv(args.output_dir / 'sensibilidade_convencao.csv', index=False)

    pd.set_option('display.width', 200)
    for base_nome in df['base'].unique():
        print(f"\n=== Kappa contra a referencia humana, base {base_nome} ===")
        pivo = df[df.base == base_nome].pivot_table(
            index='fonte', columns=['convencao', 'mascara'], values='kappa')
        print(pivo.round(4).to_string())

    print("\n=== Bootstrap pareado, Kappa(dNBR) menos Kappa(weak label) ===")
    df_testes = pd.DataFrame(testes)
    print(df_testes[['convencao', 'mascara', 'base', 'n', 'kappa_dnbr', 'kappa_weak',
                     'diferenca_media', 'ic95_inferior', 'ic95_superior',
                     'p_inversao', 'lider_pontual']].round(4).to_string(index=False))

    with open(args.output_dir / 'sensibilidade_convencao.json', 'w') as f:
        json.dump({
            'painel_csv': str(painel_path),
            'gt_csv': str(args.gt_csv) if args.gt_csv else 'padrao (validation_manual_gt.csv)',
            'semente': SEMENTE,
            'n_bootstrap': N_BOOTSTRAP,
            'nota': 'Os pares entre fontes automaticas nao mudam com a convencao. '
                    'A convencao reprojeta apenas a referencia humana.',
            'metricas': linhas,
            'testes_pareados': testes,
        }, f, indent=2, default=str)
    print(f"\n[OK] {args.output_dir / 'sensibilidade_convencao.csv'}")
    print(f"[OK] {args.output_dir / 'sensibilidade_convencao.json'}")


if __name__ == '__main__':
    main()
