"""
Regra composta de classificacao dNBR, fonte unica de verdade.

Esta e a regra que define a estrategia de rotulagem "dNBR automatico" (coluna
`manual_label`) descrita no Cap. 3. Ela estava embutida em
`auto_validate_dnbr.classify_point`, um modulo que chama `ee.Initialize()` no
import e portanto nao podia ser reusado pelos scripts de analise. Foi extraida
para ca para que MATOPIBA e Alentejo apliquem exatamente a mesma regra.

Convencao de rotulo: 0 = fogo real, 1 = espurio.

Escala de `pct_burned`: esta regra espera PERCENTUAL, 0 a 100, que e a escala
gravada pelo pipeline MATOPIBA (`compute_dnbr_stats` multiplica por 100). O
pipeline de Portugal grava fracao, 0 a 1. Use `to_percent_scale` antes de
chamar `classify_dnbr` quando a origem dos dados nao for o pipeline MATOPIBA.
"""

import numpy as np


def to_percent_scale(pct_values):
    """Normaliza `pct_burned` para percentual (0 a 100).

    Detecta a escala pelo maximo observado: uma serie inteiramente contida em
    [0, 1] esta em fracao e e multiplicada por 100. Uma serie que ja passa de 1
    esta em percentual e volta inalterada.

    Retorna (valores_em_percentual, escala_detectada).
    """
    arr = np.asarray(pct_values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(finite) <= 1.0:
        return arr * 100.0, 'fraction'
    return arr, 'percent'


def classify_dnbr(mean, max_val, pct):
    """Regra composta de oito condicoes. Retorna (label, confianca, evidencia, nota).

    `pct` em percentual (0 a 100).
    """
    # --- FOGO REAL (label=0) --- limiares mais estritos com buffer de 500 m
    if mean > 0.25 and pct > 40:
        return (0, 5, 2, f'dNBR vermelho intenso, cicatriz evidente (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    if mean > 0.20 and pct > 25:
        return (0, 4, 2, f'dNBR laranja/vermelho, cicatriz visivel (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    if mean > 0.15 and pct > 20:
        return (0, 3, 1, f'dNBR laranja, queimada provavel (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    # --- ESPURIO (label=1) ---
    if mean < 0.05 and max_val < 0.15:
        return (1, 5, 0, f'dNBR verde, sem alteracao visivel (mean={mean:.3f}, max={max_val:.3f})')

    if mean < 0.08 and max_val < 0.20 and pct < 10:
        return (1, 4, 0, f'dNBR verde, sem evidencia significativa (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    if mean < 0.10 and pct < 15:
        return (1, 3, 0, f'dNBR verde/amarelo, evidencia insuficiente (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    # --- INCERTO --- faixa estreita entre 0,10 e 0,15
    if 0.10 <= mean < 0.15:
        return (1, 2, 1, f'dNBR amarelo, evidencia ambigua (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    # Fallback: max alto com mean baixo, sinal localizado
    if max_val > 0.30 and pct > 15:
        return (0, 2, 1, f'dNBR sinal localizado (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    return (1, 2, 1, f'dNBR inconclusivo (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')


def classify_series(dnbr_mean, dnbr_max, pct_burned):
    """Aplica `classify_dnbr` elemento a elemento, normalizando a escala de pct.

    Retorna (labels ndarray int, escala_detectada).
    """
    pct, scale = to_percent_scale(pct_burned)
    mean = np.asarray(dnbr_mean, dtype=float)
    mx = np.asarray(dnbr_max, dtype=float)
    labels = np.array([classify_dnbr(a, b, c)[0] for a, b, c in zip(mean, mx, pct)], dtype=int)
    return labels, scale
