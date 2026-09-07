"""
Power analysis ex-post para o experimento decisivo.

Responde: qual diferenca de Kappa entre dois classificadores pareados o
desenho deste trabalho (n = 534) tem poder para detectar?

Duas rotas de calculo rodam lado a lado, e as duas tem de bater.

  Analitica    Rota principal, e a que alimenta a tabela do texto. Usa a mesma
               variancia sigma^2(kappa) = 1,26/n e a mesma covariancia pareada
               de 0,5*sigma^2 que ja sustentam a coluna n_min. Sob essas
               premissas Var(diferenca) = 2*sigma^2 - 2*(0,5*sigma^2) = sigma^2,
               e o poder sai fechado. Escolhida como rota principal para que as
               duas colunas da tabela venham das mesmas premissas: n_min e o
               poder saem do mesmo modelo, entao a tabela fica internamente
               consistente por construcao, e a leitura do texto ("o desenho nao
               discrimina os cinco modelos") nao depende da calibracao de uma
               simulacao.

  Monte Carlo  Conferencia independente. `simulate_paired_kappa` gera o efeito
               nominal pedido (ver a docstring da funcao).

Output:
- outputs/decisive_experiment/power_analysis_kappa.csv

Usage:
    python src/models/power_analysis_kappa.py            # analitico + MC
    python src/models/power_analysis_kappa.py --no-mc    # so analitico, rapido
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / 'outputs' / 'decisive_experiment'

PREV_SPURIOUS = 0.21
KAPPA_BASE = 0.50
N_SIM = 1000            # replicacoes Monte Carlo. O texto declara 10^3.
N_BOOT = 200            # reamostragens do teste bootstrap pareado dentro de cada replica
ALPHA = 0.05
POWER_TARGET = 0.80
N_STUDY = 534

# sigma^2(kappa) = 1,26/n, de Fleiss sob prevalencia 0,21 e kappa basal 0,50.
# Covariancia pareada tipica 0,5*sigma^2 entre modelos que compartilham features.
KAPPA_VAR_COEF = 1.26
PAIRED_COV_FRACTION = 0.5


def se_paired_diff(n, var_coef=KAPPA_VAR_COEF, cov_fraction=PAIRED_COV_FRACTION):
    """Erro padrao da diferenca de Kappa entre dois classificadores pareados."""
    sigma2 = var_coef / n
    var_diff = 2 * sigma2 - 2 * (cov_fraction * sigma2)
    return float(np.sqrt(var_diff))


def analytic_power(n, delta_kappa, alpha=ALPHA):
    """Poder de um teste bilateral para detectar `delta_kappa` com `n` amostras."""
    se = se_paired_diff(n)
    z_a = norm.ppf(1 - alpha / 2)
    lam = delta_kappa / se
    return float(norm.cdf(lam - z_a) + norm.cdf(-lam - z_a))


def analytic_n_min(delta_kappa, alpha=ALPHA, power=POWER_TARGET,
                   var_coef=KAPPA_VAR_COEF, cov_fraction=PAIRED_COV_FRACTION):
    """n necessario para detectar `delta_kappa` com o poder alvo."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    var_unit = 2 * var_coef - 2 * (cov_fraction * var_coef)
    return float(z ** 2 * var_unit / delta_kappa ** 2)


def analytic_mde(n=N_STUDY, alpha=ALPHA, power=POWER_TARGET):
    """Diferenca minima detectavel com o poder alvo em `n` amostras."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z * se_paired_diff(n))


def _realized_kappa(po, prev):
    """Kappa esperado quando o acerto e po e o erro inverte o rotulo verdadeiro.

    Inverter o rotulo desloca a marginal prevista para
    q = prev*po + (1-prev)*(1-po), o que muda a concordancia esperada. Ignorar
    esse deslocamento e usar p_e = prev^2 + (1-prev)^2 faz o Kappa realizado
    sair do alvo, que era um segundo motivo, alem da copia de `pred1`, para a
    simulacao antiga nao gerar o efeito nominal.
    """
    q = prev * po + (1 - prev) * (1 - po)
    p_e = prev * q + (1 - prev) * (1 - q)
    return (po - p_e) / (1 - p_e)


def _calibrate_agreement(k_target, prev):
    """Taxa de acerto po que produz exatamente `k_target` sob `_realized_kappa`."""
    lo = prev ** 2 + (1 - prev) ** 2
    return brentq(lambda po: _realized_kappa(po, prev) - k_target, lo, 1 - 1e-9)


def simulate_paired_kappa(n, kappa1, kappa2, prev, correlation=0.85, rng=None):
    """Simula duas predicoes binarias com os Kappas alvo, correlacionadas.

    A correlacao entra por copula gaussiana sobre os indicadores de acerto.
    Para cada amostra sorteia-se um par normal padrao com correlacao `correlation`
    e cada componente e cortada no quantil do respectivo p_o. Isso induz a
    dependencia desejada entre os dois classificadores e preserva EXATAMENTE a
    taxa de acerto marginal de cada um, logo preserva o Kappa alvo de cada um.
    """
    if rng is None:
        rng = np.random.default_rng()
    y_true = rng.binomial(1, prev, size=n)

    po1, po2 = _calibrate_agreement(kappa1, prev), _calibrate_agreement(kappa2, prev)

    cov = np.array([[1.0, correlation], [correlation, 1.0]])
    z = rng.multivariate_normal(np.zeros(2), cov, size=n)
    hit1 = z[:, 0] < norm.ppf(po1)
    hit2 = z[:, 1] < norm.ppf(po2)

    pred1 = np.where(hit1, y_true, 1 - y_true)
    pred2 = np.where(hit2, y_true, 1 - y_true)
    return y_true, pred1, pred2


def _kappa_vec(y, p):
    """Cohen's Kappa binario para cada linha de matrizes (n_boot, n)."""
    n = y.shape[1]
    po = (y == p).mean(axis=1)
    py1 = y.mean(axis=1)
    pp1 = p.mean(axis=1)
    pe = py1 * pp1 + (1 - py1) * (1 - pp1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(pe < 1, (po - pe) / (1 - pe), np.nan)


def simulated_effect(n, delta_kappa, n_rep=400, correlation=0.85, seed=42):
    """Efeito medio efetivamente gerado pela simulacao. Deve bater com o nominal."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_rep):
        y, p1, p2 = simulate_paired_kappa(n, KAPPA_BASE, KAPPA_BASE + delta_kappa,
                                          PREV_SPURIOUS, correlation, rng=rng)
        k = _kappa_vec(np.vstack([y, y]), np.vstack([p1, p2]))
        diffs.append(k[1] - k[0])
    return float(np.nanmean(diffs))


def estimate_power(n, delta_kappa, n_sim=N_SIM, alpha=ALPHA, correlation=0.85,
                   n_boot=N_BOOT, seed=42):
    """Poder por simulacao: fracao de replicas em que o IC bootstrap exclui zero."""
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_sim):
        y, p1, p2 = simulate_paired_kappa(n, KAPPA_BASE, KAPPA_BASE + delta_kappa,
                                          PREV_SPURIOUS, correlation, rng=rng)
        idx = rng.integers(0, n, size=(n_boot, n))
        diffs = _kappa_vec(y[idx], p2[idx]) - _kappa_vec(y[idx], p1[idx])
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            continue
        ci_lo = np.percentile(diffs, 100 * alpha / 2)
        ci_hi = np.percentile(diffs, 100 * (1 - alpha / 2))
        if ci_lo > 0 or ci_hi < 0:
            rejections += 1
    return rejections / n_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-mc', action='store_true',
                    help='pula a conferencia Monte Carlo, so a rota analitica')
    ap.add_argument('--n-sim', type=int, default=N_SIM)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Analise de poder ex-post (Cohen's Kappa, classificadores pareados)")
    print('=' * 78)
    print(f'Prevalencia de Espurio: {PREV_SPURIOUS}')
    print(f'Kappa basal: {KAPPA_BASE}')
    print(f'sigma^2(kappa) = {KAPPA_VAR_COEF}/n, covariancia pareada = '
          f'{PAIRED_COV_FRACTION}*sigma^2')
    print(f'Correlacao entre predicoes pareadas: 0,85')
    print(f'Alpha: {ALPHA}, poder alvo: {POWER_TARGET}, n do estudo: {N_STUDY}')
    if not args.no_mc:
        print(f'Monte Carlo: {args.n_sim} replicacoes, {N_BOOT} reamostragens por replica')
    print()

    deltas = [0.05, 0.10, 0.15, 0.20, 0.25]
    rows = []
    header = f"{'Delta_kappa':<12} {'n_min (80%)':>12} {'Poder em n=534':>15}"
    if not args.no_mc:
        header += f" {'Poder MC':>10} {'Efeito MC':>11}"
    print(header)
    print('-' * len(header))
    for d in deltas:
        n_min = analytic_n_min(d)
        power = analytic_power(N_STUDY, d)
        row = {
            'delta_kappa': d,
            'n_min_poder_80': round(n_min),
            'poder_analitico_n534': round(power, 4),
        }
        line = f'{d:<12.3f} {round(n_min):>12d} {power:>14.1%}'
        if not args.no_mc:
            eff = simulated_effect(N_STUDY, d)
            p_mc = estimate_power(N_STUDY, d, n_sim=args.n_sim)
            row['poder_mc_n534'] = round(p_mc, 4)
            row['efeito_nominal'] = d
            row['efeito_simulado'] = round(eff, 4)
            line += f' {p_mc:>9.1%} {eff:>+11.4f}'
        rows.append(row)
        print(line)

    mde = analytic_mde()
    print()
    print(f'Diferenca minima detectavel com {POWER_TARGET:.0%} de poder em '
          f'n = {N_STUDY}: Delta_kappa = {mde:.3f}')
    print(f'n necessario para Delta_kappa = 0,03: {round(analytic_n_min(0.03)):,}'
          .replace(',', '.'))
    print(f'n necessario para Delta_kappa = 0,13: {round(analytic_n_min(0.13)):,}'
          .replace(',', '.'))

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'power_analysis_kappa.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nSalvo: {out_csv}')

    meta = {
        'rota_principal': 'analitica',
        'sigma2_kappa': f'{KAPPA_VAR_COEF}/n',
        'fracao_covariancia_pareada': PAIRED_COV_FRACTION,
        'alpha': ALPHA,
        'poder_alvo': POWER_TARGET,
        'n_estudo': N_STUDY,
        'mde_poder_80_n534': round(mde, 4),
        'n_min_delta_0_03': round(analytic_n_min(0.03)),
        'n_min_delta_0_13': round(analytic_n_min(0.13)),
        'n_sim_monte_carlo': None if args.no_mc else args.n_sim,
        'linhas': rows,
    }
    out_json = OUT_DIR / 'power_analysis_kappa.json'
    out_json.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Salvo: {out_json}')


if __name__ == '__main__':
    main()
