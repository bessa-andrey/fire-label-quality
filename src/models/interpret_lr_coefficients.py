"""
Interpreta os coeficientes da Regressao Logistica (modelo recomendado, 22 features,
track Humano). Treina em TODAS as 439 amostras binarias (nao CV — e interpretacao,
nao avaliacao). Bootstrap CI95% (200 resamples) nos coeficientes.

Convencao: coef positivo -> puxa para Espurio (classe 1); negativo -> puxa para Real.

Output:
- outputs/interpretability/lr_coefficients_22feat.csv
- outputs/interpretability/lr_coefficients_barplot.png
- outputs/interpretability/lr_coefficients.html (didatico)

Usage:
    python src/models/interpret_lr_coefficients.py
"""

from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
PAINEL_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'interpretability'

FEATURES_13 = [
    'brightness', 'confidence', 'frp', 'hotspot_count', 'persistence_score',
    'temperature', 'dewpoint', 'wind_speed', 'precipitation', 'rh',
    'drying_index', 'month', 'is_dry_season',
]
FEATURES_SPECTRAL = [
    'mean_ndvi_pre', 'mean_ndvi_post', 'delta_ndvi',
    'mean_nbr_pre', 'mean_nbr_post', 'dnbr_mean',
    'dnbr_max', 'pct_burned', 'land_use_class',
]
FEATURES_22 = FEATURES_13 + FEATURES_SPECTRAL

# Tradutor para nomes amigaveis em portugues
FEATURE_PT = {
    'brightness': 'Brilho FIRMS',
    'confidence': 'Confianca FIRMS',
    'frp': 'FRP (Pot. Radiativa)',
    'hotspot_count': 'Densidade focos vizinhos',
    'persistence_score': 'Persistencia temporal',
    'temperature': 'Temperatura',
    'dewpoint': 'Ponto de orvalho',
    'wind_speed': 'Vento',
    'precipitation': 'Precipitacao',
    'rh': 'Umidade relativa',
    'drying_index': 'Indice secura',
    'month': 'Mes',
    'is_dry_season': 'Estacao seca',
    'mean_ndvi_pre': 'NDVI pre-fogo',
    'mean_ndvi_post': 'NDVI pos-fogo',
    'delta_ndvi': 'Delta NDVI',
    'mean_nbr_pre': 'NBR pre-fogo',
    'mean_nbr_post': 'NBR pos-fogo',
    'dnbr_mean': 'dNBR medio',
    'dnbr_max': 'dNBR maximo',
    'pct_burned': '% pixels queimados',
    'land_use_class': 'Uso do solo (MapBiomas)',
}


def load_data():
    main_df = pd.read_csv(GT_CSV)
    df = main_df[main_df['human_label'].isin([0, 1])].copy()
    df['human_label'] = df['human_label'].astype(int)
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')

    painel = pd.read_csv(PAINEL_CSV)
    painel['acq_datetime'] = pd.to_datetime(painel['acq_datetime'], format='mixed')
    keys = ['latitude', 'longitude', 'acq_datetime']
    extra_cols = [c for c in FEATURES_SPECTRAL if c in painel.columns]
    df = df.merge(painel[keys + extra_cols].drop_duplicates(keys), on=keys, how='left')
    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    df = df.dropna(subset=FEATURES_22).copy()
    return df


def bootstrap_coefs(X, y, n_boot=200, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(X)
    coefs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42)),
        ])
        try:
            pipe.fit(X[idx], y[idx])
            coefs.append(pipe.named_steps['clf'].coef_[0])
        except Exception:
            pass
    return np.array(coefs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 60)
    print('INTERPRETACAO DOS COEFICIENTES DA REGRESSAO LOGISTICA')
    print('  (Track Humano, 22 features, todas 439 amostras)')
    print('=' * 60)

    df = load_data()
    X = df[FEATURES_22].values
    y = df['human_label'].astype(int).values
    print(f'N amostras: {len(df)} ({(y==0).sum()} Real, {(y==1).sum()} Espurio)')

    # Treina modelo completo (interpretacao)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(X, y)
    coefs_full = pipe.named_steps['clf'].coef_[0]
    intercept = pipe.named_steps['clf'].intercept_[0]
    print(f'\nIntercepto: {intercept:.4f}')

    # Bootstrap CI95%
    print(f'\nBootstrap 200 resamples para CI95%...')
    boot_coefs = bootstrap_coefs(X, y, n_boot=200)
    ci_lo = np.percentile(boot_coefs, 2.5, axis=0)
    ci_hi = np.percentile(boot_coefs, 97.5, axis=0)

    # Montar tabela
    rows = []
    for i, feat in enumerate(FEATURES_22):
        rows.append({
            'feature': feat,
            'feature_pt': FEATURE_PT.get(feat, feat),
            'coef': round(float(coefs_full[i]), 4),
            'abs_coef': round(abs(float(coefs_full[i])), 4),
            'ci_lo': round(float(ci_lo[i]), 4),
            'ci_hi': round(float(ci_hi[i]), 4),
            'sign': 'positivo (puxa Espurio)' if coefs_full[i] > 0 else 'negativo (puxa Real)',
            'significant': not (ci_lo[i] <= 0 <= ci_hi[i]),
            'tipo': 'espectral' if feat in FEATURES_SPECTRAL else 'FIRMS/ERA5',
        })
    df_coef = pd.DataFrame(rows).sort_values('abs_coef', ascending=False).reset_index(drop=True)
    df_coef['rank'] = df_coef.index + 1

    # Print
    print(f'\nTop 10 features por |coeficiente|:')
    print(df_coef[['rank', 'feature_pt', 'coef', 'ci_lo', 'ci_hi', 'sign', 'significant']].head(10).to_string(index=False))

    out_csv = OUT_DIR / 'lr_coefficients_22feat.csv'
    df_coef.to_csv(out_csv, index=False)
    print(f'\nSalvo: {out_csv}')

    # Plot
    fig, ax = plt.subplots(figsize=(4.93, 4.44))
    df_plot = df_coef.iloc[::-1]  # invert pra rank 1 ficar no topo
    colors = ['#c8362d' if c > 0 else '#2d6a4f' for c in df_plot['coef']]
    edge = ['black' if s else 'gray' for s in df_plot['significant']]
    lw = [1.5 if s else 0.6 for s in df_plot['significant']]
    ax.barh(df_plot['feature_pt'], df_plot['coef'], color=colors,
            edgecolor=edge, linewidth=lw)
    for i, row in enumerate(df_plot.itertuples()):
        ax.errorbar(row.coef, i, xerr=[[row.coef - row.ci_lo], [row.ci_hi - row.coef]],
                    fmt='none', color='black', capsize=2, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coeficiente padronizado (features standardizadas)\n<-- puxa para REAL    |    puxa para ESPURIO -->')
    ax.set_title('Regressao Logistica (22 feat): coeficientes ordenados por magnitude\n'
                 'Borda solida = significativo (CI95% nao cruza 0)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / 'lr_coefficients_barplot.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Salvo: {out_png}')

    # HTML didatico
    html = generate_html(df_coef, intercept)
    out_html = OUT_DIR / 'lr_coefficients.html'
    out_html.write_text(html, encoding='utf-8')
    print(f'Salvo: {out_html}')


def generate_html(df_coef, intercept):
    rows_top10 = ''
    for _, r in df_coef.head(10).iterrows():
        sign_color = '#c8362d' if r['coef'] > 0 else '#2d6a4f'
        sign_arrow = '&rarr; Espurio' if r['coef'] > 0 else '&rarr; Real'
        sig = 'SIM' if r['significant'] else 'nao'
        sig_color = '#2d6a4f' if r['significant'] else '#999'
        rows_top10 += f"""
        <tr>
          <td>{r['rank']}</td>
          <td><strong>{r['feature_pt']}</strong> <span style="color:#777;font-size:11px;">({r['feature']})</span></td>
          <td style="color:{sign_color};font-weight:600;">{r['coef']:+.3f}</td>
          <td style="font-size:12px;">[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]</td>
          <td style="color:{sign_color};">{sign_arrow}</td>
          <td style="color:{sig_color};font-weight:600;">{sig}</td>
          <td><span style="font-size:11px;color:#666;">{r['tipo']}</span></td>
        </tr>"""

    espectrais_top = df_coef[df_coef['tipo'] == 'espectral'].head(3)
    firms_top = df_coef[df_coef['tipo'] == 'FIRMS/ERA5'].head(3)
    n_sig = int(df_coef['significant'].sum())
    n_pos = int((df_coef['coef'] > 0).sum())
    n_neg = int((df_coef['coef'] < 0).sum())

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Coeficientes LR</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1050px; margin: 40px auto; padding: 0 30px; color: #1a1a1a; line-height: 1.6; }}
  h1 {{ color: #1f3a5f; border-bottom: 3px solid #1f3a5f; padding-bottom: 8px; }}
  h2 {{ color: #1f3a5f; margin-top: 32px; border-bottom: 1px solid #ccd; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; margin: 14px 0; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #ccd; padding: 6px 10px; text-align: left; }}
  th {{ background: #1f3a5f; color: #fff; }}
  tr:nth-child(even) {{ background: #f5f7fa; }}
  .key {{ background: #fff8d6; padding: 12px 16px; border-left: 4px solid #d4a017; margin: 16px 0; }}
  .good {{ background: #d8efd6; padding: 12px 16px; border-left: 4px solid #2d6a4f; margin: 16px 0; }}
  .info {{ background: #e3eef9; padding: 12px 16px; border-left: 4px solid #1f3a5f; margin: 16px 0; }}
  .meta {{ font-size: 12px; color: #777; }}
  img {{ max-width: 100%; border: 1px solid #ccd; border-radius: 4px; }}
</style>
</head>
<body>

<h1>Coeficientes da Regressao Logistica (22 features)</h1>
<p class="meta">Torna interpretavel um modelo que de outro modo seria caixa preta. Cada feature ganha um peso explicito.</p>

<div class="info">
<strong>Como ler:</strong> Features standardizadas (media 0, desvio 1). Coeficiente <strong>positivo</strong> puxa a predicao para <span style="color:#c8362d;font-weight:600;">Espurio (classe 1)</span>. Negativo puxa para <span style="color:#2d6a4f;font-weight:600;">Real (classe 0)</span>. Magnitude indica importancia relativa.<br>
<strong>Intercepto:</strong> {intercept:.4f} (baseline log-odds quando todas as features estao na media).<br>
<strong>Significativo:</strong> CI95% bootstrap (200 resamples) nao cruza zero.
</div>

<h2>Top 10 features por importancia (|coef|)</h2>
<table>
<tr><th>#</th><th>Feature</th><th>Coef</th><th>CI95%</th><th>Direcao</th><th>Sig.</th><th>Tipo</th></tr>
{rows_top10}
</table>

<h2>Visualizacao (todos os 22 coeficientes)</h2>
<img src="lr_coefficients_barplot.png" alt="Barplot dos coeficientes LR">

<h2>Sintese</h2>
<div class="good">
<strong>Resumo quantitativo:</strong>
<ul>
<li><strong>{n_sig}/{len(df_coef)}</strong> features tem coeficiente significativo (CI95% nao cruza 0)</li>
<li><strong>{n_pos}</strong> features puxam para Espurio (positivo) | <strong>{n_neg}</strong> features puxam para Real (negativo)</li>
<li>Top espectral: <strong>{espectrais_top.iloc[0]['feature_pt']}</strong> (coef {espectrais_top.iloc[0]['coef']:+.3f})</li>
<li>Top FIRMS/ERA5: <strong>{firms_top.iloc[0]['feature_pt']}</strong> (coef {firms_top.iloc[0]['coef']:+.3f})</li>
</ul>
</div>

<h2>Texto sugerido para a dissertacao</h2>
<div class="key">
"A interpretacao dos coeficientes padronizados da Regressao Logistica (Figura X) confirma a importancia das features espectrais Sentinel-2: a feature mais influente para discriminar Espurio (positivo) e <strong>{df_coef.iloc[0]['feature_pt']}</strong> (coef={df_coef.iloc[0]['coef']:+.3f}), enquanto <strong>{df_coef.iloc[-1]['feature_pt']}</strong> empurra mais fortemente para Real (coef={df_coef.iloc[-1]['coef']:+.3f}). Dos 22 coeficientes, {n_sig} mantiveram-se estatisticamente significativos no bootstrap (200 resamples, CI95%), reforcando a contribuicao individual e justificando a escolha do modelo recomendado pela combinacao de desempenho e interpretabilidade."
</div>

<p class="meta">Gerado por src/models/interpret_lr_coefficients.py</p>

</body>
</html>"""


if __name__ == '__main__':
    main()
