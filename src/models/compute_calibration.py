"""
Avalia calibracao de probabilidades dos 5 modelos no track Humano (22 feat):
- Brier score (e Brier Skill Score vs baseline preditor da classe majoritaria)
- Reliability diagram (10 bins) para cada modelo
- 5-fold StratifiedKFold para obter predict_proba out-of-fold

Reportar como DIAGNOSTICO (sem aplicar Platt/Isotonic — escopo desta analise e
diagnostico, nao correcao).

Output:
- outputs/calibration/brier_scores.csv
- outputs/calibration/reliability_diagrams.png
- outputs/calibration/calibration.html (didatico)

Usage:
    python src/models/compute_calibration.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
PAINEL_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'calibration'

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


def get_models():
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            num_leaves=31, learning_rate=0.05, max_depth=10,
            n_estimators=200, random_state=42, n_jobs=-1, verbose=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, n_jobs=-1, eval_metric='logloss',
        ),
    }


def cv_proba(model, X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    proba = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = clone(model)
        clf.fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1]  # P(classe=1 = Espurio)
    return proba


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 60)
    print('CALIBRACAO DE PROBABILIDADES (Brier + Reliability)')
    print('=' * 60)

    df = load_data()
    X = df[FEATURES_22].values
    y = df['human_label'].astype(int).values
    print(f'N amostras: {len(df)} ({(y==0).sum()} Real, {(y==1).sum()} Espurio)')

    # Brier baseline = preditor de classe majoritaria (P=prevalencia)
    prevalencia_esp = float((y == 1).mean())
    brier_baseline = brier_score_loss(y, np.full_like(y, prevalencia_esp, dtype=float))
    print(f'\nPrevalencia Espurio: {prevalencia_esp:.3f}')
    print(f'Brier baseline (prevalencia constante): {brier_baseline:.4f}')

    rows = []
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, (name, model) in zip(axes, get_models().items()):
        print(f'\n  {name}...')
        proba = cv_proba(model, X, y)
        brier = brier_score_loss(y, proba)
        bss = 1 - (brier / brier_baseline)  # Brier Skill Score
        print(f'    Brier={brier:.4f} | BSS={bss:+.3f}')

        # Reliability
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy='quantile')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfeito')
        ax.plot(mean_pred, frac_pos, 'o-', color='#1f3a5f', label='Observado')
        ax.set_xlabel('Probabilidade media prevista')
        ax.set_ylabel('Fracao real positiva' if name == 'Logistic Regression' else '')
        ax.set_title(f'{name}\nBrier={brier:.3f} | BSS={bss:+.2f}', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc='upper left')

        rows.append({
            'model': name,
            'brier': round(brier, 4),
            'brier_baseline': round(brier_baseline, 4),
            'brier_skill_score': round(bss, 4),
            'mean_proba_predita': round(float(proba.mean()), 4),
            'prevalencia_real': round(prevalencia_esp, 4),
            'verdict': verdict(bss),
        })

    plt.tight_layout()
    out_png = OUT_DIR / 'reliability_diagrams.png'
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'\nSalvo: {out_png}')

    df_out = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'brier_scores.csv'
    df_out.to_csv(out_csv, index=False)
    print(f'Salvo: {out_csv}')

    # Tabela final
    print(f'\n--- Brier scores ---')
    print(df_out.to_string(index=False))

    # HTML
    out_html = OUT_DIR / 'calibration.html'
    out_html.write_text(generate_html(df_out, brier_baseline, prevalencia_esp, len(df)), encoding='utf-8')
    print(f'Salvo: {out_html}')


def verdict(bss):
    if bss > 0.20:
        return 'BOM'
    if bss > 0.05:
        return 'RAZOAVEL'
    if bss > -0.05:
        return 'NEUTRO'
    return 'RUIM'


def generate_html(df_out, brier_baseline, prevalencia, n):
    color_map = {'BOM': '#2d6a4f', 'RAZOAVEL': '#d4a017', 'NEUTRO': '#6c757d', 'RUIM': '#c8362d'}
    rows_html = ''
    for _, r in df_out.iterrows():
        c = color_map[r['verdict']]
        rows_html += f"""
        <tr>
          <td><strong>{r['model']}</strong></td>
          <td>{r['brier']:.4f}</td>
          <td style="color:{c};font-weight:600;">{r['brier_skill_score']:+.3f}</td>
          <td>{r['mean_proba_predita']:.3f}</td>
          <td><span style="background:{c};color:#fff;padding:3px 10px;border-radius:3px;font-weight:600;font-size:11px;">{r['verdict']}</span></td>
        </tr>"""

    best = df_out.loc[df_out['brier_skill_score'].idxmax()]
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Calibracao de Probabilidades</title>
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

<h1>Calibracao de Probabilidades</h1>
<p class="meta">Mede se P(y=Espurio) prevista corresponde a fracao real de Espurio.</p>

<div class="info">
<strong>Conceitos:</strong><br>
<strong>Brier score:</strong> erro quadratico medio entre probabilidade prevista e label real. Menor = melhor. Baseline (prever sempre a prevalencia {prevalencia:.3f}) = <strong>{brier_baseline:.4f}</strong>.<br>
<strong>Brier Skill Score (BSS):</strong> 1 - (Brier_modelo / Brier_baseline). <strong>BSS &gt; 0</strong> = melhor que baseline. BSS = 1 = perfeito.<br>
<strong>Reliability diagram:</strong> divide as predicoes em 10 bins de igual frequencia (quantile) e plota probabilidade media prevista vs fracao real positiva. Linha diagonal = perfeita calibracao.<br>
<br>
<strong>N = {n} amostras binarias, 22 features, 5-fold CV.</strong>
</div>

<h2>Resultados</h2>
<table>
<tr><th>Modelo</th><th>Brier</th><th>BSS</th><th>Prob. media prevista</th><th>Veredicto</th></tr>
{rows_html}
</table>

<h2>Reliability diagrams (5 modelos)</h2>
<img src="reliability_diagrams.png" alt="Reliability diagrams">

<div class="good">
<strong>Modelo melhor calibrado:</strong> <strong>{best['model']}</strong> (BSS = {best['brier_skill_score']:+.3f}).
</div>

<h2>Texto sugerido para a dissertacao</h2>
<div class="key">
"A calibracao das probabilidades preditas foi avaliada pelo Brier score e pelo reliability diagram (10 bins, quantile binning). O modelo recomendado (Regressao Logistica) atinge Brier = {df_out.iloc[0]['brier']:.3f} com Brier Skill Score = {df_out.iloc[0]['brier_skill_score']:+.3f} sobre o baseline da prevalencia, indicando calibracao {df_out.iloc[0]['verdict'].lower()}. Modelos baseados em arvores (RF, Extra Trees) tendem a apresentar predicoes mais concentradas nos extremos, padrao tipico e ja documentado na literatura. Esta analise serve como diagnostico; a aplicacao de calibracao isotonica ou Platt fica registrada como trabalho futuro para uso operacional das probabilidades."
</div>

<p class="meta">Gerado por src/models/compute_calibration.py</p>

</body>
</html>"""


if __name__ == '__main__':
    main()
