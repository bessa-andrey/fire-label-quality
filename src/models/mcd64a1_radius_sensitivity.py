"""
Sensibilidade ao raio de busca MCD64A1 para weak labels.

Substituto metodologico para FireCCI51 (que cobre apenas 2001-2020 e nao serve para o
periodo 2022-2024 deste trabalho).

Testa a hipotese 'a resolucao 500m do MCD64A1 e o gargalo':
- Para cada amostra humana, verifica se ALGUM pixel MCD64A1 dentro de raio R esta queimado
- Computa is_burned_R (1 se queimado dentro de R, 0 caso contrario)
- Convenção: is_reliable_R = is_burned_R; espurio se NAO queimado dentro do raio
- Compara is_reliable_R contra human_label via Cohen's Kappa
- Varia R em [250, 500, 1000, 2000] metros

Output:
  outputs/decisive_experiment/mcd64a1_radius_sensitivity.csv
  outputs/decisive_experiment/mcd64a1_radius_sensitivity_summary.csv

Usage:
  python src/models/mcd64a1_radius_sensitivity.py

NOTA (22/08/2026). Este script cobre apenas as amostras da referencia humana e usa um
criterio de janela temporal que depende do dia do mes. `filterDate(D-15, D+15)` seleciona
compostos mensais pelo `system:time_start`, que em um produto mensal e o primeiro dia do
mes, entao a janela de mais ou menos 15 dias seleciona zero, um ou dois compostos conforme
a data da deteccao, e o teste `BurnDate > 0` nao confere a data da queima. Deteccoes no dia
17 de um mes de 31 dias ficam sem composto, o que explica as 14 amostras sem resposta da
saida de 27/05/2026. `src/preprocessing/recompute_weak_labels_mcd64a1.py` estende a escada
de raios ao painel completo, reproduz este criterio na coluna `mcd64a1_burned_r{R}_monthly`
e implementa a janela de mais ou menos 15 dias de fato em `mcd64a1_burned_r{R}`.
"""

from __future__ import annotations
from pathlib import Path
import time

import ee
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'decisive_experiment'

RADII_METERS = [250, 500, 1000, 2000]
TEMPORAL_WINDOW_DAYS = 15

# ============================================================
# GEE: MCD64A1 query
# ============================================================

def init_gee():
    ee.Initialize()
    return ee.ImageCollection('MODIS/061/MCD64A1')


def query_mcd64a1_any_burn(coll, lat, lon, date_str, radius_m, days=TEMPORAL_WINDOW_DAYS):
    """Retorna 1 se ALGUM pixel MCD64A1 dentro de radius_m esta queimado em [date-15d, date+15d]."""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(radius_m)
    date = ee.Date(date_str)
    start = date.advance(-days, 'day')
    end = date.advance(days, 'day')

    # MCD64A1 'BurnDate' band: 0 = not burned, >0 = day of year burned
    filtered = coll.filterDate(start, end).filterBounds(region)
    size = filtered.size().getInfo()
    if size == 0:
        return None
    # Reduce: max BurnDate over time series and region
    img = filtered.select('BurnDate').max()
    stats = img.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=region,
        scale=500,  # native MCD64A1 resolution
        maxPixels=1e9,
    ).getInfo()
    val = stats.get('BurnDate', None)
    if val is None:
        return 0
    return int(val) > 0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('MCD64A1 RADIUS SENSITIVITY (substituto para FireCCI51)')
    print('=' * 60)

    coll = init_gee()
    print(f'MCD64A1 v6.1 inicializado.')

    df_full = pd.read_csv(GT_CSV)
    df = df_full[df_full['human_label'].isin([0, 1])].copy().reset_index(drop=True)
    df['date_str'] = pd.to_datetime(df['acq_datetime'], format='mixed').dt.strftime('%Y-%m-%d')
    print(f'Amostras humanas binarias: n={len(df)}')

    results = {r: [] for r in RADII_METERS}
    n_total = len(df)

    t0 = time.time()
    for i, row in df.iterrows():
        for r in RADII_METERS:
            try:
                v = query_mcd64a1_any_burn(coll, row['latitude'], row['longitude'], row['date_str'], r)
                results[r].append(v)
            except Exception as e:
                results[r].append(None)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            print(f'  [{i+1}/{n_total}] {rate:.2f} amostras/s, ETA {eta/60:.1f} min')
        if (i + 1) % 10 == 0:
            time.sleep(0.5)  # gentle rate limiting

    # Save raw results
    rows = []
    for i, row in df.iterrows():
        for r in RADII_METERS:
            rows.append({
                'sample_id': row.get('sample_id', i),
                'lat': row['latitude'],
                'lon': row['longitude'],
                'date': row['date_str'],
                'human_label': int(row['human_label']),
                'radius_m': r,
                'is_burned_R': results[r][i] if i < len(results[r]) else None,
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / 'mcd64a1_radius_sensitivity.csv', index=False)
    print(f'\nSalvo: {OUT_DIR / "mcd64a1_radius_sensitivity.csv"}')

    # Compute Kappa per radius
    print('\n--- Kappa weak_R vs human_label por raio ---')
    summary_rows = []
    for r in RADII_METERS:
        df_r = df_out[(df_out['radius_m'] == r) & df_out['is_burned_R'].notna()].copy()
        # weak label convention: is_burned (1) -> Real (0); not burned (0) -> Espurio (1)
        # i.e. weak = 1 - is_burned
        weak = 1 - df_r['is_burned_R'].astype(int)
        human = df_r['human_label'].astype(int)
        k = cohen_kappa_score(human, weak)
        p0 = (human == weak).mean()
        n = len(df_r)
        # Bootstrap CI
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(500):
            idx = rng.choice(n, size=n, replace=True)
            try:
                bk = cohen_kappa_score(human.iloc[idx].values, weak.iloc[idx].values)
                boots.append(bk)
            except Exception:
                pass
        klo = float(np.percentile(boots, 2.5)) if boots else float('nan')
        khi = float(np.percentile(boots, 97.5)) if boots else float('nan')
        print(f'  R={r:5d}m: n={n}, p0={p0:.3f}, k={k:+.3f} [{klo:+.3f},{khi:+.3f}]')
        summary_rows.append({
            'radius_m': r, 'n': n, 'p0': round(p0, 4),
            'kappa': round(k, 4), 'kappa_ci_lo': round(klo, 4), 'kappa_ci_hi': round(khi, 4),
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'mcd64a1_radius_sensitivity_summary.csv', index=False)
    print(f'Salvo: {OUT_DIR / "mcd64a1_radius_sensitivity_summary.csv"}')


if __name__ == '__main__':
    main()
