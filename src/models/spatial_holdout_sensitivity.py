"""
Spatial holdout sensitivity: LOSO com (i) shapefile poligonal IBGE/MATOPIBA
e (ii) restricao a 2022-only (sem imputacao land_use_class).

Duas fontes de sensibilidade tratadas aqui:
  Imputacao land_use_class 2023/24 -> rodar 22 feat em 2022-only (n=439)
  Bounding box approximation -> usar matopiba_limites.geojson

Outputs:
  outputs/decisive_experiment/spatial_holdout_sensitivity.csv
  outputs/decisive_experiment/spatial_holdout_sensitivity_summary.csv
  outputs/decisive_experiment/spatial_holdout_state_assignment_comparison.csv
  outputs/decisive_experiment/spurious_prevalence_by_state_polygon.csv

Usage:
  python src/models/spatial_holdout_sensitivity.py
"""

from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
PAINEL_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'
SHAPEFILE = PROJECT_ROOT / 'data' / 'matopiba_limites.geojson'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'decisive_experiment'

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


def classify_state_bbox(lat, lon):
    """Aproximacao por bounding box (metodo legado)."""
    if lat < -10: return 'BA'
    if lat >= -10 and lon >= -45: return 'PI'
    if lat >= -7 and lon < -45: return 'MA'
    if lat >= -13 and lat < -7 and lon < -45: return 'TO'
    return 'UNK'


# `matopiba_limites.geojson` identifica cada feicao pelo nome por extenso, na
# coluna `name`. O codigo esperava uma coluna `estado` com a sigla, que nao
# existe no arquivo, e o script quebrava em `load_shapefile`.
NOME_PARA_UF = {
    'Maranhão': 'MA',
    'Tocantins': 'TO',
    'Piauí': 'PI',
    'Bahia': 'BA',
}


def load_shapefile():
    gdf = gpd.read_file(SHAPEFILE)
    if 'estado' not in gdf.columns:
        gdf['estado'] = gdf['name'].map(NOME_PARA_UF)
    # A feicao agregada MATOPIBA, quando presente, nao entra na particao por estado.
    sub = gdf[gdf['estado'].notna() & (gdf['name'] != 'MATOPIBA')].copy()
    faltando = set(NOME_PARA_UF.values()) - set(sub['estado'])
    if faltando:
        raise RuntimeError(f'Estados ausentes em {SHAPEFILE.name}: {sorted(faltando)}. '
                           f'Feicoes encontradas: {gdf["name"].tolist()}')
    sub = sub[['estado', 'geometry']].reset_index(drop=True)
    return sub


def assign_state_polygon(df, gdf_states):
    """Atribui cada ponto ao estado via point-in-polygon."""
    pts = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])],
        crs='EPSG:4326',
    )
    pts_gdf = gpd.GeoDataFrame({'_idx': df.index}, geometry=pts, crs='EPSG:4326')
    joined = gpd.sjoin(pts_gdf, gdf_states[['estado', 'geometry']], how='left', predicate='within')
    # Drop duplicates from sjoin (point on border may match multiple)
    joined = joined.drop_duplicates('_idx').set_index('_idx')
    return joined['estado'].fillna('UNK')


def bootstrap_kappa_ci(y_true, y_pred, n_boot=500, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            k = cohen_kappa_score(y_true[idx], y_pred[idx])
            boots.append(k)
        except Exception:
            pass
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def prepare_data(features, gdf_states, year_filter=None):
    main_df = pd.read_csv(GT_CSV)
    df = main_df[main_df['human_label'].isin([0, 1])].copy()
    df['human_label'] = df['human_label'].astype(int)
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')

    if any(f in FEATURES_SPECTRAL for f in features):
        painel = pd.read_csv(PAINEL_CSV)
        painel['acq_datetime'] = pd.to_datetime(painel['acq_datetime'], format='mixed')
        keys = ['latitude', 'longitude', 'acq_datetime']
        extra_cols = [c for c in FEATURES_SPECTRAL if c in painel.columns]
        df = df.merge(painel[keys + extra_cols].drop_duplicates(keys), on=keys, how='left')

    df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    # Estado por bounding box (legacy) e por shapefile (correto)
    df['estado_bbox'] = df.apply(lambda r: classify_state_bbox(r['latitude'], r['longitude']), axis=1)
    df['estado_poly'] = assign_state_polygon(df.reset_index(drop=True), gdf_states).values

    if year_filter is not None:
        df = df[df['acq_datetime'].dt.year == year_filter].copy()
    df = df.dropna(subset=features).copy()
    return df


def run_loso(scenario_name, features, df, state_col):
    print(f'\n{"=" * 60}')
    print(f'SCENARIO {scenario_name}  ({len(features)} feat, state={state_col}, n={len(df)})')
    print(f'{"=" * 60}')
    estados = sorted([e for e in df[state_col].unique() if e not in ('UNK', None)])
    rows = []
    for estado_test in estados:
        train_df = df[df[state_col] != estado_test]
        test_df = df[df[state_col] == estado_test]
        if len(test_df) < 20 or test_df['human_label'].nunique() < 2:
            print(f'[{estado_test}] skip: n={len(test_df)}, classes={test_df["human_label"].nunique()}')
            continue
        X_train = train_df[features].values
        y_train = train_df['human_label'].values
        X_test = test_df[features].values
        y_test = test_df['human_label'].values

        for name, model in get_models().items():
            clf = clone(model)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            kappa = cohen_kappa_score(y_test, y_pred)
            acc = (y_pred == y_test).mean()
            klo, khi = bootstrap_kappa_ci(y_test, y_pred, n_boot=500)
            rep = classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
            rows.append({
                'scenario': scenario_name,
                'state_col': state_col,
                'state_test': estado_test,
                'model': name,
                'n_features': len(features),
                'n_train': len(train_df),
                'n_test': len(test_df),
                'kappa': round(kappa, 4),
                'kappa_ci_lo': round(klo, 4),
                'kappa_ci_hi': round(khi, 4),
                'accuracy': round(acc, 4),
                'f1_real': round(rep['0']['f1-score'], 4),
                'f1_espurio': round(rep['1']['f1-score'], 4),
            })
            print(f'  [{estado_test}] {name:25s} k={kappa:+.3f} [{klo:+.3f},{khi:+.3f}] acc={acc:.3f}')
    return rows


def prevalencia_espurio_por_estado(df_22, df_22_2022, df_13):
    """Prevalencia de Espurio segundo o anotador principal, por estado.

    Sob a particao poligonal (`estado_poly`), que e a adotada na dissertacao.
    Este insumo nao existia em nenhuma saida do projeto. Ele responde quanto da
    variacao de Kappa entre estados no LOSO pode vir da marginal de cada estado,
    e nao da dificuldade do estado. A linha UNK cobre as deteccoes que caem fora
    dos quatro poligonos.
    """
    rows = []
    for rotulo, frame in (('22feat_todos_anos', df_22),
                          ('22feat_2022', df_22_2022),
                          ('13feat_todos_anos', df_13)):
        for estado, g in frame.groupby('estado_poly'):
            n = len(g)
            n_esp = int((g['human_label'] == 1).sum())
            n_real = int((g['human_label'] == 0).sum())
            # IC 95% de Wilson para a proporcao
            z = 1.959963985
            phat = n_esp / n if n else float('nan')
            den = 1 + z ** 2 / n if n else float('nan')
            centro = (phat + z ** 2 / (2 * n)) / den if n else float('nan')
            meia = (z / den) * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) if n else float('nan')
            rows.append({
                'conjunto': rotulo,
                'estado': estado,
                'n': n,
                'n_real': n_real,
                'n_espurio': n_esp,
                'prevalencia_espurio': round(phat, 4),
                'ic95_lo': round(max(centro - meia, 0.0), 4),
                'ic95_hi': round(min(centro + meia, 1.0), 4),
            })
        n = len(frame)
        rows.append({
            'conjunto': rotulo, 'estado': 'TOTAL', 'n': n,
            'n_real': int((frame['human_label'] == 0).sum()),
            'n_espurio': int((frame['human_label'] == 1).sum()),
            'prevalencia_espurio': round(float((frame['human_label'] == 1).mean()), 4),
            'ic95_lo': None, 'ic95_hi': None,
        })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('SPATIAL HOLDOUT SENSITIVITY: bbox vs shapefile, 2022-only vs all')
    print('=' * 60)

    gdf_states = load_shapefile()
    print(f'\nShapefile estados carregados: {len(gdf_states)} ({gdf_states["estado"].tolist()})')

    df_22 = prepare_data(FEATURES_22, gdf_states, year_filter=None)
    print(f'\nData 22 feat all years: n={len(df_22)}')
    df_22_2022 = prepare_data(FEATURES_22, gdf_states, year_filter=2022)
    print(f'Data 22 feat 2022-only: n={len(df_22_2022)}')
    df_13 = prepare_data(FEATURES_13, gdf_states, year_filter=None)
    print(f'Data 13 feat all years: n={len(df_13)}')

    # State assignment comparison
    cmp = pd.crosstab(df_22['estado_bbox'], df_22['estado_poly'], margins=True)
    print('\n--- Comparacao bbox vs polygon (22 feat all years) ---')
    print(cmp)
    cmp.to_csv(OUT_DIR / 'spatial_holdout_state_assignment_comparison.csv')

    prev = prevalencia_espurio_por_estado(df_22, df_22_2022, df_13)
    print('\n--- Prevalencia de espurio por estado, particao poligonal ---')
    print(prev.to_string(index=False))
    out_prev = OUT_DIR / 'spurious_prevalence_by_state_polygon.csv'
    prev.to_csv(out_prev, index=False)
    print(f'Salvo: {out_prev}')

    rows = []
    # 4 scenarios
    rows += run_loso('A_22feat_bbox_all',  FEATURES_22, df_22, 'estado_bbox')
    rows += run_loso('A_22feat_poly_all',  FEATURES_22, df_22, 'estado_poly')
    rows += run_loso('A_22feat_poly_2022', FEATURES_22, df_22_2022, 'estado_poly')
    rows += run_loso('B_13feat_poly_all',  FEATURES_13, df_13, 'estado_poly')

    df_out = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'spatial_holdout_sensitivity.csv'
    df_out.to_csv(out_csv, index=False)
    print(f'\nSalvo: {out_csv}')

    summary = df_out.groupby(['scenario', 'model']).agg(
        kappa_mean=('kappa', 'mean'),
        kappa_std=('kappa', 'std'),
        kappa_min=('kappa', 'min'),
        kappa_max=('kappa', 'max'),
        n_states=('state_test', 'count'),
        n_train_avg=('n_train', 'mean'),
        n_test_total=('n_test', 'sum'),
    ).round(4).reset_index()
    print('\n--- Summary ---')
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / 'spatial_holdout_sensitivity_summary.csv', index=False)
    print(f'Salvo: {OUT_DIR / "spatial_holdout_sensitivity_summary.csv"}')


if __name__ == '__main__':
    main()
