"""
Spatial holdout experiment: Leave-One-State-Out CV nos 4 estados do MATOPIBA.

Para cada estado (MA, TO, PI, BA):
  Treino: os 3 outros estados
  Teste: estado isolado

Mede generalizacao espacial (vs CV aleatorio que pode misturar regioes).
Roda em DOIS cenarios:

Cenario A (22 feat, todas amostras): permite features espectrais
Cenario B (13 feat, todas amostras): mais robusto em N

A particao por estado e feita por point-in-polygon sobre
`data/matopiba_limites.geojson`, que e a particao adotada na dissertacao. A
aproximacao por bounding box, usada na abordagem preliminar, continua sendo
calculada e gravada como coluna de comparacao, nunca como particao do LOSO.

O geojson identifica cada feicao pela coluna `name`, com o nome por extenso.
Nao existe coluna `estado`. Ver `spatial_holdout_sensitivity.py`, que sofria
do mesmo problema de esquema.

Output:
- outputs/decisive_experiment/spatial_holdout.csv
- outputs/decisive_experiment/spatial_holdout_summary.csv
- outputs/decisive_experiment/spatial_holdout_state_counts.csv
- outputs/decisive_experiment/spatial_holdout_state_profile.csv
- outputs/decisive_experiment/spatial_holdout.png

Usage:
    python src/models/spatial_holdout_experiment.py
"""

from __future__ import annotations

import json
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
    """Classificacao aproximada por bounding box (abordagem preliminar)."""
    if lat < -10: return 'BA'
    if lat >= -10 and lon >= -45: return 'PI'
    if lat >= -7 and lon < -45: return 'MA'
    if lat >= -13 and lat < -7 and lon < -45: return 'TO'
    return 'UNK'


# `matopiba_limites.geojson` nomeia cada feicao por extenso, na coluna `name`.
# Nao ha coluna `estado` com a sigla.
NOME_PARA_UF = {
    'Maranhao': 'MA',
    'Maranh\u00e3o': 'MA',
    'Tocantins': 'TO',
    'Piaui': 'PI',
    'Piau\u00ed': 'PI',
    'Bahia': 'BA',
}


def load_shapefile():
    gdf = gpd.read_file(SHAPEFILE)
    if 'estado' not in gdf.columns:
        gdf['estado'] = gdf['name'].map(NOME_PARA_UF)
    sub = gdf[gdf['estado'].notna() & (gdf['name'] != 'MATOPIBA')].copy()
    faltando = set(['MA', 'TO', 'PI', 'BA']) - set(sub['estado'])
    if faltando:
        raise RuntimeError(f'Estados ausentes em {SHAPEFILE.name}: {sorted(faltando)}. '
                           f'Feicoes encontradas: {gdf["name"].tolist()}')
    return sub[['estado', 'geometry']].reset_index(drop=True)


def contar_vertices(gdf):
    """Vertices totais do arquivo de limites, para registro na saida."""
    total = 0
    for geom in gdf.geometry:
        for poly in (geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]):
            total += len(poly.exterior.coords)
            total += sum(len(r.coords) for r in poly.interiors)
    return total


def assign_state_polygon(df, gdf_states):
    """Atribui cada ponto ao estado via point-in-polygon."""
    pts = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])],
        crs='EPSG:4326',
    )
    pts_gdf = gpd.GeoDataFrame({'_idx': df.index}, geometry=pts, crs='EPSG:4326')
    joined = gpd.sjoin(pts_gdf, gdf_states[['estado', 'geometry']], how='left', predicate='within')
    joined = joined.drop_duplicates('_idx').set_index('_idx')
    return joined['estado'].fillna('UNK')


def bootstrap_kappa_ci(y_true, y_pred, n_boot=1000, random_state=42):
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


def prepare_data(features, gdf_states):
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
    df['estado_bbox'] = df.apply(lambda r: classify_state_bbox(r['latitude'], r['longitude']), axis=1)
    df['estado_poly'] = assign_state_polygon(df.reset_index(drop=True), gdf_states).values
    # `estado` e a particao usada pelo LOSO, e ela e a poligonal.
    df['estado'] = df['estado_poly']
    df = df.dropna(subset=features).copy()
    return df


def run_scenario(scenario_name, features, df):
    print(f'\n{"=" * 60}')
    print(f'CENARIO {scenario_name} ({len(features)} features, n={len(df)})')
    print(f'{"=" * 60}')
    print(df.groupby(['estado', 'human_label']).size().unstack(fill_value=0))
    n_unk = int((df['estado'] == 'UNK').sum())
    print(f'Fora dos quatro poligonos (UNK): {n_unk} de {len(df)}. '
          f'LOSO roda com n = {len(df) - n_unk}.')

    estados = sorted([e for e in df['estado'].unique() if e != 'UNK'])
    rows = []
    for estado_test in estados:
        train_df = df[df['estado'] != estado_test]
        test_df = df[df['estado'] == estado_test]
        if len(test_df) < 20 or test_df['human_label'].nunique() < 2:
            print(f'\n[{estado_test}] Pulando: n_teste={len(test_df)}, classes={test_df["human_label"].nunique()}')
            continue
        print(f'\n[Test={estado_test}] train n={len(train_df)} ({(train_df["human_label"]==0).sum()}R, {(train_df["human_label"]==1).sum()}E) | test n={len(test_df)} ({(test_df["human_label"]==0).sum()}R, {(test_df["human_label"]==1).sum()}E)')

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
            report = classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
            rows.append({
                'scenario': scenario_name,
                'state_test': estado_test,
                'model': name,
                'n_features': len(features),
                'n_train': len(train_df),
                'n_test': len(test_df),
                'kappa': round(kappa, 4),
                'kappa_ci_lo': round(klo, 4),
                'kappa_ci_hi': round(khi, 4),
                'accuracy': round(acc, 4),
                'f1_real': round(report['0']['f1-score'], 4),
                'f1_espurio': round(report['1']['f1-score'], 4),
            })
            print(f'  {name:25s} Kappa={kappa:+.3f} [{klo:+.3f}, {khi:+.3f}] Acc={acc:.3f}')
    return rows


def contagens_por_estado(scenario_name, df):
    """n, Real, Espurio e prevalencia de espurio por estado, particao poligonal."""
    rows = []
    for estado, g in df.groupby('estado'):
        n = len(g)
        n_esp = int((g['human_label'] == 1).sum())
        rows.append({
            'scenario': scenario_name, 'estado': estado, 'n': n,
            'n_real': int((g['human_label'] == 0).sum()), 'n_espurio': n_esp,
            'prevalencia_espurio': round(n_esp / n, 4),
        })
    n = len(df)
    n_dentro = int((df['estado'] != 'UNK').sum())
    rows.append({
        'scenario': scenario_name, 'estado': 'TOTAL', 'n': n,
        'n_real': int((df['human_label'] == 0).sum()),
        'n_espurio': int((df['human_label'] == 1).sum()),
        'prevalencia_espurio': round(float((df['human_label'] == 1).mean()), 4),
    })
    rows.append({
        'scenario': scenario_name, 'estado': 'LOSO_n_efetivo', 'n': n_dentro,
        'n_real': None, 'n_espurio': None, 'prevalencia_espurio': None,
    })
    return rows


def perfil_por_estado(df_res, df_dados, scenario_name):
    """Acuracia media, Kappa medio e prevalencia por estado, mais as correlacoes.

    Reproduz o corte que o texto usa para argumentar que a variacao de acuracia
    entre estados vem do desbalanceamento local de classes.
    """
    sub = df_res[df_res['scenario'] == scenario_name]
    if sub.empty:
        return [], {}
    prev = (df_dados[df_dados['estado'] != 'UNK']
            .groupby('estado')['human_label'].mean())
    agg = sub.groupby('state_test').agg(
        acc_media=('accuracy', 'mean'),
        kappa_medio=('kappa', 'mean'),
        kappa_ci_lo_min=('kappa_ci_lo', 'min'),
        kappa_ci_hi_max=('kappa_ci_hi', 'max'),
        n_test=('n_test', 'first'),
    ).reset_index()
    agg['prevalencia_espurio'] = agg['state_test'].map(prev)
    agg['scenario'] = scenario_name
    corr_acc = float(agg['prevalencia_espurio'].corr(agg['acc_media']))
    corr_kappa = float(agg['prevalencia_espurio'].corr(agg['kappa_medio']))
    linhas = agg.round(4).to_dict('records')
    resumo = {
        'corr_prevalencia_acuracia': round(corr_acc, 4),
        'corr_prevalencia_kappa': round(corr_kappa, 4),
        'acc_min': round(float(agg['acc_media'].min()), 4),
        'acc_min_estado': agg.loc[agg['acc_media'].idxmin(), 'state_test'],
        'acc_max': round(float(agg['acc_media'].max()), 4),
        'acc_max_estado': agg.loc[agg['acc_media'].idxmax(), 'state_test'],
        'kappa_min': round(float(agg['kappa_medio'].min()), 4),
        'kappa_min_estado': agg.loc[agg['kappa_medio'].idxmin(), 'state_test'],
        'kappa_max': round(float(agg['kappa_medio'].max()), 4),
        'kappa_max_estado': agg.loc[agg['kappa_medio'].idxmax(), 'state_test'],
        'prev_min': round(float(agg['prevalencia_espurio'].min()), 4),
        'prev_max': round(float(agg['prevalencia_espurio'].max()), 4),
    }
    return linhas, resumo


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 60)
    print('SPATIAL HOLDOUT: Leave-One-State-Out MATOPIBA')
    print('=' * 60)

    gdf_states = load_shapefile()
    n_vert = contar_vertices(gdf_states)
    print(f'Limites: {SHAPEFILE.name}, {len(gdf_states)} poligonos, '
          f'{n_vert} vertices totais. Particao do LOSO: point-in-polygon.')

    all_rows = []

    df_a = prepare_data(FEATURES_22, gdf_states)
    all_rows += run_scenario('A_22feat', FEATURES_22, df_a)

    df_b = prepare_data(FEATURES_13, gdf_states)
    all_rows += run_scenario('B_13feat', FEATURES_13, df_b)

    df_out = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / 'spatial_holdout.csv'
    df_out.to_csv(out_csv, index=False)
    print(f'\nSalvo: {out_csv}')

    # Sumario por modelo (media + std sobre os 4 estados)
    print('\n--- Sumario por modelo (Kappa medio across estados) ---')
    summary = df_out.groupby(['scenario', 'model']).agg(
        kappa_mean=('kappa', 'mean'),
        kappa_std=('kappa', 'std'),
        kappa_min=('kappa', 'min'),
        kappa_max=('kappa', 'max'),
        n_states=('state_test', 'count'),
    ).round(4).reset_index()
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / 'spatial_holdout_summary.csv', index=False)

    # Contagens por estado, com a linha das amostras fora dos quatro poligonos
    counts = pd.DataFrame(
        contagens_por_estado('A_22feat', df_a) + contagens_por_estado('B_13feat', df_b)
    )
    print('\n--- n por estado (particao poligonal) ---')
    print(counts.to_string(index=False))
    counts.to_csv(OUT_DIR / 'spatial_holdout_state_counts.csv', index=False)

    # Perfil por estado: acuracia, Kappa e prevalencia, com as correlacoes
    perfis, resumos = [], {}
    for nome, dados in (('A_22feat', df_a), ('B_13feat', df_b)):
        linhas, resumo = perfil_por_estado(df_out, dados, nome)
        perfis += linhas
        resumos[nome] = resumo
    perfil = pd.DataFrame(perfis)
    print('\n--- Perfil por estado ---')
    print(perfil.to_string(index=False))
    perfil.to_csv(OUT_DIR / 'spatial_holdout_state_profile.csv', index=False)
    print('\n--- Correlacoes prevalencia x desempenho ---')
    for nome, r in resumos.items():
        print(f'{nome}: {r}')
    with open(OUT_DIR / 'spatial_holdout_state_profile.json', 'w') as fh:
        json.dump({'shapefile': SHAPEFILE.name, 'n_poligonos': len(gdf_states),
                   'n_vertices': n_vert, 'resumos': resumos}, fh, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, scenario, title in [
        (axes[0], 'A_22feat', f'22 features (n={int((df_a["estado"] != "UNK").sum())} no poligono)'),
        (axes[1], 'B_13feat', f'13 features (n={int((df_b["estado"] != "UNK").sum())} no poligono)'),
    ]:
        sub = df_out[df_out['scenario'] == scenario]
        if sub.empty:
            ax.set_title(title + '\n(sem dados)')
            continue
        pivot = sub.pivot(index='model', columns='state_test', values='kappa')
        pivot.plot(kind='bar', ax=ax, colormap='Set2')
        ax.set_title(f'Leave-One-State-Out: {title}')
        ax.set_ylabel("Cohen's Kappa")
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(title='Estado teste', loc='upper right', fontsize=9)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / 'spatial_holdout.png'
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f'Salvo: {out_png}')


if __name__ == '__main__':
    main()
