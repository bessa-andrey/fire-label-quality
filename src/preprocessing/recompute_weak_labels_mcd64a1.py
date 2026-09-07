"""
Recomputa a Estrategia 1 (weak labels) pelo cruzamento FIRMS x MCD64A1.

CONTEXTO
--------
A coluna `is_reliable` distribuida em `data/processed/training/module_a_full.csv` e em
`outputs/ground_truth_validation/painel_labeled_*.csv` nao e o cruzamento FIRMS x MCD64A1
que o texto da dissertacao descreve. Ela e uma particao pela mediana de `reliability_score`,
escore construido com variaveis do proprio FIRMS, e nenhum codigo do repositorio a produz.
Este script existe para dar procedencia limpa a Estrategia 1: ele recomputa a coluna a
partir do painel de deteccoes e do produto de area queimada MCD64A1, com parametros e
semente declarados, e grava a coluna nova AO LADO da antiga, sem sobrescrever nada.

O QUE CONSOME
-------------
  outputs/ground_truth_validation/painel_labeled_5strategies.csv
      painel de 3.698 deteccoes FIRMS do MATOPIBA, ja com `is_reliable` (coluna antiga),
      `gt_painel` e `gt_mapbiomas_fire`
  data/processed/validation/validation_manual_gt.csv
      604 amostras da principal referencia de avaliacao, 535 binarias
  MODIS/061/MCD64A1 (Google Earth Engine)
      produto mensal de area queimada, banda BurnDate, resolucao nativa 500 m

O QUE PRODUZ
------------
  outputs/weak_labels_v2/mcd64a1_query_cache.csv
      cache bruto por (lat, lon, data, raio). Reexecucoes reaproveitam o cache
  outputs/weak_labels_v2/painel_weak_labels_v2.csv
      painel completo com as colunas novas ao lado da antiga
  outputs/weak_labels_v2/gt_weak_labels_v2.csv
      referencia humana com as colunas novas ao lado da antiga
  outputs/weak_labels_v2/weak_labels_v2_summary.csv
      escada de raios: proporcao de fogo real e concordancia contra o anotador principal
  outputs/weak_labels_v2/weak_labels_v2_report.json
      parametros, contagens e comparacao entre a coluna antiga e a nova
  outputs/weak_labels_v2/painel_labeled_5strategies_weakv2.csv
  outputs/weak_labels_v2/validation_manual_gt_weakv2.csv
      copias dos dois insumos com `is_reliable` ja apontando para a coluna nova, e com a
      coluna antiga preservada sob o nome `is_reliable_legado_mediana`. Servem de insumo
      direto para `run_decisive_experiment.py` e `compute_extended_concordance.py` pelas
      opcoes `--painel-csv` e `--gt-csv`

COLUNAS PRODUZIDAS
------------------
  mcd64a1_burned_r{R}          1 se algum pixel MCD64A1 dentro de R metros registra
                               cicatriz com data de queima dentro de [D-15, D+15].
                               Este e o criterio estrito, o que o texto descreve
  mcd64a1_burned_r{R}_monthly  1 sob o criterio por composto mensal, que e o que
                               `src/models/mcd64a1_radius_sensitivity.py` implementou em
                               mai/2026: seleciona os compostos mensais cujo inicio cai
                               em [D-15, D+15) e testa BurnDate > 0, sem conferir a data
                               de queima. Mantido para reproduzir a saida de mai/2026 e
                               para servir de comparacao
  is_reliable_mcd64a1          alias de `mcd64a1_burned_r500`, o rotulo primario da
                               Estrategia 1 recomputada. Mesma polaridade da coluna
                               antiga `is_reliable`, ou seja, 1 = fogo real, 0 = espurio
  is_reliable_mcd64a1_monthly  alias de `mcd64a1_burned_r500_monthly`

CONVENCAO DE ROTULO
-------------------
O projeto usa 0 = fogo real, 1 = espurio no rotulo de treino, e as colunas `is_reliable`
e `gt_painel` guardam a polaridade invertida (1 = fogo real). As colunas novas seguem a
polaridade de `is_reliable`, entao `y = 1 - is_reliable_mcd64a1` continua valendo nos
scripts a jusante.

PARAMETROS
----------
  raio primario     500 m, que e o que o texto descreve e o que a replicacao do Alentejo
                    usou em `src/portugal/select_and_extract_features.py`
  escada de raios   250, 500, 1000 e 2000 m
  janela temporal   mais ou menos 15 dias
  escala de reducao 500 m, resolucao nativa do MCD64A1
  semente           42, usada apenas nos intervalos de confianca por bootstrap

USO
---
  python src/preprocessing/recompute_weak_labels_mcd64a1.py
  python src/preprocessing/recompute_weak_labels_mcd64a1.py --target gt
  python src/preprocessing/recompute_weak_labels_mcd64a1.py --radii 500 --no-cache

CREDENCIAIS
-----------
Usa `ee.Initialize()`. Quando o arquivo padrao de credenciais nao esta no lugar esperado,
aponte `EE_CREDENTIALS_FILE` para o JSON de credenciais e `EE_PROJECT` para o projeto.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PAINEL_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_5strategies.csv'
PAINEL_CSV_FALLBACK = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'
GT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'validation' / 'validation_manual_gt.csv'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'weak_labels_v2'
CACHE_CSV = OUT_DIR / 'mcd64a1_query_cache.csv'

COLLECTION_ID = 'MODIS/061/MCD64A1'
RADII_METERS = [250, 500, 1000, 2000]
PRIMARY_RADIUS_M = 500
TEMPORAL_WINDOW_DAYS = 15
REDUCE_SCALE_M = 500
SEED = 42
N_BOOTSTRAP = 500
DEFAULT_WORKERS = 8

# Coordenadas arredondadas nesta casa formam a chave do cache e a chave de juncao entre
# o painel e a referencia humana. Cinco casas decimais dao cerca de 1 m no equador.
COORD_DECIMALS = 5


# ============================================================
# Earth Engine
# ============================================================

def init_gee():
    """Inicializa o Earth Engine.

    Caminho normal: `ee.Initialize()`, que le o arquivo de credenciais padrao.
    Caminho alternativo: `EE_CREDENTIALS_FILE` apontando para o JSON de credenciais,
    util quando o processo roda em um ambiente que nao e o do usuario autenticado.
    """
    cred_file = os.environ.get('EE_CREDENTIALS_FILE')
    if cred_file:
        import ee.oauth as oauth
        from google.oauth2.credentials import Credentials
        blob = json.load(open(cred_file))
        creds = Credentials(
            None,
            refresh_token=blob['refresh_token'],
            token_uri=oauth.TOKEN_URI,
            client_id=blob.get('client_id', oauth.CLIENT_ID),
            client_secret=blob.get('client_secret', oauth.CLIENT_SECRET),
            scopes=blob.get('scopes', oauth.SCOPES),
            quota_project_id=blob.get('project'),
        )
        project = os.environ.get('EE_PROJECT') or blob.get('project')
        ee.Initialize(credentials=creds, project=project)
    else:
        project = os.environ.get('EE_PROJECT')
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    return ee.ImageCollection(COLLECTION_ID)


def available_months(coll):
    """Conjunto de (ano, mes) presentes na colecao, obtido em uma unica consulta."""
    stamps = coll.aggregate_array('system:time_start').getInfo()
    out = set()
    for ms in stamps:
        d = datetime.utcfromtimestamp(ms / 1000.0)
        out.add((d.year, d.month))
    return out


def _month_start(y, m):
    return date(y, m, 1)


def _next_month(y, m):
    return (y + 1, 1) if m == 12 else (y, m + 1)


def _months_overlapping(d0: date, d1: date):
    """Meses (ano, mes) que intersectam o intervalo [d0, d1]."""
    out = []
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        out.append((y, m))
        y, m = _next_month(y, m)
    return out


def build_masks_for_date(coll, months_avail, acq: date, window_days=TEMPORAL_WINDOW_DAYS):
    """Monta a imagem de mascaras para uma data de aquisicao.

    Devolve (imagem, tem_estrito, tem_mensal). A imagem carrega ate duas bandas:

      strict   1 onde algum pixel registra queima com data dentro de [D-15, D+15].
               Usa o proprio valor de BurnDate, que e o dia juliano da queima, entao a
               janela de mais ou menos 15 dias vale de fato
      monthly  1 onde BurnDate > 0 nos compostos mensais cujo inicio cai em
               [D-15, D+15). Reproduz o criterio de `mcd64a1_radius_sensitivity.py`
    """
    w0 = acq - timedelta(days=window_days)
    w1 = acq + timedelta(days=window_days)

    bands = []

    # --- criterio estrito ---
    # A selecao de compostos e alargada em um mes para cada lado da janela porque o
    # produto mensal MCD64A1 reporta queimas com data proxima da virada do mes tanto no
    # composto do mes quanto no do mes vizinho. O que decide o rotulo e sempre o valor de
    # BurnDate, entao incluir compostos a mais so aumenta a cobertura, nunca cria
    # correspondencia fora da janela de mais ou menos 15 dias.
    strict_parts = []
    for (y, m) in _months_overlapping(w0 - timedelta(days=31), w1 + timedelta(days=31)):
        if (y, m) not in months_avail:
            continue
        lo_date = max(w0, date(y, 1, 1))
        hi_date = min(w1, date(y, 12, 31))
        if lo_date > hi_date:
            continue
        lo_doy = lo_date.timetuple().tm_yday
        hi_doy = hi_date.timetuple().tm_yday
        ny, nm = _next_month(y, m)
        img = ee.Image(
            coll.filterDate(str(_month_start(y, m)), str(_month_start(ny, nm)))
                .select('BurnDate').first()
        )
        strict_parts.append(img.gte(lo_doy).And(img.lte(hi_doy)))
    has_strict = len(strict_parts) > 0
    if has_strict:
        bands.append(ee.ImageCollection(strict_parts).max().rename('strict'))
    else:
        # Banda inteiramente mascarada. As duas bandas precisam existir sempre, porque em
        # uma imagem de banda unica o reduceRegions nomeia a propriedade pelo redutor e nao
        # pela banda, e a leitura por nome de banda passaria a devolver nulo em silencio.
        bands.append(ee.Image.constant(0).selfMask().rename('strict'))

    # --- criterio por composto mensal, como em mai/2026 ---
    # `filterDate(start, end)` seleciona imagens cujo system:time_start cai em [start, end).
    # Em um composto mensal isso e o primeiro dia do mes, entao a janela de mais ou menos
    # 15 dias seleciona zero, um ou dois compostos conforme o dia do mes da deteccao.
    monthly_months = [
        (y, m) for (y, m) in _months_overlapping(w0, w1)
        if (y, m) in months_avail and w0 <= _month_start(y, m) < w1
    ]
    has_monthly = len(monthly_months) > 0
    if has_monthly:
        first = monthly_months[0]
        last_next = _next_month(*monthly_months[-1])
        sub = coll.filterDate(
            str(_month_start(*first)), str(_month_start(*last_next))
        ).select('BurnDate')
        bands.append(sub.max().gt(0).rename('monthly'))
    else:
        bands.append(ee.Image.constant(0).selfMask().rename('monthly'))

    if not (has_strict or has_monthly):
        return None, False, False
    return ee.Image.cat(bands), has_strict, has_monthly


def query_one_date(coll, months_avail, acq_str, points, radii, window_days):
    """Consulta todos os pontos de uma data, em todos os raios, com uma unica chamada.

    points e uma lista de (lat, lon). Devolve lista de dicionarios prontos para o cache.
    """
    acq = datetime.strptime(acq_str, '%Y-%m-%d').date()
    image, has_strict, has_monthly = build_masks_for_date(coll, months_avail, acq, window_days)

    rows = []
    if image is None:
        for (lat, lon) in points:
            for r in radii:
                rows.append({'lat': lat, 'lon': lon, 'date': acq_str, 'radius_m': r,
                             'strict': np.nan, 'monthly': np.nan})
        return rows

    feats = []
    for i, (lat, lon) in enumerate(points):
        for r in radii:
            feats.append(ee.Feature(
                ee.Geometry.Point([lon, lat]).buffer(r),
                {'pid': i, 'radius_m': r},
            ))
    fc = ee.FeatureCollection(feats)
    reduced = image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.max(),
        scale=REDUCE_SCALE_M,
    ).getInfo()

    got = {}
    for f in reduced['features']:
        p = f['properties']
        got[(int(p['pid']), int(p['radius_m']))] = p

    def _read(props, band, present):
        """Le uma banda do resultado do reduceRegions.

        A propriedade vem ausente ou nula quando nenhum pixel valido cai no buffer. O
        script de mai/2026 tratava esse caso como nao queimado, e a leitura e mantida.
        NaN fica reservado para o caso em que a banda nao existe, ou seja, quando a
        janela nao seleciona nenhum composto.
        """
        if not present:
            return np.nan
        v = props.get(band)
        return 0.0 if v is None else float(v)

    for i, (lat, lon) in enumerate(points):
        for r in radii:
            p = got.get((i, r), {})
            strict = _read(p, 'strict', has_strict)
            monthly = _read(p, 'monthly', has_monthly)
            rows.append({'lat': lat, 'lon': lon, 'date': acq_str, 'radius_m': r,
                         'strict': strict, 'monthly': monthly})
    return rows


# ============================================================
# Carga dos insumos
# ============================================================

def _key_frame(df, lat_col='latitude', lon_col='longitude', date_col='acq_datetime'):
    out = df.copy()
    out['_lat'] = out[lat_col].round(COORD_DECIMALS)
    out['_lon'] = out[lon_col].round(COORD_DECIMALS)
    out['_date'] = pd.to_datetime(out[date_col], format='mixed').dt.strftime('%Y-%m-%d')
    return out


def load_inputs(target):
    painel = gt = None
    if target in ('painel', 'both'):
        path = PAINEL_CSV if PAINEL_CSV.exists() else PAINEL_CSV_FALLBACK
        painel = _key_frame(pd.read_csv(path))
        print(f'[OK] painel: {path.name}, n={len(painel)}')
    if target in ('gt', 'both'):
        gt = _key_frame(pd.read_csv(GT_CSV))
        print(f'[OK] referencia humana: {GT_CSV.name}, n={len(gt)}, '
              f'binarias={(gt["human_label"].isin([0, 1])).sum()}')
    return painel, gt


def collect_points(frames):
    pts = set()
    for f in frames:
        if f is None:
            continue
        pts.update(zip(f['_lat'], f['_lon'], f['_date']))
    return sorted(pts, key=lambda t: (t[2], t[0], t[1]))


# ============================================================
# Cache
# ============================================================

def load_cache(path):
    if not path.exists():
        return pd.DataFrame(columns=['lat', 'lon', 'date', 'radius_m', 'strict', 'monthly'])
    df = pd.read_csv(path)
    df['lat'] = df['lat'].round(COORD_DECIMALS)
    df['lon'] = df['lon'].round(COORD_DECIMALS)
    return df


def cache_index(df):
    return set(zip(df['lat'], df['lon'], df['date'], df['radius_m']))


# ============================================================
# Metricas
# ============================================================

def kappa_with_ci(y_ref, y_pred, seed=SEED, n_boot=N_BOOTSTRAP):
    y_ref = np.asarray(y_ref, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = float(cohen_kappa_score(y_ref, y_pred))
    rng = np.random.default_rng(seed)
    n = len(y_ref)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_ref[idx])) < 2 and len(np.unique(y_pred[idx])) < 2:
            continue
        boots.append(cohen_kappa_score(y_ref[idx], y_pred[idx]))
    lo = float(np.percentile(boots, 2.5)) if boots else float('nan')
    hi = float(np.percentile(boots, 97.5)) if boots else float('nan')
    return k, lo, hi, float((y_ref == y_pred).mean())


# ============================================================
# Main
# ============================================================

def attach_columns(df, cache, radii):
    """Cola as colunas novas em df, casando por (lat, lon, data)."""
    out = df.copy()
    for r in radii:
        sub = cache[cache['radius_m'] == r][['lat', 'lon', 'date', 'strict', 'monthly']]
        sub = sub.rename(columns={
            'strict': f'mcd64a1_burned_r{r}',
            'monthly': f'mcd64a1_burned_r{r}_monthly',
        })
        out = out.merge(sub, left_on=['_lat', '_lon', '_date'],
                        right_on=['lat', 'lon', 'date'], how='left')
        out = out.drop(columns=['lat', 'lon', 'date'], errors='ignore')
    out['is_reliable_mcd64a1'] = out[f'mcd64a1_burned_r{PRIMARY_RADIUS_M}']
    out['is_reliable_mcd64a1_monthly'] = out[f'mcd64a1_burned_r{PRIMARY_RADIUS_M}_monthly']
    return out


def write_dropin(df_out, path, source_col='is_reliable_mcd64a1'):
    """Grava uma copia do insumo com `is_reliable` ja apontando para a coluna nova.

    A coluna antiga sobrevive sob o nome `is_reliable_legado_mediana`, que descreve o que
    ela e de fato, a particao pela mediana de `reliability_score`. Os arquivos originais
    nao sao tocados. Estes arquivos servem de insumo direto para
    `run_decisive_experiment.py` e `compute_extended_concordance.py` pelas opcoes
    `--painel-csv` e `--gt-csv`, sem que nenhum desses scripts precise saber da troca.

    Sao gravados dois pares, um por criterio: `source_col` igual a `is_reliable_mcd64a1`
    produz o par do criterio estrito, e `is_reliable_mcd64a1_monthly` produz o par do
    criterio por composto mensal, que serve de comparacao.
    """
    out = df_out.drop(columns=[c for c in df_out.columns if c.startswith('_')]).copy()
    out = out.rename(columns={'is_reliable': 'is_reliable_legado_mediana'})
    n_sem = int(out[source_col].isna().sum())
    out['is_reliable'] = out[source_col].fillna(0).astype(int)
    out.to_csv(path, index=False)
    print(f'[OK] {path.name} ({len(out)} linhas, fonte {source_col}, '
          f'{n_sem} sem resposta tratadas como espurio)')


def summarize(gt, painel, radii):
    """Escada de raios: proporcao de fogo real e concordancia contra o anotador principal."""
    rows = []
    for variant, suffix in (('estrito', ''), ('composto mensal', '_monthly')):
        for r in radii:
            col = f'mcd64a1_burned_r{r}{suffix}'
            rec = {'variante': variant, 'radius_m': r}
            if painel is not None:
                v = painel[col]
                rec.update({
                    'painel_n': int(len(v)),
                    'painel_n_valido': int(v.notna().sum()),
                    'painel_real': int(v.fillna(0).sum()),
                    'painel_pct_real': round(float(v.fillna(0).mean()) * 100, 2),
                })
            if gt is not None:
                g = gt[gt['human_label'].isin([0, 1])]
                g = g[g[col].notna()]
                if len(g) >= 10:
                    weak = (1 - g[col]).astype(int).values   # 0 = fogo real
                    human = g['human_label'].astype(int).values
                    k, lo, hi, p0 = kappa_with_ci(human, weak)
                    rec.update({
                        'gt_n': int(len(g)),
                        'gt_real': int(g[col].sum()),
                        'gt_pct_real': round(float(g[col].mean()) * 100, 2),
                        'gt_p0': round(p0, 4),
                        'gt_kappa': round(k, 4),
                        'gt_kappa_ci_lo': round(lo, 4),
                        'gt_kappa_ci_hi': round(hi, 4),
                    })
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--target', choices=['gt', 'painel', 'both'], default='both')
    ap.add_argument('--radii', type=int, nargs='+', default=RADII_METERS)
    ap.add_argument('--window-days', type=int, default=TEMPORAL_WINDOW_DAYS)
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    ap.add_argument('--no-cache', action='store_true', help='ignora o cache e reconsulta tudo')
    ap.add_argument('--cache', type=Path, default=CACHE_CSV)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 72)
    print('RECOMPUTA A ESTRATEGIA 1: FIRMS x MCD64A1')
    print('=' * 72)
    print(f'raios={args.radii} m, janela=+/-{args.window_days} d, '
          f'escala={REDUCE_SCALE_M} m, semente={SEED}')

    painel, gt = load_inputs(args.target)
    points = collect_points([painel, gt])
    by_date = {}
    for lat, lon, d in points:
        by_date.setdefault(d, []).append((lat, lon))
    print(f'[OK] pontos unicos: {len(points)}, datas unicas: {len(by_date)}')

    cache = load_cache(args.cache) if not args.no_cache else load_cache(Path('/nonexistent'))
    have = cache_index(cache)
    pending = {
        d: [(lat, lon) for (lat, lon) in pts
            if any((lat, lon, d, r) not in have for r in args.radii)]
        for d, pts in by_date.items()
    }
    pending = {d: p for d, p in pending.items() if p}
    n_pending = sum(len(p) for p in pending.values())
    print(f'[OK] em cache: {len(points) - n_pending} pontos, a consultar: {n_pending} '
          f'em {len(pending)} datas')

    if pending:
        coll = init_gee()
        months = available_months(coll)
        print(f'[OK] {COLLECTION_ID}: {len(months)} meses disponiveis, '
              f'de {min(months)} a {max(months)}')

        t0 = time.time()
        done = [0]
        results = []

        def work(item):
            d, pts = item
            for attempt in range(3):
                try:
                    return query_one_date(coll, months, d, pts, args.radii, args.window_days)
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        print(f'  [ERRO] data {d}: {exc}')
                        return [{'lat': lat, 'lon': lon, 'date': d, 'radius_m': r,
                                 'strict': np.nan, 'monthly': np.nan}
                                for (lat, lon) in pts for r in args.radii]
                    time.sleep(2 * (attempt + 1))
            return []

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rows in pool.map(work, sorted(pending.items())):
                results.extend(rows)
                done[0] += 1
                if done[0] % 25 == 0:
                    el = time.time() - t0
                    rate = done[0] / el
                    print(f'  [{done[0]}/{len(pending)}] {rate:.2f} datas/s, '
                          f'ETA {(len(pending) - done[0]) / rate / 60:.1f} min')
        elapsed = time.time() - t0
        print(f'[OK] consulta concluida em {elapsed/60:.1f} min '
              f'({len(results)} linhas de resultado)')

        new = pd.DataFrame(results)
        new['lat'] = new['lat'].round(COORD_DECIMALS)
        new['lon'] = new['lon'].round(COORD_DECIMALS)
        cache = pd.concat([cache, new], ignore_index=True)
        cache = cache.drop_duplicates(subset=['lat', 'lon', 'date', 'radius_m'], keep='last')
        cache.to_csv(args.cache, index=False)
        print(f'[OK] cache: {args.cache} ({len(cache)} linhas)')

    painel_out = attach_columns(painel, cache, args.radii) if painel is not None else None
    gt_out = attach_columns(gt, cache, args.radii) if gt is not None else None

    report = {
        'gerado_em': datetime.now().isoformat(timespec='seconds'),
        'parametros': {
            'colecao': COLLECTION_ID,
            'raios_m': args.radii,
            'raio_primario_m': PRIMARY_RADIUS_M,
            'janela_dias': args.window_days,
            'escala_reducao_m': REDUCE_SCALE_M,
            'semente': SEED,
            'n_bootstrap': N_BOOTSTRAP,
        },
    }

    if painel_out is not None:
        drop = [c for c in painel_out.columns if c.startswith('_')]
        painel_out.drop(columns=drop).to_csv(OUT_DIR / 'painel_weak_labels_v2.csv', index=False)
        write_dropin(painel_out, OUT_DIR / 'painel_labeled_5strategies_weakv2.csv')
        write_dropin(painel_out, OUT_DIR / 'painel_labeled_5strategies_weakv2_monthly.csv',
                     source_col='is_reliable_mcd64a1_monthly')
        old = painel_out['is_reliable']
        new_col = painel_out['is_reliable_mcd64a1'].fillna(0)
        report['painel'] = {
            'n': int(len(painel_out)),
            'antiga_is_reliable_real': int((old == 1).sum()),
            'antiga_is_reliable_pct_real': round(float((old == 1).mean()) * 100, 2),
            'nova_is_reliable_mcd64a1_real': int(new_col.sum()),
            'nova_is_reliable_mcd64a1_pct_real': round(float(new_col.mean()) * 100, 2),
            'nova_sem_resposta': int(painel_out['is_reliable_mcd64a1'].isna().sum()),
            'concordancia_antiga_vs_nova': round(
                float(((old == 1) == (new_col == 1)).mean()), 4),
            'kappa_antiga_vs_nova': round(float(cohen_kappa_score(
                (old == 1).astype(int), (new_col == 1).astype(int))), 4),
        }
        print(f"[OK] painel_weak_labels_v2.csv, "
              f"antiga {report['painel']['antiga_is_reliable_pct_real']}% real, "
              f"nova {report['painel']['nova_is_reliable_mcd64a1_pct_real']}% real")

    if gt_out is not None:
        drop = [c for c in gt_out.columns if c.startswith('_')]
        gt_out.drop(columns=drop).to_csv(OUT_DIR / 'gt_weak_labels_v2.csv', index=False)
        write_dropin(gt_out, OUT_DIR / 'validation_manual_gt_weakv2.csv')
        write_dropin(gt_out, OUT_DIR / 'validation_manual_gt_weakv2_monthly.csv',
                     source_col='is_reliable_mcd64a1_monthly')
        g = gt_out[gt_out['human_label'].isin([0, 1])]
        old = g['is_reliable']
        new_col = g['is_reliable_mcd64a1'].fillna(0)
        k_old, lo_old, hi_old, p0_old = kappa_with_ci(
            g['human_label'].astype(int), (1 - old).astype(int))
        k_new, lo_new, hi_new, p0_new = kappa_with_ci(
            g['human_label'].astype(int), (1 - new_col).astype(int))
        report['referencia_humana'] = {
            'n_binarias': int(len(g)),
            'antiga_pct_real': round(float((old == 1).mean()) * 100, 2),
            'antiga_p0': round(p0_old, 4),
            'antiga_kappa': round(k_old, 4),
            'antiga_kappa_ci': [round(lo_old, 4), round(hi_old, 4)],
            'nova_real': int(new_col.sum()),
            'nova_pct_real': round(float(new_col.mean()) * 100, 2),
            'nova_p0': round(p0_new, 4),
            'nova_kappa': round(k_new, 4),
            'nova_kappa_ci': [round(lo_new, 4), round(hi_new, 4)],
        }
        print(f"[OK] gt_weak_labels_v2.csv, "
              f"antiga kappa {k_old:.4f}, nova kappa {k_new:.4f}")

    summary = summarize(gt_out, painel_out, args.radii)
    summary.to_csv(OUT_DIR / 'weak_labels_v2_summary.csv', index=False)
    print('\n--- escada de raios ---')
    print(summary.to_string(index=False))

    with open(OUT_DIR / 'weak_labels_v2_report.json', 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f'\n[OK] {OUT_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
