"""
Download de dados do Painel do Fogo (CENSIPAM) via WFS.

Baixa eventos historicos de incendio do GeoServer do CENSIPAM,
filtrando pela regiao MATOPIBA e periodo 2022-2024.

Camadas disponiveis:
  - tb_evento_extinto: Eventos historicos extintos (~490k total)
  - mv_evento_filtro: Eventos ativos/observacao
  - tb_evento_filtro_download: Eventos para download (ativos)
  - mv_deteccoes_classificadas: Deteccoes classificadas

Uso:
    python src/data_ingest/download_painel_fogo.py
    python src/data_ingest/download_painel_fogo.py --ano-inicio 2022 --ano-fim 2024
    python src/data_ingest/download_painel_fogo.py --layer tb_evento_extinto --formato csv
    python src/data_ingest/download_painel_fogo.py --count-only
"""

import argparse
import sys
import time
from pathlib import Path

import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "painel_do_fogo"

# GeoServer WFS endpoint do Painel do Fogo (CENSIPAM)
WFS_BASE = "https://panorama.sipam.gov.br/geoserver/painel_do_fogo/wfs"

# Bounding box da regiao MATOPIBA (MA, TO, PI, BA)
# Margem generosa para cobrir toda a regiao
MATOPIBA_BBOX = {
    "west": -50.5,
    "south": -16.0,
    "east": -41.0,
    "north": -2.0,
}

# Camadas principais
LAYERS = {
    "tb_evento_extinto": "Eventos historicos extintos (principal)",
    "mv_evento_filtro": "Eventos ativos/observacao",
    "tb_evento_filtro_download": "Eventos ativos para download",
    "mv_deteccoes_classificadas": "Deteccoes classificadas (atual)",
}

PAGE_SIZE = 5000


def build_cql_filter(ano_inicio=2022, ano_fim=2024, bbox=None):
    """Constroi filtro CQL para o WFS."""
    filters = []

    # Filtro temporal
    filters.append(f"dt_minima >= '{ano_inicio}-01-01T00:00:00Z'")
    filters.append(f"dt_minima <= '{ano_fim}-12-31T23:59:59Z'")

    # Filtro espacial (BBOX com SRID explicito - necessario para EPSG:4674)
    if bbox:
        filters.append(
            f"BBOX(geom,{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']},'EPSG:4674')"
        )

    return " AND ".join(filters)


def count_features(layer, cql_filter=None):
    """Conta o numero de features que atendem ao filtro."""
    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typeName": f"painel_do_fogo:{layer}",
        "resultType": "hits",
    }
    if cql_filter:
        params["CQL_FILTER"] = cql_filter

    try:
        resp = requests.get(WFS_BASE, params=params, timeout=60)
        resp.raise_for_status()

        # O resultado e XML com numberOfFeatures
        import re
        match = re.search(r'numberOfFeatures="(\d+)"', resp.text)
        if match:
            return int(match.group(1))

        match = re.search(r'numberMatched="(\d+)"', resp.text)
        if match:
            return int(match.group(1))

    except Exception as e:
        print(f"  [ERRO] Nao foi possivel contar features: {e}")

    return None


def download_geojson_page(layer, cql_filter=None, start_index=0, max_features=PAGE_SIZE):
    """Baixa uma pagina de features em GeoJSON."""
    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typeName": f"painel_do_fogo:{layer}",
        "outputFormat": "application/json",
        "startIndex": start_index,
        "maxFeatures": max_features,
    }
    if cql_filter:
        params["CQL_FILTER"] = cql_filter

    resp = requests.get(WFS_BASE, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def download_csv_page(layer, cql_filter=None, start_index=0, max_features=PAGE_SIZE):
    """Baixa uma pagina de features em CSV."""
    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typeName": f"painel_do_fogo:{layer}",
        "outputFormat": "csv",
        "startIndex": start_index,
        "maxFeatures": max_features,
    }
    if cql_filter:
        params["CQL_FILTER"] = cql_filter

    resp = requests.get(WFS_BASE, params=params, timeout=120)
    resp.raise_for_status()
    return resp.text


def geojson_to_dataframe(geojson):
    """Converte GeoJSON FeatureCollection em DataFrame."""
    features = geojson.get("features", [])
    if not features:
        return pd.DataFrame()

    rows = []
    for f in features:
        props = f.get("properties", {})
        geom = f.get("geometry", {})

        # Extrair centroide do poligono
        if geom and geom.get("coordinates"):
            coords = geom["coordinates"]
            geom_type = geom.get("type", "")

            if geom_type == "Polygon" and coords:
                ring = coords[0]
                props["centroid_lon"] = sum(c[0] for c in ring) / len(ring)
                props["centroid_lat"] = sum(c[1] for c in ring) / len(ring)
            elif geom_type == "MultiPolygon" and coords:
                # Usar o primeiro poligono
                ring = coords[0][0]
                props["centroid_lon"] = sum(c[0] for c in ring) / len(ring)
                props["centroid_lat"] = sum(c[1] for c in ring) / len(ring)

        rows.append(props)

    return pd.DataFrame(rows)


def download_all(layer, cql_filter=None, formato="geojson"):
    """Baixa todos os features com paginacao."""
    total = count_features(layer, cql_filter)

    if total is None:
        print("  [AVISO] Nao foi possivel contar. Tentando download direto...")
        total = 999999
    else:
        print(f"  Total de eventos encontrados: {total:,}")

    if total == 0:
        print("  Nenhum evento encontrado com os filtros aplicados.")
        return None

    all_dfs = []
    downloaded = 0
    page = 0

    while downloaded < total:
        page += 1
        start = downloaded
        remaining = total - downloaded
        batch_size = min(PAGE_SIZE, remaining)

        print(f"  Pagina {page}: baixando {start+1}-{start+batch_size} de {total:,}...", end="", flush=True)

        try:
            if formato == "csv":
                csv_text = download_csv_page(layer, cql_filter, start, batch_size)
                from io import StringIO
                df_page = pd.read_csv(StringIO(csv_text))
            else:
                geojson = download_geojson_page(layer, cql_filter, start, batch_size)
                df_page = geojson_to_dataframe(geojson)
                n_returned = geojson.get("numberReturned", len(df_page))

            if len(df_page) == 0:
                print(" fim dos dados.")
                break

            all_dfs.append(df_page)
            downloaded += len(df_page)
            print(f" OK ({len(df_page)} eventos)")

            # Pausa entre requisicoes para nao sobrecarregar o servidor
            if downloaded < total:
                time.sleep(1)

        except Exception as e:
            print(f" ERRO: {e}")
            if page > 1:
                print(f"  Continuando com {downloaded} eventos baixados...")
                break
            else:
                return None

    if not all_dfs:
        return None

    df = pd.concat(all_dfs, ignore_index=True)

    # Limpar colunas com referencias Java quebradas
    for col in df.columns:
        if df[col].dtype == object:
            mask = df[col].astype(str).str.contains(r"\[Ljava\.lang\.", na=False)
            if mask.any():
                df[col] = df[col].where(~mask, None)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download de dados do Painel do Fogo (CENSIPAM) via WFS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python src/data_ingest/download_painel_fogo.py
  python src/data_ingest/download_painel_fogo.py --ano-inicio 2023 --ano-fim 2024
  python src/data_ingest/download_painel_fogo.py --layer mv_evento_filtro --sem-bbox
  python src/data_ingest/download_painel_fogo.py --count-only
  python src/data_ingest/download_painel_fogo.py --formato csv
        """,
    )
    parser.add_argument(
        "--layer", type=str, default="tb_evento_extinto",
        choices=list(LAYERS.keys()),
        help="Camada WFS (default: tb_evento_extinto)"
    )
    parser.add_argument("--ano-inicio", type=int, default=2022, help="Ano inicial (default: 2022)")
    parser.add_argument("--ano-fim", type=int, default=2024, help="Ano final (default: 2024)")
    parser.add_argument("--sem-bbox", action="store_true", help="Nao filtrar por MATOPIBA bbox")
    parser.add_argument("--formato", choices=["geojson", "csv"], default="geojson", help="Formato (default: geojson)")
    parser.add_argument("--count-only", action="store_true", help="Apenas contar eventos, sem baixar")
    parser.add_argument("--output", type=str, default=None, help="Caminho de saida")

    args = parser.parse_args()

    print("=" * 65)
    print("DOWNLOAD - PAINEL DO FOGO (CENSIPAM)")
    print("=" * 65)
    print(f"  Camada:  {args.layer} ({LAYERS[args.layer]})")
    print(f"  Periodo: {args.ano_inicio}-{args.ano_fim}")
    print(f"  Regiao:  {'Brasil inteiro' if args.sem_bbox else 'MATOPIBA (MA, TO, PI, BA)'}")
    print(f"  Formato: {args.formato.upper()}")
    print(f"  WFS:     {WFS_BASE}")
    print("-" * 65)

    # Construir filtro CQL
    bbox = None if args.sem_bbox else MATOPIBA_BBOX
    cql_filter = build_cql_filter(args.ano_inicio, args.ano_fim, bbox)
    print(f"\n  Filtro CQL: {cql_filter}")

    # Contar eventos
    print(f"\n[1] Contando eventos...")
    total = count_features(args.layer, cql_filter)
    if total is not None:
        print(f"  Eventos encontrados: {total:,}")
    else:
        print("  Nao foi possivel contar (o servidor pode nao suportar resultType=hits)")

    if args.count_only:
        # Mostrar contagem por ano tambem
        print(f"\n  Detalhamento por ano:")
        for ano in range(args.ano_inicio, args.ano_fim + 1):
            cql_ano = build_cql_filter(ano, ano, bbox)
            n = count_features(args.layer, cql_ano)
            if n is not None:
                print(f"    {ano}: {n:,} eventos")
        return

    if total == 0:
        print("\n  Nenhum evento encontrado. Verifique os filtros.")
        return

    # Download
    print(f"\n[2] Baixando dados (paginas de {PAGE_SIZE:,} eventos)...")
    df = download_all(args.layer, cql_filter, args.formato)

    if df is None or len(df) == 0:
        print("\n  [ERRO] Nenhum dado baixado.")
        return

    print(f"\n[3] Dados baixados: {len(df):,} eventos")
    print(f"  Colunas: {list(df.columns)}")

    # Estatisticas basicas
    if "persistencia_dias" in df.columns:
        print(f"\n  Persistencia (dias): media={df['persistencia_dias'].mean():.1f}, "
              f"max={df['persistencia_dias'].max()}")
    if "qtd_deteccoes" in df.columns:
        print(f"  Deteccoes por evento: media={df['qtd_deteccoes'].mean():.1f}, "
              f"max={df['qtd_deteccoes'].max()}")
    if "area_km2" in df.columns:
        print(f"  Area (km2): media={df['area_km2'].mean():.2f}, "
              f"max={df['area_km2'].max():.2f}")

    # Salvar
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "csv" if args.formato == "csv" else "csv"
        filename = f"painel_fogo_{args.layer}_{args.ano_inicio}-{args.ano_fim}_matopiba.{suffix}"
        output_path = OUTPUT_DIR / filename

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Salvo em: {output_path}")
    print(f"  Tamanho: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Resumo final
    print(f"\n{'=' * 65}")
    print(f"RESUMO")
    print(f"{'=' * 65}")
    print(f"  Eventos baixados:  {len(df):,}")
    print(f"  Arquivo:           {output_path.name}")
    print(f"  Proximo passo:     python src/models/compare_painel_fogo.py --painel {output_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
